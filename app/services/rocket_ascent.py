"""Apollo 11 (Saturn V AS-506) launch: NASA-reported flight values plus textbook physics.

Data: data/processed/rocket/*.csv, built by src/build_rocket_ascent.py from NASA/SP-2000-4029 (NTRS 20010008244)
and cross-checked against the primary report MPR-SAT-FE-69-9. NASA values are never altered here.

Physics (standard formulas, nothing fitted):
  * Two-body (Kepler) motion from one state: what the vehicle does if every engine stops at that moment.
    Air drag is ignored, which is accurate above about 60 km and optimistic below.
  * Circular orbital speed sqrt(GM/r): the speed needed to stay up at that height.
  * U.S. Standard Atmosphere 1976 pressure, 0-86 km: how much outside air is left around the rocket.
  * Earth's rotation speed at the pad: omega * r * cos(latitude).
  * Between NASA's event times the track is interpolated for drawing only (cubic Hermite with NASA's own climb
    rates as slopes); every NASA point is reproduced exactly.
Constants: WGS-84 ellipsoid, GM and rotation rate; U.S. Standard Atmosphere 1976 (NOAA/NASA/USAF).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "rocket"
ASCENT_PATH = DATA_DIR / "apollo11_ascent_events.csv"
TIMELINE_PATH = DATA_DIR / "apollo11_launch_timeline.csv"
ORBIT_PATH = DATA_DIR / "apollo11_earth_orbit.csv"
PRIMARY_PATH = DATA_DIR / "as506_primary_reference.csv"

GM_KM3_S2 = 398600.4418           # WGS-84 Earth gravitational parameter
WGS84_A_KM = 6378.137
WGS84_F = 1 / 298.257223563
WGS84_B_KM = WGS84_A_KM * (1 - WGS84_F)
EARTH_ROTATION_RAD_S = 7.2921159e-5
KARMAN_LINE_KM = 100.0
SEA_LEVEL_PA = 101325.0

# U.S. Standard Atmosphere 1976: base geopotential height (km), base temperature (K), lapse rate (K/km), base pressure (Pa)
US76_LAYERS = (
    (0.0, 288.15, -6.5, 101325.0),
    (11.0, 216.65, 0.0, 22632.06),
    (20.0, 216.65, 1.0, 5474.889),
    (32.0, 228.65, 2.8, 868.0187),
    (47.0, 270.65, 0.0, 110.9063),
    (51.0, 270.65, -2.8, 66.93887),
    (71.0, 214.65, -2.0, 3.956420),
)
US76_TOP_GEOMETRIC_KM = 86.0      # (84.852 km geopotential); above this the standard is not a simple layer formula
US76_R0_KM = 6356.766              # radius the standard uses to convert geometric to geopotential height
US76_GMR = 34.1632                 # g0 * M / R*, in K per km

# Burn phases come from NASA's printed ignition and cutoff times; propellants from MPR-SAT-FE-69-9 section 3.4.
STAGES = (
    {"stage": "S-IC", "ignition": "s_ic_ignition", "center_cutoff": "s_ic_center_cutoff",
     "cutoff": "s_ic_outboard_cutoff", "separation": "s_ic_separation", "max_g": "s_ic_max_acceleration",
     "propellant": ("kerosene (RP-1) + liquid oxygen", "煤油 (RP-1) + 液氧"), "fuel": "kerosene"},
    {"stage": "S-II", "ignition": "s_ii_ignition", "center_cutoff": "s_ii_center_cutoff",
     "cutoff": "s_ii_outboard_cutoff", "separation": "s_ii_separation", "max_g": "s_ii_max_acceleration",
     "propellant": ("liquid hydrogen + liquid oxygen", "液氢 + 液氧"), "fuel": "hydrogen"},
    {"stage": "S-IVB", "ignition": "s_ivb_ignition", "center_cutoff": None,
     "cutoff": "s_ivb_cutoff", "separation": None, "max_g": "s_ivb_max_acceleration",
     "propellant": ("liquid hydrogen + liquid oxygen", "液氢 + 液氧"), "fuel": "hydrogen"},
)
CUTOFF_EVENTS = ("s_ic_center_cutoff", "s_ic_outboard_cutoff", "s_ii_center_cutoff", "s_ii_outboard_cutoff", "s_ivb_cutoff")

EVENT_TEXT = {  # event_id -> (English, Chinese) display label
    "s_ic_start_command": ("First-stage engine start command", "第一级发动机启动指令"),
    "s_ic_ignition": ("First-stage engines ignite (on the pad)", "第一级发动机点火（仍在发射台上）"),
    "s_ic_thrust_ok": ("All first-stage engines at thrust", "第一级全部发动机推力正常"),
    "range_zero": ("Range zero (clock T = 0)", "计时零点 (T = 0)"),
    "first_motion": ("Hold-down arms released: first motion", "释放压紧臂：火箭开始移动"),
    "liftoff": ("Liftoff", "起飞"),
    "pitch_roll_start": ("Pitch and roll program starts", "开始俯仰与滚转程序"),
    "mach_1": ("Faster than sound (Mach 1)", "超过音速 (1 马赫)"),
    "max_q": ("Maximum aerodynamic pressure (Max Q)", "最大动压 (Max Q)"),
    "s_ic_center_cutoff": ("First stage: centre engine cut off", "第一级：中心发动机关机"),
    "s_ic_outboard_cutoff": ("First stage: last engines cut off, flame out", "第一级：其余发动机关机，火焰熄灭"),
    "s_ic_separation": ("First stage separates", "第一级分离"),
    "s_ii_ignition": ("Second stage ignites", "第二级点火"),
    "les_jettison": ("Launch escape tower jettisoned", "抛掉逃逸塔"),
    "s_ii_center_cutoff": ("Second stage: centre engine cut off", "第二级：中心发动机关机"),
    "s_ii_outboard_cutoff": ("Second stage: last engines cut off, flame out", "第二级：其余发动机关机，火焰熄灭"),
    "s_ii_separation": ("Second stage separates", "第二级分离"),
    "s_ivb_ignition": ("Third stage ignites", "第三级点火"),
    "s_ivb_cutoff": ("Third stage cut off: final flame out", "第三级关机：最后一次火焰熄灭"),
    "orbit_insertion": ("Earth orbit insertion: free fall begins", "进入地球轨道：开始自由落体（失重）"),
}
ANIMATION_EVENTS = ("s_ic_ignition", "liftoff", "pitch_roll_start", "mach_1", "max_q", "s_ic_center_cutoff",
                    "s_ic_outboard_cutoff", "s_ic_separation", "s_ii_ignition", "les_jettison", "s_ii_center_cutoff",
                    "s_ii_outboard_cutoff", "s_ii_separation", "s_ivb_ignition", "s_ivb_cutoff", "orbit_insertion")


@lru_cache(maxsize=1)
def load_tables() -> dict[str, pd.DataFrame]:
    """Processed NASA tables, indexed by event_id. Raises FileNotFoundError if the build has not been run."""
    return {
        "ascent": pd.read_csv(ASCENT_PATH).set_index("event_id"),
        "timeline": pd.read_csv(TIMELINE_PATH, keep_default_na=False, na_values=[""]).set_index("event_id"),
        "orbit": pd.read_csv(ORBIT_PATH).set_index("event_id"),
        "primary": pd.read_csv(PRIMARY_PATH),
    }


def ellipsoid_radius_km(geocentric_lat_deg: float) -> float:
    """Distance from Earth's centre to the WGS-84 surface at a geocentric latitude."""
    phi = math.radians(geocentric_lat_deg)
    return WGS84_A_KM * WGS84_B_KM / math.hypot(WGS84_B_KM * math.cos(phi), WGS84_A_KM * math.sin(phi))


def us76_pressure_pa(altitude_km: float) -> float | None:
    """Static air pressure of the U.S. Standard Atmosphere 1976 at a geometric altitude; None above 86 km."""
    z = max(0.0, float(altitude_km))
    h = US76_R0_KM * z / (US76_R0_KM + z)
    if z > US76_TOP_GEOMETRIC_KM + 1e-9:
        return None
    for base_h, base_t, lapse, base_p in reversed(US76_LAYERS):
        if h >= base_h:
            if lapse == 0.0:
                return base_p * math.exp(-US76_GMR * (h - base_h) / base_t)
            return base_p * (base_t / (base_t + lapse * (h - base_h))) ** (US76_GMR / lapse)
    raise AssertionError("unreachable")


def circular_speed_km_s(radius_km: float) -> float:
    return math.sqrt(GM_KM3_S2 / radius_km)


def orbit_if_engines_stop(altitude_km: float, speed_m_s: float, flight_path_angle_deg: float,
                          latitude_deg: float) -> dict:
    """Two-body result of switching every engine off at this state (space-fixed speed and angle; no air drag)."""
    r_earth = ellipsoid_radius_km(latitude_deg)
    r = r_earth + altitude_km
    v = speed_m_s / 1000.0
    gamma = math.radians(flight_path_angle_deg)
    energy = v * v / 2 - GM_KM3_S2 / r
    if energy >= 0:
        raise ValueError("escape trajectory: not a bound orbit")
    a = -GM_KM3_S2 / (2 * energy)
    p = (r * v * math.cos(gamma)) ** 2 / GM_KM3_S2
    e = math.sqrt(max(0.0, 1 - p / a))
    perigee_alt = a * (1 - e) - r_earth
    verdict = ("orbit" if perigee_alt >= KARMAN_LINE_KM else "grazes_atmosphere" if perigee_alt > 0 else "falls_back")
    return {
        "radius_km": r, "earth_radius_km": r_earth, "speed_km_s": v, "flight_path_angle_deg": flight_path_angle_deg,
        "circular_speed_km_s": circular_speed_km_s(r), "semi_major_axis_km": a, "eccentricity": e,
        "semi_latus_rectum_km": p, "apogee_altitude_km": a * (1 + e) - r_earth, "perigee_altitude_km": perigee_alt,
        "period_min": 2 * math.pi * math.sqrt(a ** 3 / GM_KM3_S2) / 60, "verdict": verdict,
    }


def _solve_kepler(mean_anomaly: float, e: float) -> float:
    ecc = mean_anomaly if e < 0.8 else math.pi
    for _ in range(60):
        step = (ecc - e * math.sin(ecc) - mean_anomaly) / (1 - e * math.cos(ecc))
        ecc -= step
        if abs(step) < 1e-13:
            break
    return ecc


def coast_track(event_id: str, max_duration_s: float = 1500.0, step_s: float = 2.0) -> dict:
    """Drag-free coast from a NASA state: time, downrange and altitude until ground impact (or max_duration_s).

    Downrange is Earth-relative: the distance travelled around Earth's centre minus the ground's own eastward
    motion along the track (first-order correction), so the curve joins the NASA track smoothly.
    """
    row = load_tables()["ascent"].loc[event_id]
    o = orbit_if_engines_stop(row["altitude_km"], row["space_fixed_velocity_m_s"],
                              row["space_fixed_flight_path_angle_deg"], row["geocentric_latitude_deg"])
    a, e, p, r0, r_e = o["semi_major_axis_km"], o["eccentricity"], o["semi_latus_rectum_km"], o["radius_km"], o["earth_radius_km"]
    climbing = row["space_fixed_flight_path_angle_deg"] >= 0
    nu0 = math.acos(min(1.0, max(-1.0, (p / r0 - 1) / e))) if e > 1e-12 else 0.0
    nu0 = nu0 if climbing else -nu0
    ecc0 = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(nu0 / 2))
    m0 = ecc0 - e * math.sin(ecc0)
    n = math.sqrt(GM_KM3_S2 / a ** 3)
    ground_speed = (EARTH_ROTATION_RAD_S * r_e * math.cos(math.radians(row["geocentric_latitude_deg"]))
                    * math.sin(math.radians(row["space_fixed_heading_deg"])))
    times = np.arange(0.0, max_duration_s + step_s / 2, step_s)
    ecc = np.array([_solve_kepler(m0 + n * t, e) for t in times])
    nu = np.unwrap(2 * np.arctan2(math.sqrt(1 + e) * np.sin(ecc / 2), math.sqrt(1 - e) * np.cos(ecc / 2)))
    nu += nu0 - nu[0]
    alt = a * (1 - e * np.cos(ecc)) - r_e
    downrange = row["range_km"] + r_e * (nu - nu0) - ground_speed * times
    impact_time = impact_range = None
    below = np.nonzero(alt <= 0)[0]
    if below.size:
        k = int(below[0])
        frac = alt[k - 1] / (alt[k - 1] - alt[k])
        impact_time = float(times[k - 1] + frac * (times[k] - times[k - 1]))
        impact_range = float(downrange[k - 1] + frac * (downrange[k] - downrange[k - 1]))
        times, alt, downrange = (np.append(times[:k], impact_time), np.append(alt[:k], 0.0),
                                 np.append(downrange[:k], impact_range))
    apex_t = (math.pi - m0) / n if climbing and m0 < math.pi else 0.0
    apex_ecc = _solve_kepler(m0 + n * apex_t, e)
    apex_nu = nu0 + (2 * math.atan2(math.sqrt(1 + e) * math.sin(apex_ecc / 2), math.sqrt(1 - e) * math.cos(apex_ecc / 2)) - nu0) % (2 * math.pi)
    return {
        "event_id": event_id, "orbit": o, "start_time_s": float(row["get_s"]),
        "time_s": (times + row["get_s"]).tolist(), "downrange_km": downrange.tolist(), "altitude_km": alt.tolist(),
        "apex_time_s": float(row["get_s"] + apex_t), "apex_altitude_km": float(a * (1 - e * math.cos(apex_ecc)) - r_e),
        "apex_downrange_km": float(row["range_km"] + r_e * (apex_nu - nu0) - ground_speed * apex_t),
        "impact_time_s": None if impact_time is None else float(row["get_s"] + impact_time),
        "impact_downrange_km": impact_range,
    }


def _knots() -> pd.DataFrame:
    """NASA ascent points with climb rate (km/s) and ground speed (km/s) derived from the printed velocities."""
    k = load_tables()["ascent"].copy()
    gamma = np.radians(k["space_fixed_flight_path_angle_deg"].astype(float))
    climb = k["space_fixed_velocity_m_s"] * np.sin(gamma) / 1000.0          # radial speed: same in both frames
    gap = k["get_s"].diff()
    climb = climb.fillna((k["altitude_km"].diff() / gap))                   # insertion row prints no angle
    horiz = np.sqrt(np.clip((k["earth_fixed_velocity_m_s"] / 1000.0) ** 2 - climb ** 2, 0, None))
    radius = np.array([ellipsoid_radius_km(lat) for lat in k["geocentric_latitude_deg"].ffill()])
    k["climb_km_s"] = climb
    k["ground_km_s"] = horiz * radius / (radius + k["altitude_km"])
    k["earth_relative_path_angle_deg"] = np.degrees(np.arcsin(np.clip(
        climb * 1000.0 / k["earth_fixed_velocity_m_s"], -1, 1)))
    return k


def burn_phases() -> list[dict]:
    """Engine-on intervals from NASA's printed ignition and cutoff times (nothing estimated)."""
    t = load_tables()["timeline"]["get_s"]
    a = load_tables()["ascent"]
    g = load_tables()["timeline"]["value_g"]
    out = []
    for s in STAGES:
        out.append({
            "stage": s["stage"], "fuel": s["fuel"], "propellant": s["propellant"],
            "ignition_s": float(t[s["ignition"]]), "cutoff_s": float(t[s["cutoff"]]),
            "center_cutoff_s": None if s["center_cutoff"] is None else float(t[s["center_cutoff"]]),
            "separation_s": None if s["separation"] is None else float(t[s["separation"]]),
            "cutoff_event": s["cutoff"], "cutoff_altitude_km": float(a.at[s["cutoff"], "altitude_km"]),
            "cutoff_speed_m_s": float(a.at[s["cutoff"], "space_fixed_velocity_m_s"]),
            "max_acceleration_g": float(g[s["max_g"]]),
        })
    return out


def ascent_track(t_start: float = -10.0, t_end: float = 760.0, step_s: float = 0.5) -> pd.DataFrame:
    """Per-step track for drawing. NASA points are matched exactly; values in between are interpolated."""
    k = _knots()
    t_k = k["get_s"].to_numpy(float)
    times = np.round(np.arange(t_start, t_end + step_s / 2, step_s), 3)
    inside = np.clip(times, t_k[0], t_k[-1])
    alt = CubicHermiteSpline(t_k, k["altitude_km"].to_numpy(float), k["climb_km_s"].to_numpy(float))(inside)
    rng = CubicHermiteSpline(t_k, k["range_km"].to_numpy(float), k["ground_km_s"].to_numpy(float))(inside)
    speed = PchipInterpolator(t_k, k["space_fixed_velocity_m_s"].to_numpy(float))(inside)
    after = times > t_k[-1]                      # past orbit insertion: coast at insertion speed and height
    rng[after] = k["range_km"].iloc[-1] + k["ground_km_s"].iloc[-1] * (times[after] - t_k[-1])
    timeline = load_tables()["timeline"]["get_s"]
    angle_t = [t_k[0], float(timeline["pitch_roll_start"]), *t_k[1:]]   # vertical until the pitch program starts
    angle_v = [90.0, 90.0, *k["earth_relative_path_angle_deg"].iloc[1:].to_numpy(float)]
    path_angle = np.interp(times, angle_t, angle_v)
    pressure = [us76_pressure_pa(z) for z in alt]
    phases = burn_phases()
    stage_idx = np.full(times.shape, -1)
    engines = np.zeros(times.shape, dtype=int)  # 0 off, 1 some engines (centre off), 2 all engines
    for i, ph in enumerate(phases):
        on = (times >= ph["ignition_s"]) & (times < ph["cutoff_s"])
        stage_idx[on] = i
        engines[on] = 2
        if ph["center_cutoff_s"] is not None:
            engines[on & (times >= ph["center_cutoff_s"])] = 1
    return pd.DataFrame({
        "time_s": times, "altitude_km": alt, "downrange_km": rng, "speed_m_s": speed,
        "path_angle_deg": path_angle, "pressure_pa": pressure, "stage_index": stage_idx, "engines": engines,
        "interpolated": ~np.isin(times, t_k),
    })


@lru_cache(maxsize=8)
def crossing_time_s(altitude_km: float = KARMAN_LINE_KM) -> float:
    """First time the interpolated track reaches an altitude (between NASA points, so approximate)."""
    track = ascent_track(t_start=0.0, step_s=0.05)
    above = np.nonzero(track["altitude_km"].to_numpy() >= altitude_km)[0]
    return float(track["time_s"].iloc[above[0]]) if above.size else float("nan")


def validation_checks() -> list[dict]:
    """Our physics against numbers NASA printed. Every row states its source, the difference and the allowed difference.

    The allowed differences reflect what this simple model leaves out (air drag, Earth's oblateness, the
    first-order rotation correction to downrange); the actual differences are always shown alongside.
    """
    tables = load_tables()
    a, orbit, primary = tables["ascent"], tables["orbit"], tables["primary"]
    ref = lambda ev, par: float(primary[(primary["event"] == ev) & (primary["parameter"] == par)]["actual"].iloc[0])  # noqa: E731
    lo = a.loc["liftoff"]
    pad = (EARTH_ROTATION_RAD_S * ellipsoid_radius_km(lo["geocentric_latitude_deg"]) * 1000.0
           * math.cos(math.radians(lo["geocentric_latitude_deg"])))
    eoi = a.loc["orbit_insertion"]
    # Insertion prints no latitude or angle; the period depends only on height and speed, so the cutoff latitude 10 s
    # earlier is used for the local radius (an effect of well under 0.01 min).
    period = orbit_if_engines_stop(eoi["altitude_km"], eoi["space_fixed_velocity_m_s"], 0.0,
                                   a.at["s_ivb_cutoff", "geocentric_latitude_deg"])["period_min"]
    s_ic, s_ii = coast_track("s_ic_separation"), coast_track("s_ii_separation")
    rows = [
        ("pad_rotation", "Speed at liftoff = Earth's spin at the pad", "m/s", pad, float(lo["space_fixed_velocity_m_s"]), 0.5,
         "SP-4029 p.103"),
        ("orbit_period", "Parking-orbit period from insertion height and speed", "min", period,
         float(orbit.at["orbit_insertion", "period_min"]), 0.05, "SP-4029 p.103"),
        ("s_ic_apex_time", "Spent first stage: time of highest point", "s", s_ic["apex_time_s"],
         ref("Apex: S-IC Stage", "Range Time"), 2.0, "MPR-SAT-FE-69-9 Table 4-1"),
        ("s_ic_apex_altitude", "Spent first stage: highest point", "km", s_ic["apex_altitude_km"],
         ref("Apex: S-IC Stage", "Altitude"), 1.5, "MPR-SAT-FE-69-9 Table 4-1"),
        ("s_ic_apex_range", "Spent first stage: distance downrange at highest point", "km", s_ic["apex_downrange_km"],
         ref("Apex: S-IC Stage", "Surface Range"), 5.0, "MPR-SAT-FE-69-9 Table 4-1"),
        ("s_ii_apex_time", "Spent second stage: time of highest point", "s", s_ii["apex_time_s"],
         ref("Apex: S-II Stage", "Range Time"), 2.0, "MPR-SAT-FE-69-9 Table 4-1"),
        ("s_ii_apex_altitude", "Spent second stage: highest point", "km", s_ii["apex_altitude_km"],
         ref("Apex: S-II Stage", "Altitude"), 1.5, "MPR-SAT-FE-69-9 Table 4-1"),
        ("s_ii_apex_range", "Spent second stage: distance downrange at highest point", "km", s_ii["apex_downrange_km"],
         ref("Apex: S-II Stage", "Surface Range"), 5.0, "MPR-SAT-FE-69-9 Table 4-1"),
    ]
    return [{"check_id": cid, "check": text, "unit": unit, "computed": round(comp, 3), "nasa": nasa,
             "difference": round(comp - nasa, 3), "tolerance": tol, "passed": abs(comp - nasa) <= tol, "nasa_source": src}
            for cid, text, unit, comp, nasa, tol, src in rows]
