"""Build the Apollo 11 (Saturn V AS-506) launch tables from NASA's published flight values.

Source (NASA-hosted; checksum registered in config/data_sources.json, dataset APOLLO11_ASCENT):
  NASA/SP-2000-4029, "Apollo by the Numbers: A Statistical Reference" (R. W. Orloff, 2000), NTRS 20010008244
    printed page 103 (PDF page 113): "Apollo 11 Ascent Phase" and "Apollo 11 Earth Orbit Phase"
    printed page 105 (PDF page 115): "Apollo 11 Timeline" (launch rows only)
  The book is a 255 MB scanned PDF and is not stored. The table pages are stored as images in
  data/raw/nasa_ntrs/, and their values were transcribed by hand, exactly as printed, into data/transcribed/.

Cross-check reference (not displayed as data): the primary report MPR-SAT-FE-69-9, "Saturn V Launch Vehicle
Flight Evaluation Report AS-506 Apollo 11 Mission" (NASA MSFC, 1969), Table 4-1, transcribed selectively.

Checks (recomputed by src/provenance_audit.py; results also written to the metadata file):
  * every number transcribed from page 103 appears in NTRS's own OCR text of that page; times are compared
    on their minute/second digits because the OCR drops colons;
  * times shared by the ascent table and the timeline are identical, and each cutoff time minus its burn
    duration equals the ignition time printed in the timeline;
  * values shared with the primary report agree within the primary report's printed precision.

Outputs (values are only unit-converted; nothing is fitted, smoothed or filled in):
  data/processed/rocket/apollo11_ascent_events.csv
  data/processed/rocket/apollo11_launch_timeline.csv
  data/processed/rocket/apollo11_earth_orbit.csv
  data/processed/rocket/as506_primary_reference.csv
  data/processed/rocket/apollo11_launch.metadata.json

Run:  python -m src.build_rocket_ascent
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "data_sources.json"
TRANSCRIBED_DIR = ROOT / "data" / "transcribed"
ASCENT_IN = TRANSCRIBED_DIR / "apollo11_sp4029_p103_ascent_phase.csv"
ORBIT_IN = TRANSCRIBED_DIR / "apollo11_sp4029_p103_earth_orbit_phase.csv"
TIMELINE_IN = TRANSCRIBED_DIR / "apollo11_sp4029_p105_timeline_launch.csv"
PRIMARY_IN = TRANSCRIBED_DIR / "as506_mpr-sat-fe-69-9_table4-1_selected.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "rocket"
ASCENT_OUT = OUTPUT_DIR / "apollo11_ascent_events.csv"
TIMELINE_OUT = OUTPUT_DIR / "apollo11_launch_timeline.csv"
ORBIT_OUT = OUTPUT_DIR / "apollo11_earth_orbit.csv"
PRIMARY_OUT = OUTPUT_DIR / "as506_primary_reference.csv"
METADATA_OUT = OUTPUT_DIR / "apollo11_launch.metadata.json"

NMI_TO_KM = 1.852  # international nautical mile (exact)
FT_TO_M = 0.3048   # international foot (exact)
OCR_START, OCR_END = "Apollo I 1 Ascent Phase", "Translunar Phase"  # NTRS OCR text of printed page 103
FLOAT_FORMAT = "%.10g"

ASCENT_IDS = {
    "Liftoff": "liftoff",
    "Mach 1 achieved": "mach_1",
    "Maximum dynamic pressure": "max_q",
    "S-IC center engine cutoff": "s_ic_center_cutoff",
    "S-IC outboard engine cutoff": "s_ic_outboard_cutoff",
    "S-IC/S-II separation": "s_ic_separation",
    "S-II center engine cutoff": "s_ii_center_cutoff",
    "S-II outboard engine cutoff": "s_ii_outboard_cutoff",
    "S-II/S-IVB separation": "s_ii_separation",
    "S-IVB 1st burn cutoff": "s_ivb_cutoff",
    "Earth orbit insertion": "orbit_insertion",
}
ORBIT_IDS = {
    "Earth orbit insertion": "orbit_insertion",
    "S-IVB 2nd burn ignition": "s_ivb_2nd_ignition",
    "S-IVB 2nd burn cutoff": "s_ivb_2nd_cutoff",
}
TIMELINE_IDS = {  # printed event -> (event_id, kind)
    "S-IC engine start command.": ("s_ic_start_command", "command"),
    "S-IC engine ignition (#5).": ("s_ic_ignition", "ignition"),
    "All S-IC engines thrust OK.": ("s_ic_thrust_ok", "milestone"),
    "Range zero.": ("range_zero", "milestone"),
    "All holddown arms released (1st motion).": ("first_motion", "milestone"),
    "Liftoff (umbilical disconnected) (1.07 g).": ("liftoff", "milestone"),
    "Tower clearance yaw maneuver started.": ("yaw_start", "maneuver"),
    "Yaw maneuver ended.": ("yaw_end", "maneuver"),
    "Pitch and roll maneuver started.": ("pitch_roll_start", "maneuver"),
    "Roll maneuver ended.": ("roll_end", "maneuver"),
    "Mach 1 achieved.": ("mach_1", "milestone"),
    "Maximum dynamic pressure (735.17 lb/ft2).": ("max_q", "milestone"),
    "Maximum bending moment (33,200,000 lbf-in).": ("max_bending_moment", "milestone"),
    "S-IC center engine cutoff command.": ("s_ic_center_cutoff", "cutoff"),
    "Pitch maneuver ended.": ("pitch_end", "maneuver"),
    "S-IC outboard engine cutoff.": ("s_ic_outboard_cutoff", "cutoff"),
    "S-IC maximum total inertial acceleration (3.94 g).": ("s_ic_max_acceleration", "max_acceleration"),
    "S-IC maximum Earth-fixed velocity. S-IC/S-II separation command.": ("s_ic_separation", "separation"),
    "S-II engine start command.": ("s_ii_start_command", "command"),
    "S-II ignition.": ("s_ii_ignition", "ignition"),
    "S-II aft interstage jettisoned.": ("s_ii_interstage_jettison", "jettison"),
    "Launch escape tower jettisoned.": ("les_jettison", "jettison"),
    "Iterative guidance mode initiated.": ("guidance_mode", "maneuver"),
    "S-IC apex.": ("s_ic_apex", "apex"),
    "S-II center engine cutoff.": ("s_ii_center_cutoff", "cutoff"),
    "S-II maximum total inertial acceleration (1.82 g).": ("s_ii_max_acceleration", "max_acceleration"),
    "S-IC impact (theoretical).": ("s_ic_impact", "impact"),
    "S-II outboard engine cutoff.": ("s_ii_outboard_cutoff", "cutoff"),
    "S-II maximum Earth-fixed velocity. S-II/S-IVB separation command.": ("s_ii_separation", "separation"),
    "S-IVB 1st burn start command.": ("s_ivb_start_command", "command"),
    "S-IVB 1st burn ignition.": ("s_ivb_ignition", "ignition"),
    "S-IVB ullage case jettisoned.": ("s_ivb_ullage_jettison", "jettison"),
    "S-II apex.": ("s_ii_apex", "apex"),
    "S-IVB 1st burn cutoff.": ("s_ivb_cutoff", "cutoff"),
    "S-IVB 1st burn maximum total inertial acceleration (0.69 g).": ("s_ivb_max_acceleration", "max_acceleration"),
    "Earth orbit insertion. S-IVB 1st burn maximum Earth-fixed velocity.": ("orbit_insertion", "orbit"),
    "Maneuver to local horizontal attitude started.": ("local_horizontal", "maneuver"),
    "Orbital navigation started.": ("orbital_navigation", "maneuver"),
    "S-II impact (theoretical).": ("s_ii_impact", "impact"),
}
# Ignition time printed in the timeline vs cutoff time minus burn duration from the ascent table.
IGNITION_CHECKS = {
    "s_ic_ignition": ("s_ic_center_cutoff", "s_ic_outboard_cutoff"),
    "s_ii_ignition": ("s_ii_center_cutoff", "s_ii_outboard_cutoff"),
    "s_ivb_ignition": ("s_ivb_cutoff",),
}
# (primary event, parameter, unit) -> (table, event_id, column, factor applied to our value)
PRIMARY_COMPARISONS = [
    (("First Motion", "Range Time", "sec"), ("timeline", "first_motion", "get_s", 1.0)),
    (("Mach 1", "Range Time", "sec"), ("ascent", "mach_1", "get_s", 1.0)),
    (("Mach 1", "Altitude", "km"), ("ascent", "mach_1", "altitude_km", 1.0)),
    (("Maximum Dynamic Pressure", "Range Time", "sec"), ("ascent", "max_q", "get_s", 1.0)),
    (("Maximum Dynamic Pressure", "Altitude", "km"), ("ascent", "max_q", "altitude_km", 1.0)),
    (("Maximum Total Inertial Acceleration: S-IC", "Range Time", "sec"), ("timeline", "s_ic_max_acceleration", "get_s", 1.0)),
    (("Maximum Total Inertial Acceleration: S-IC", "Acceleration", "g"), ("timeline", "s_ic_max_acceleration", "value_g", 1.0)),
    (("Maximum Total Inertial Acceleration: S-II", "Range Time", "sec"), ("timeline", "s_ii_max_acceleration", "get_s", 1.0)),
    (("Maximum Total Inertial Acceleration: S-II", "Acceleration", "g"), ("timeline", "s_ii_max_acceleration", "value_g", 1.0)),
    (("Maximum Total Inertial Acceleration: S-IVB 1st Burn", "Range Time", "sec"), ("timeline", "s_ivb_max_acceleration", "get_s", 1.0)),
    (("Maximum Total Inertial Acceleration: S-IVB 1st Burn", "Acceleration", "g"), ("timeline", "s_ivb_max_acceleration", "value_g", 1.0)),
    (("Maximum Earth-Fixed Velocity: S-IC", "Range Time", "sec"), ("ascent", "s_ic_separation", "get_s", 1.0)),
    (("Maximum Earth-Fixed Velocity: S-IC", "Velocity", "m/s"), ("ascent", "s_ic_separation", "earth_fixed_velocity_m_s", 1.0)),
    (("Maximum Earth-Fixed Velocity: S-II", "Range Time", "sec"), ("ascent", "s_ii_separation", "get_s", 1.0)),
    (("Maximum Earth-Fixed Velocity: S-II", "Velocity", "m/s"), ("ascent", "s_ii_separation", "earth_fixed_velocity_m_s", 1.0)),
    (("Maximum Earth-Fixed Velocity: S-IVB 1st Burn", "Range Time", "sec"), ("ascent", "orbit_insertion", "get_s", 1.0)),
    (("Maximum Earth-Fixed Velocity: S-IVB 1st Burn", "Velocity", "m/s"), ("ascent", "orbit_insertion", "earth_fixed_velocity_m_s", 1.0)),
    (("Apex: S-IC Stage", "Range Time", "sec"), ("timeline", "s_ic_apex", "get_s", 1.0)),
    (("Apex: S-II Stage", "Range Time", "sec"), ("timeline", "s_ii_apex", "get_s", 1.0)),
]
NOTES = [
    {"code": "HEADER_UNIT_MISPRINT",
     "detail": "The ascent table header prints 'Event Duration (deg E)'. The values are seconds: the Earth-orbit "
               "table on the same page labels the same column '(sec)', and each cutoff time minus its duration "
               "equals the ignition time printed in the timeline."},
    {"code": "SUSPECT_VALUE",
     "detail": "S-IC center engine cutoff longitude -81.1517 deg E lies west of the launch pad (-80.6041) although "
               "every other point moves east. Kept as printed; longitude is not used by any calculation."},
]


def get_seconds(text: str) -> float:
    """'000:02:41.63' or '-000:00:06.40' (hhh:mm:ss.ss, as printed) -> seconds."""
    match = re.fullmatch(r"(-?)(\d{3}):(\d{2}):(\d{2}(?:\.\d+)?)", str(text).strip())
    if not match:
        raise ValueError(f"not a GET value: {text!r}")
    seconds = int(match.group(2)) * 3600 + int(match.group(3)) * 60 + float(match.group(4))
    return round(-seconds if match.group(1) else seconds, 3)


def number(text) -> float | None:
    """Printed number ('1,340.7', '-80.6041', blank) -> float or None."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or str(text).strip() == "":
        return None
    return float(str(text).replace(",", ""))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Repository-relative name; just the file name when building into a temporary folder (audit rebuilds)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def official_text_entry() -> dict:
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return sources["datasets"]["APOLLO11_ASCENT"]["official_files"][0]


def read_transcriptions() -> dict[str, pd.DataFrame]:
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731 - keep printed strings
    return {"ascent": read(ASCENT_IN), "orbit": read(ORBIT_IN), "timeline": read(TIMELINE_IN), "primary": read(PRIMARY_IN)}


def convert(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Printed values -> SI tables. Raises on any event name not in the id maps (no silent guessing)."""
    a = raw["ascent"]
    unknown = sorted(set(a["event"]) - set(ASCENT_IDS)) + sorted(set(raw["timeline"]["event"]) - set(TIMELINE_IDS))
    unknown += sorted(set(raw["orbit"]["event"]) - set(ORBIT_IDS))
    if unknown:
        raise ValueError(f"Unmapped printed event names: {unknown}")

    def col(frame: pd.DataFrame, name: str, factor: float = 1.0) -> list:
        return [None if number(v) is None else number(v) * factor for v in frame[name]]

    ascent = pd.DataFrame({
        "event_id": [ASCENT_IDS[e] for e in a["event"]],
        "event": a["event"],
        "commanded_time_only": a["footnote"].eq("3"),
        "get_s": [get_seconds(v) for v in a["get_hhh_mm_ss"]],
        "altitude_km": col(a, "altitude_n_mi", NMI_TO_KM),
        "range_km": col(a, "range_n_mi", NMI_TO_KM),
        "earth_fixed_velocity_m_s": col(a, "earth_fixed_velocity_ft_s", FT_TO_M),
        "space_fixed_velocity_m_s": col(a, "space_fixed_velocity_ft_s", FT_TO_M),
        "event_duration_s": col(a, "event_duration_s"),
        "geocentric_latitude_deg": col(a, "geocentric_latitude_deg_n"),
        "longitude_deg": col(a, "longitude_deg_e"),
        "space_fixed_flight_path_angle_deg": col(a, "space_fixed_flight_path_angle_deg"),
        "space_fixed_heading_deg": col(a, "space_fixed_heading_angle_deg_e_of_n"),
        "source": [f"NASA/SP-2000-4029 p.103, Apollo 11 Ascent Phase, row {i + 1}" for i in range(len(a))],
    })

    t = raw["timeline"]
    flagged = {"s_ic_apex": "CONFLICTS_WITH_PRIMARY_REPORT: MPR-SAT-FE-69-9 Table 4-1 gives 269.1 s; not used"}
    timeline = pd.DataFrame({
        "event_id": [TIMELINE_IDS[e][0] for e in t["event"]],
        "kind": [TIMELINE_IDS[e][1] for e in t["event"]],
        "event": t["event"],
        "get_s": [get_seconds(v) for v in t["get_hhh_mm_ss"]],
        "value_g": [float(m.group(1)) if (m := re.search(r"\(([\d.]+) g\)", e)) else None for e in t["event"]],
        "flag": [flagged.get(TIMELINE_IDS[e][0], "") for e in t["event"]],
        "source": [f"NASA/SP-2000-4029 p.105, Apollo 11 Timeline, launch row {i + 1}" for i in range(len(t))],
    })

    o = raw["orbit"]
    orbit = pd.DataFrame({
        "event_id": [ORBIT_IDS[e] for e in o["event"]],
        "event": o["event"],
        "get_s": [get_seconds(v) for v in o["get_hhh_mm_ss"]],
        "space_fixed_velocity_m_s": col(o, "space_fixed_velocity_ft_s", FT_TO_M),
        "event_duration_s": col(o, "event_duration_s"),
        "velocity_change_m_s": col(o, "velocity_change_ft_s", FT_TO_M),
        "apogee_km": col(o, "apogee_n_mi", NMI_TO_KM),
        "perigee_km": col(o, "perigee_n_mi", NMI_TO_KM),
        "period_min": col(o, "period_min"),
        "inclination_deg": col(o, "inclination_deg"),
        "source": [f"NASA/SP-2000-4029 p.103, Apollo 11 Earth Orbit Phase, row {i + 1}" for i in range(len(o))],
    })

    p = raw["primary"]
    primary = pd.DataFrame({
        "event": p["event"], "parameter": p["parameter"], "unit": p["unit"],
        "actual": col(p, "actual"), "nominal": col(p, "nominal"),
        "actual_as_printed": p["actual"],
        "source": "MPR-SAT-FE-69-9 (NASA MSFC, 1969) Table 4-1, p. 4-7",
    })
    return {"ascent": ascent, "timeline": timeline, "orbit": orbit, "primary": primary}


def _ocr_block(text: str) -> str:
    start = text.index(OCR_START)
    return text[start:text.index(OCR_END, start)]


def check_against_ocr(raw: dict[str, pd.DataFrame], official_text: str) -> dict:
    """Every page-103 number must appear in NTRS's OCR of that page; times are matched on mm:ss.cc digits."""
    block = _ocr_block(official_text)
    lines = [ln.strip() for ln in block.splitlines()]
    # A GET line has hhh, mm, ss and hundredths: at least 8 digits even after the OCR drops some characters.
    is_time = lambda ln: re.fullmatch(r"[\d :]+\.\d\d", ln) is not None and len(re.sub(r"\D", "", ln)) >= 8  # noqa: E731
    time_digits = {re.sub(r"\D", "", ln)[-6:] for ln in lines if is_time(ln)}
    tokens = {tok for ln in lines if not is_time(ln) for tok in re.findall(r"-?\d[\d,]*\.\d+|-?\d[\d,]*", ln)}

    values, times = [], []
    for name in ("ascent", "orbit"):
        frame = raw[name]
        for column in frame.columns:
            if column in ("event", "footnote"):
                continue
            for printed in frame[column]:
                if printed == "":
                    continue
                (times if column == "get_hhh_mm_ss" else values).append(printed)
    missing_values = sorted({v for v in values if v not in tokens})
    missing_times = sorted({t for t in times if re.sub(r"\D", "", t)[-6:] not in time_digits})
    decimal_tokens = {tok for tok in tokens if "." in tok}
    return {
        "values_checked": len(values), "values_missing_from_ocr": missing_values,
        "times_checked": len(times), "times_missing_from_ocr": missing_times,
        "ocr_decimal_tokens_not_transcribed": sorted(decimal_tokens - set(values)),
        "passed": not missing_values and not missing_times,
    }


def check_internal_consistency(tables: dict[str, pd.DataFrame]) -> dict:
    ascent = tables["ascent"].set_index("event_id")
    timeline = tables["timeline"].set_index("event_id")
    shared = sorted(set(ascent.index) & set(timeline.index))
    time_mismatches = [{"event_id": e, "ascent_s": float(ascent.at[e, "get_s"]), "timeline_s": float(timeline.at[e, "get_s"])}
                       for e in shared if abs(ascent.at[e, "get_s"] - timeline.at[e, "get_s"]) > 1e-6]
    ignition = []
    for ign, cutoffs in IGNITION_CHECKS.items():
        printed = float(timeline.at[ign, "get_s"])
        implied = [round(float(ascent.at[c, "get_s"] - ascent.at[c, "event_duration_s"]), 3) for c in cutoffs]
        ignition.append({"ignition_event": ign, "timeline_s": printed, "cutoff_minus_duration_s": implied,
                         "agrees": all(abs(i - printed) < 0.005 for i in implied)})
    order_ok = bool(ascent["get_s"].is_monotonic_increasing and timeline["get_s"].is_monotonic_increasing)
    return {"shared_times_checked": len(shared), "shared_time_mismatches": time_mismatches,
            "ignition_vs_burn_duration": ignition, "times_in_order": order_ok,
            "passed": not time_mismatches and all(i["agrees"] for i in ignition) and order_ok}


def check_against_primary(tables: dict[str, pd.DataFrame]) -> dict:
    """Agreement within half a unit of the primary report's last printed digit."""
    primary = tables["primary"]
    frames = {"ascent": tables["ascent"].set_index("event_id"), "timeline": tables["timeline"].set_index("event_id")}
    rows = []
    for (event, parameter, unit), (table, event_id, column, factor) in PRIMARY_COMPARISONS:
        ref = primary[(primary["event"] == event) & (primary["parameter"] == parameter) & (primary["unit"] == unit)].iloc[0]
        printed = str(ref["actual_as_printed"]).replace(",", "")
        decimals = len(printed.split(".")[1]) if "." in printed else 0
        ours = float(frames[table].at[event_id, column]) * factor
        diff = ours - float(ref["actual"])
        rows.append({"primary": f"{event} / {parameter} ({unit})", "primary_value": float(ref["actual"]),
                     "ours": round(ours, 6), "difference": round(diff, 6),
                     "agrees": abs(diff) <= 0.5 * 10 ** -decimals + 1e-9, "our_source": f"{table}:{event_id}.{column}"})
    conflicts = [r for r in rows if not r["agrees"]]
    return {"compared": len(rows), "agree": len(rows) - len(conflicts), "conflicts": conflicts, "rows": rows}


def run_checks(raw: dict[str, pd.DataFrame] | None = None) -> dict:
    raw = raw or read_transcriptions()
    tables = convert(raw)
    entry = official_text_entry()
    official = ROOT / entry["path"]
    checks = {"official_text": {"path": entry["path"], "present": official.exists()}}
    if official.exists():
        digest = sha256(official)
        checks["official_text"].update({"sha256": digest, "matches_registry": digest == entry["sha256"]})
        checks["ocr"] = check_against_ocr(raw, official.read_text(encoding="utf-8"))
    checks["internal"] = check_internal_consistency(tables)
    checks["primary_report"] = check_against_primary(tables)
    return checks


def build() -> dict:
    raw = read_transcriptions()
    tables = convert(raw)
    checks = run_checks(raw)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path in ((tables["ascent"], ASCENT_OUT), (tables["timeline"], TIMELINE_OUT),
                        (tables["orbit"], ORBIT_OUT), (tables["primary"], PRIMARY_OUT)):
        frame.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    conflicts = checks["primary_report"]["conflicts"]
    notes = NOTES + [{"code": "SECONDARY_SOURCE_COPYING_ERROR",
                      "detail": f"{c['primary']}: SP-4029 timeline gives {c['ours']} s, the primary report gives "
                                f"{c['primary_value']} s. The timeline value is kept as printed, flagged and not used."}
                     for c in conflicts if c["our_source"] == "timeline:s_ic_apex.get_s"]
    metadata = {
        "built_by": "src/build_rocket_ascent.py",
        "dataset_key": "APOLLO11_ASCENT",
        "unit_conversions": {"n mi -> km": NMI_TO_KM, "ft -> m (and ft/s -> m/s)": FT_TO_M},
        "transcriptions": {_rel(p): sha256(p) for p in (ASCENT_IN, ORBIT_IN, TIMELINE_IN, PRIMARY_IN)},
        "outputs": {_rel(p): sha256(p) for p in (ASCENT_OUT, TIMELINE_OUT, ORBIT_OUT, PRIMARY_OUT)},
        "rows": {"ascent": len(tables["ascent"]), "timeline": len(tables["timeline"]),
                 "orbit": len(tables["orbit"]), "primary_reference": len(tables["primary"])},
        "timeline_rows_cross_checked": sorted(
            set(tables["timeline"]["event_id"]) & (set(tables["ascent"]["event_id"]) | set(IGNITION_CHECKS)
                                                  | {"first_motion", "s_ic_max_acceleration", "s_ii_max_acceleration",
                                                     "s_ivb_max_acceleration", "s_ic_apex", "s_ii_apex"})),
        "checks": checks,
        "notes": notes,
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return metadata


if __name__ == "__main__":
    meta = build()
    c = meta["checks"]
    print("official text :", c["official_text"])
    if "ocr" in c:
        print("OCR check     :", "PASS" if c["ocr"]["passed"] else "FAIL",
              f"({c['ocr']['values_checked']} values, {c['ocr']['times_checked']} times;",
              f"missing values {c['ocr']['values_missing_from_ocr']}, missing times {c['ocr']['times_missing_from_ocr']};",
              f"OCR tokens not transcribed {c['ocr']['ocr_decimal_tokens_not_transcribed']})")
    print("internal      :", "PASS" if c["internal"]["passed"] else "FAIL", c["internal"]["ignition_vs_burn_duration"])
    pr = c["primary_report"]
    print(f"primary report: {pr['agree']}/{pr['compared']} agree; conflicts: "
          f"{[(x['primary'], x['ours'], x['primary_value']) for x in pr['conflicts']]}")
    print("wrote", ", ".join(meta["outputs"]))
