"""Apollo 11 launch data and physics: every number traces to NASA's pages, and the physics reproduces NASA's values."""

import math
from pathlib import Path

import pytest

from app.services import rocket_ascent as rocket
from src import build_rocket_ascent as builder
from src import fetch_nasa_psi_tables

ROOT = Path(__file__).resolve().parents[1]


def test_official_ntrs_text_matches_registry():
    entry = next(e for e in fetch_nasa_psi_tables.official_files() if e["dataset_key"] == "APOLLO11_ASCENT")
    ok, why = fetch_nasa_psi_tables.verify_local(entry)
    assert ok, why


def test_every_transcribed_number_is_in_nasas_own_ocr_of_the_page():
    ocr = builder.run_checks()["ocr"]
    assert ocr["passed"], ocr
    assert ocr["values_checked"] == 99 and ocr["times_checked"] == 14
    assert ocr["ocr_decimal_tokens_not_transcribed"] == []  # nothing on the page was skipped


def test_nasa_tables_agree_with_each_other():
    internal = builder.run_checks()["internal"]
    assert internal["passed"] and internal["shared_times_checked"] == 11
    for row in internal["ignition_vs_burn_duration"]:  # cutoff - burn duration == printed ignition time
        assert row["agrees"], row


def test_primary_report_agrees_except_the_documented_copying_error():
    primary = builder.run_checks()["primary_report"]
    assert [c["our_source"] for c in primary["conflicts"]] == ["timeline:s_ic_apex.get_s"]
    assert primary["agree"] == primary["compared"] - 1


def test_processed_tables_rebuild_byte_for_byte(tmp_path, monkeypatch):
    outputs = (builder.ASCENT_OUT, builder.TIMELINE_OUT, builder.ORBIT_OUT, builder.PRIMARY_OUT)
    for name in ("ASCENT_OUT", "TIMELINE_OUT", "ORBIT_OUT", "PRIMARY_OUT"):
        monkeypatch.setattr(builder, name, tmp_path / getattr(builder, name).name)
    monkeypatch.setattr(builder, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(builder, "METADATA_OUT", tmp_path / "meta.json")
    builder.build()
    for committed in outputs:
        assert (tmp_path / committed.name).read_bytes() == committed.read_bytes(), committed.name


def test_unit_conversions_are_exact_definitions():
    a = rocket.load_tables()["ascent"]
    assert a.at["liftoff", "altitude_km"] == pytest.approx(0.032 * 1.852, abs=1e-12)
    assert a.at["s_ivb_cutoff", "space_fixed_velocity_m_s"] == pytest.approx(25561.6 * 0.3048, abs=1e-9)
    assert a.at["s_ic_outboard_cutoff", "get_s"] == pytest.approx(161.63)


def test_standard_atmosphere_reproduces_its_published_layer_pressures():
    geometric = lambda h: rocket.US76_R0_KM * h / (rocket.US76_R0_KM - h)  # noqa: E731 (geopotential -> geometric)
    assert rocket.us76_pressure_pa(0.0) == pytest.approx(101325.0)
    assert rocket.us76_pressure_pa(geometric(11.0)) == pytest.approx(22632.06, rel=1e-5)
    assert rocket.us76_pressure_pa(geometric(47.0)) == pytest.approx(110.9063, rel=1e-5)
    assert rocket.us76_pressure_pa(86.0) == pytest.approx(0.3734, rel=2e-3)
    assert rocket.us76_pressure_pa(87.0) is None  # outside the simple layer formulas: shown as "< 0.0004 %"


def test_ellipsoid_and_circular_orbit_sanity():
    assert rocket.ellipsoid_radius_km(0.0) == pytest.approx(6378.137)
    assert rocket.ellipsoid_radius_km(90.0) == pytest.approx(6356.752, abs=1e-3)
    r = rocket.ellipsoid_radius_km(0.0) + 200.0
    o = rocket.orbit_if_engines_stop(200.0, rocket.circular_speed_km_s(r) * 1000, 0.0, 0.0)
    assert o["apogee_altitude_km"] == pytest.approx(200.0, abs=1e-6)
    assert o["perigee_altitude_km"] == pytest.approx(200.0, abs=1e-6)
    assert o["period_min"] == pytest.approx(2 * math.pi * math.sqrt(r ** 3 / rocket.GM_KM3_S2) / 60)
    assert o["verdict"] == "orbit"


def test_physics_reproduces_nasa_numbers():
    checks = rocket.validation_checks()
    assert len(checks) == 8
    for c in checks:
        assert c["passed"], c


def test_what_if_verdicts_match_the_real_flight():
    verdicts = {ev: rocket.coast_track(ev)["orbit"]["verdict"] for ev in rocket.CUTOFF_EVENTS}
    assert verdicts.pop("s_ivb_cutoff") == "orbit"  # the real final cutoff reaches orbit
    assert set(verdicts.values()) == {"falls_back"}  # any earlier cutoff falls back to Earth
    first = rocket.coast_track("s_ic_outboard_cutoff")
    assert first["apex_altitude_km"] > rocket.KARMAN_LINE_KM  # reaches space briefly, still falls back
    assert first["impact_time_s"] > first["apex_time_s"] > first["start_time_s"]


def test_drawing_track_passes_through_every_nasa_point():
    a = rocket.load_tables()["ascent"]
    for event_id, row in a.iterrows():
        point = rocket.ascent_track(t_start=row["get_s"], t_end=row["get_s"], step_s=1.0).iloc[0]
        assert point["altitude_km"] == pytest.approx(row["altitude_km"], abs=1e-9), event_id
        assert point["downrange_km"] == pytest.approx(row["range_km"], abs=1e-9), event_id
        assert point["speed_m_s"] == pytest.approx(row["space_fixed_velocity_m_s"], abs=1e-9), event_id


def test_engine_states_follow_nasa_ignition_and_cutoff_times():
    track = rocket.ascent_track(t_start=-10.0, t_end=760.0, step_s=0.5).set_index("time_s")
    assert track.at[-8.0, "engines"] == 0      # before S-IC ignition (-6.40 s)
    assert track.at[100.0, "engines"] == 2     # all first-stage engines
    assert track.at[150.0, "engines"] == 1     # centre engine off at 135.20 s
    assert track.at[163.0, "engines"] == 0     # between first-stage cutoff (161.63) and S-II ignition (164.00)
    assert track.at[500.0, "stage_index"] == 1 and track.at[500.0, "engines"] == 1
    assert track.at[600.0, "stage_index"] == 2
    assert track.at[700.0, "engines"] == 0     # final cutoff at 699.33 s
    assert 190 < rocket.crossing_time_s() < 215  # 100 km is crossed while the second stage burns


def test_animation_template_has_the_data_slot_the_app_fills():
    html = (ROOT / "assets" / "rocket_launch.html").read_text(encoding="utf-8")
    assert html.count("/*__DATA__*/null") == 1
    assert "window.rocketSim" in html


def test_dashboard_rocket_section_renders_in_both_languages():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=240).run()
    assert not at.exception
    next(r for r in at.radio if r.key == "rocket_cutoff").set_value("s_ivb_cutoff").run()
    assert not at.exception
    assert any("Stays in orbit" in s.value for s in at.success)
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[1]).run()  # switch to 中文 (the label change resets the radio)
    assert not at.exception
    next(r for r in at.radio if r.key == "rocket_cutoff").set_value("s_ivb_cutoff").run()
    assert not at.exception
    assert any("留在轨道" in s.value for s in at.success)
