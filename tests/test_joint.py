"""连接点健康：判定口径、无缺口转移、只增台账与复投观察。"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from batterystore.app import build_platform
from batterystore.clock import ManualClock
from batterystore.config import default_config
from batterystore.errors import InterlockError
from batterystore.joint.features import DayFeature, aggregate_day
from batterystore.joint.ledger import JointLedger
from batterystore.joint.ranking import evaluate_persistence, rank_day
from batterystore.joint.sample import JointSample, sample_metrics
from batterystore.joint.transfer import ClusterPower, allocate, execute_plan, plan_outage, plan_reentry
from batterystore.store.journal import JsonlJournal


def _telemetry(rack_id: str, r_mohm: float, k_norm: float, count: int = 24, t_amb: float = 25.0) -> list[dict]:
    """按目标电阻与温升系数反演一天的遥测（确定性小波动，保持准稳态）。"""

    rows: list[dict] = []
    for index in range(count):
        current_a = 120.0 + ((index * 7) % 5) - 2.0
        rows.append(
            {
                "ts": float(index * 600),
                "current_a": current_a,
                "v_terminal_v": 800.0 + r_mohm / 1000.0 * current_a,
                "v_bus_v": 800.0,
                "t_conn_c": t_amb + k_norm * (current_a / 1000.0) ** 2,
                "t_amb_c": t_amb,
            }
        )
    return rows


def _platform(tmp_path, racks: int = 4, **joint_overrides):
    config = default_config()
    if joint_overrides:
        config = replace(config, joint=replace(config.joint, **joint_overrides))
    return build_platform(config, clock=ManualClock(), storage_root=str(tmp_path / "state"), rack_count=racks)


def _healthy_map(platform) -> dict:
    return {rack_id: (1.0 + 0.03 * pos, 800.0 + 8.0 * pos) for pos, rack_id in enumerate(sorted(platform.racks))}


def _drive_to_isolation(platform, healthy: dict, days: range, degrade_from: int = 1) -> dict:
    """让 rk-04 从 degrade_from 起持续劣化，返回最后一次评估报告。"""

    report: dict = {}
    for day_index in days:
        telemetry = {}
        for rack_id, (r_mohm, k_norm) in healthy.items():
            degrading = rack_id == "rk-04" and day_index >= degrade_from
            telemetry[rack_id] = _telemetry(rack_id, 1.45 if degrading else r_mohm, 1160.0 if degrading else k_norm)
        report = platform.joint_review(telemetry, day=f"2026-09-{day_index:02d}")
    return report


# ----------------------------------------------------------------------
# 特征折算
# ----------------------------------------------------------------------


def test_sample_metrics_recovers_resistance_and_normalized_temp_rise():
    limits = default_config().joint
    samples = [
        JointSample(
            ts=float(index * 600),
            rack_id="rk-01",
            current_a=120.0,
            v_terminal_v=800.0 + 0.0015 * 120.0,
            v_bus_v=800.0,
            t_conn_c=25.0 + 900.0 * 0.12**2,
            t_amb_c=25.0,
        )
        for index in range(30)
    ]
    rows = sample_metrics(samples, limits)
    assert len(rows) == 30
    assert statistics.median(row[1] for row in rows) == pytest.approx(1.5, rel=0.01)
    assert statistics.median(row[2] for row in rows) == pytest.approx(900.0, rel=0.01)


def test_sample_metrics_drops_unusable_samples():
    limits = default_config().joint
    low_current = JointSample(1.0, "rk-01", 10.0, 800.1, 800.0, 40.0, 25.0)
    over_temp = JointSample(2.0, "rk-01", 120.0, 800.1, 800.0, 200.0, 25.0)
    offline = JointSample(3.0, "rk-01", 120.0, 800.1, 800.0, 40.0, 25.0, online=False)
    assert sample_metrics([low_current, over_temp, offline], limits) == []


def test_daily_feature_quality_gates():
    limits = default_config().joint
    few = [JointSample.from_dict("rk-01", raw) for raw in _telemetry("rk-01", 1.0, 800.0, count=5)]
    assert aggregate_day("rk-01", "d1", few, limits).quality == "INSUFFICIENT_SAMPLES"
    stuck = [
        JointSample(float(i * 600), "rk-01", 100.0 + (i % 5) * 20.0, 800.1, 800.0, 40.0, 25.0)
        for i in range(30)
    ]
    assert aggregate_day("rk-01", "d1", stuck, limits).quality == "STUCK_SENSOR"


# ----------------------------------------------------------------------
# 判定口径：相对位次 vs 固定幅度（选定前者的理由）
# ----------------------------------------------------------------------


def _features(scale: float, day: str = "d1") -> dict:
    return {
        f"rk-{pos + 1:02d}": DayFeature(
            f"rk-{pos + 1:02d}", day, (1.0 + 0.03 * pos) * scale, (800.0 + 8.0 * pos) * scale, 24, "OK", 1
        )
        for pos in range(4)
    }


def test_peer_ranking_flags_only_the_degrading_cluster():
    limits = default_config().joint
    features = _features(1.0)
    features["rk-03"] = DayFeature("rk-03", "d1", 1.6, 1300.0, 24, "OK", 1)
    scores = rank_day(features, limits)
    assert scores["rk-03"].flag == "OUTLIER"
    assert [scores[key].flag for key in ("rk-01", "rk-02", "rk-04")] == ["NORMAL"] * 3


def test_peer_ranking_is_ambient_invariant_where_fixed_threshold_flips():
    """环境温度整体变化时两种口径结论不同：相对位次不变，固定幅度翻转。"""

    limits = default_config().joint
    before = rank_day(_features(1.0), limits)
    after = rank_day(_features(1.12), limits)  # 季节共模上移 12%（≈30℃ 温差）
    assert [score.flag for score in before.values()] == [score.flag for score in after.values()]
    # 对照口径：固定上升幅度（基线 1.0mΩ、门限 +10%）在共模上移后全部误报。
    assert not any(feature.r_mohm > 1.0 * 1.1 for feature in _features(1.0).values())
    assert all(feature.r_mohm > 1.0 * 1.1 for feature in _features(1.12).values())
    # 同一轮共模上移下，真正劣化的那台仍被相对位次口径挑出。
    degraded = _features(1.12)
    degraded["rk-02"] = DayFeature("rk-02", "d1", 1.6, 1300.0, 24, "OK", 1)
    assert rank_day(degraded, limits)["rk-02"].flag == "OUTLIER"


def test_persistence_turns_flags_into_verdicts():
    limits = default_config().joint
    assert evaluate_persistence(["OUTLIER"] * 7 + ["NORMAL"] * 3, limits) == "DEGRADED"
    assert evaluate_persistence(["OUTLIER"] * 3 + ["NORMAL"] * 7, limits) == "NORMAL"
    assert evaluate_persistence(["WATCH"] * 7 + ["NORMAL"] * 3, limits) == "WATCH"
    assert evaluate_persistence(["OUTLIER"] * 3, limits) == "SUSPENDED"


# ----------------------------------------------------------------------
# 无缺口负载转移
# ----------------------------------------------------------------------


def _powers(p_now: float = 250.0, p_max: float = 500.0, ramp: float = 40.0) -> dict:
    return {f"rk-{i:02d}": ClusterPower(f"rk-{i:02d}", p_now, p_max, ramp) for i in range(1, 5)}


def test_allocate_splits_by_capacity_without_violating_limits():
    even = allocate(1000.0, {key: (0.0, 500.0) for key in "abcd"})
    assert sum(even.values()) == pytest.approx(1000.0)
    assert all(value == pytest.approx(250.0) for value in even.values())
    skewed = allocate(1000.0, {"a": (0.0, 100.0), "b": (0.0, 900.0)})
    assert skewed["a"] == pytest.approx(100.0)
    assert skewed["b"] == pytest.approx(900.0)
    with pytest.raises(InterlockError):
        allocate(2000.0, {"a": (0.0, 500.0)})


def test_plan_outage_conserves_total_power_every_step():
    powers = _powers()
    plan = plan_outage(powers, "rk-04", 0.05)
    assert plan.final["rk-04"] == 0.0
    assert all(0.0 <= value <= 500.0 for value in plan.final.values())
    previous = plan.initial
    for step in plan.steps:
        assert sum(step.values()) == pytest.approx(1000.0, abs=1e-6)
        for rack_id in powers:
            assert abs(step[rack_id] - previous[rack_id]) <= 40.0 + 1e-9
        previous = step


def test_plan_outage_refuses_when_headroom_is_short():
    powers = _powers(p_now=490.0, p_max=500.0)
    with pytest.raises(InterlockError):
        plan_outage(powers, "rk-04", 0.05)


def test_plan_reentry_keeps_total_constant():
    powers = _powers(p_now=1000.0 / 3.0)
    powers["rk-04"] = ClusterPower("rk-04", 0.0, 500.0, 40.0)
    plan = plan_reentry(powers, "rk-04")
    assert plan.final["rk-04"] == pytest.approx(250.0)
    for step in plan.steps:
        assert sum(step.values()) == pytest.approx(1000.0, abs=1e-6)


def test_execute_plan_rolls_back_when_confirm_fails():
    plan = plan_outage(_powers(), "rk-04", 0.05)
    applied: list[dict] = []
    calls = {"n": 0}

    def apply(setpoints: dict) -> None:
        applied.append(dict(setpoints))

    def confirm(setpoints: dict) -> bool:
        calls["n"] += 1
        return calls["n"] != 2  # 第 2 步确认失败

    with pytest.raises(InterlockError):
        execute_plan(plan, apply, confirm)
    assert applied[-1] == plan.initial


# ----------------------------------------------------------------------
# 只增台账：观察记录不被重算覆盖
# ----------------------------------------------------------------------


def test_ledger_detects_post_hoc_tampering(tmp_path):
    journal = JsonlJournal(str(tmp_path), "joint")
    ledger = JointLedger(journal)
    ledger.append("feature", "rk-01", "2026-09-01", {"r_mohm": 1.0})
    ledger.append("observation", "rk-01", "2026-09-02", {"z": 0.5})
    assert ledger.verify()["ok"]
    path = Path(journal.segments()[0])
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload"]["r_mohm"] = 9.9  # 模拟事后覆盖
    lines[0] = json.dumps(first, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not ledger.verify()["ok"]


def test_ledger_recompute_appends_new_version_without_overwriting(tmp_path):
    ledger = JointLedger(JsonlJournal(str(tmp_path), "joint"))
    ledger.append("feature", "rk-01", "2026-09-01", {"r_mohm": 1.0, "version": 1})
    ledger.append("feature", "rk-01", "2026-09-01", {"r_mohm": 1.1, "version": 2})
    versions = [record["payload"]["version"] for record in ledger.records(kind="feature")]
    assert versions == [1, 2]
    assert ledger.verify()["ok"]


# ----------------------------------------------------------------------
# 平台级全生命周期
# ----------------------------------------------------------------------


def test_full_lifecycle_from_degradation_to_recommission(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.joint_dispatch(total_kw=1000.0)
    healthy = _healthy_map(platform)

    isolated_day = None
    for day_index in range(1, 16):
        report = _drive_to_isolation(platform, healthy, range(day_index, day_index + 1), degrade_from=8)
        if report["states"]["rk-04"] == "out_of_service":
            isolated_day = day_index
            break
    assert isolated_day == 14  # 第 8 天起连续 7 天离群 → 第 14 天判定并停用
    snapshot = platform.joint_snapshot()
    assert snapshot["allocation_kw"]["rk-04"] == 0.0
    assert sum(snapshot["allocation_kw"].values()) == pytest.approx(1000.0)

    repair = platform.joint_repair_done(rack_id="rk-04", day="2026-09-15")
    assert repair["state"] == "observation"
    assert platform.joint_snapshot()["allocation_kw"]["rk-04"] == pytest.approx(100.0)

    for day_index in range(15, 22):
        telemetry = {rack_id: _telemetry(rack_id, r, k) for rack_id, (r, k) in healthy.items()}
        platform.joint_review(telemetry, day=f"2026-09-{day_index:02d}")
    close = platform.joint_close_observation(rack_id="rk-04", day="2026-09-22")
    assert close["decision"] == "reinstated"
    final = platform.joint_snapshot()
    assert final["states"]["rk-04"] == "in_service"
    assert final["allocation_kw"]["rk-04"] == pytest.approx(250.0)
    assert sum(final["allocation_kw"].values()) == pytest.approx(1000.0)
    assert platform.joint_ledger()["verify"]["ok"]


def test_observation_records_survive_same_day_recompute(tmp_path):
    platform = _platform(tmp_path, persist_days=3, window_days=4, observation_days=3)
    platform.start()
    platform.joint_dispatch(total_kw=1000.0)
    healthy = _healthy_map(platform)
    _drive_to_isolation(platform, healthy, range(1, 4))
    assert platform.joint_snapshot()["states"]["rk-04"] == "out_of_service"
    platform.joint_repair_done(rack_id="rk-04", day="2026-09-04")
    for day_index in (4, 5, 6):
        telemetry = {rack_id: _telemetry(rack_id, r, k) for rack_id, (r, k) in healthy.items()}
        platform.joint_review(telemetry, day=f"2026-09-{day_index:02d}")

    journal = JsonlJournal(platform.store.root, "joint")
    observed = [r for r in journal.read_all() if r["kind"] == "observation" and r["day"] == "2026-09-05"]
    assert len(observed) == 1
    frozen_r = observed[0]["payload"]["feature"]["r_mohm"]

    # 同一天重算：注入差异很大的数据，观察记录不得被覆盖，特征只追加新版本。
    recomputed = {rack_id: _telemetry(rack_id, r, k) for rack_id, (r, k) in healthy.items()}
    recomputed["rk-04"] = _telemetry("rk-04", 9.9, 9000.0)
    platform.joint_review(recomputed, day="2026-09-05")
    after = [r for r in journal.read_all() if r["kind"] == "observation" and r["day"] == "2026-09-05"]
    assert after == observed
    versions = {
        r["payload"]["version"]
        for r in journal.read_all()
        if r["kind"] == "feature" and r["rack_id"] == "rk-04" and r["day"] == "2026-09-05"
    }
    assert versions == {1, 2}
    assert observed[0]["payload"]["feature"]["r_mohm"] == frozen_r

    close = platform.joint_close_observation(rack_id="rk-04", day="2026-09-07")
    assert close["decision"] == "reinstated"  # 复投评估只认冻结的观察记录


def test_observation_reject_returns_out_of_service_without_gap(tmp_path):
    platform = _platform(tmp_path, persist_days=3, window_days=4, observation_days=3)
    platform.start()
    platform.joint_dispatch(total_kw=1000.0)
    healthy = _healthy_map(platform)
    _drive_to_isolation(platform, healthy, range(1, 4))
    platform.joint_repair_done(rack_id="rk-04", day="2026-09-04")
    # 观察期内 rk-04 的指标仍然离群 → 复投不合格。
    for day_index in (4, 5, 6):
        telemetry = {rack_id: _telemetry(rack_id, r, k) for rack_id, (r, k) in healthy.items()}
        telemetry["rk-04"] = _telemetry("rk-04", 1.45, 1160.0)
        platform.joint_review(telemetry, day=f"2026-09-{day_index:02d}")
    close = platform.joint_close_observation(rack_id="rk-04", day="2026-09-07")
    assert close["decision"] == "rejected"
    snapshot = platform.joint_snapshot()
    assert snapshot["states"]["rk-04"] == "out_of_service"
    assert snapshot["allocation_kw"]["rk-04"] == 0.0
    assert sum(snapshot["allocation_kw"].values()) == pytest.approx(1000.0)


def test_isolation_waits_for_headroom_and_retries(tmp_path):
    platform = _platform(tmp_path, persist_days=3, window_days=4)
    platform.start()
    platform.joint_dispatch(total_kw=1960.0)  # 每簇 490kW，几乎顶格
    healthy = _healthy_map(platform)
    _drive_to_isolation(platform, healthy, range(1, 4))
    snapshot = platform.joint_snapshot()
    assert snapshot["states"]["rk-04"] == "degraded"  # 判定成立但裕度不足，未停用
    assert snapshot["allocation_kw"]["rk-04"] == pytest.approx(490.0)
    blocked = platform.audit.query(component="joint", action="isolation_blocked")
    assert blocked

    platform.joint_dispatch(total_kw=1000.0)  # 调度腾出裕度后，下一次评估重试成功
    _drive_to_isolation(platform, healthy, range(4, 5))
    assert platform.joint_snapshot()["states"]["rk-04"] == "out_of_service"


def test_lifecycle_rejects_illegal_transitions(tmp_path):
    platform = _platform(tmp_path)
    with pytest.raises(InterlockError):
        platform.joint_close_observation(rack_id="rk-01")
    with pytest.raises(InterlockError):
        platform.joint_repair_done(rack_id="rk-01")


def test_absolute_temperature_protection_is_independent_of_ranking(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    telemetry = {rack_id: _telemetry(rack_id, 1.0, 800.0) for rack_id in sorted(platform.racks)}
    telemetry["rk-02"][0]["t_conn_c"] = 110.0
    report = platform.joint_review(telemetry, day="2026-09-01")
    assert report["protection"] == ["rk-02"]


def test_console_exposes_joint_routes(tmp_path):
    from batterystore.console.api import Api
    from batterystore.console.pages import PageLibrary

    api = Api(_platform(tmp_path), PageLibrary("web"))
    paths = [route["path"] for route in api.routes()]
    assert "/api/joint" in paths
    assert "/api/joint/ledger" in paths
