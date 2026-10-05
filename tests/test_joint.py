"""连接点健康：劣化判定口径、无缺口负载转移与复投观察。

判定口径的选型理由被写成可执行用例：环境温度整体漂移时，固定上升幅度口径
会把健康簇误判为劣化，而相对位次口径保持沉默——本站因此只采用相对位次。
"""

from __future__ import annotations

import json

import pytest

from batterystore.app import build_platform
from batterystore.clock import ManualClock
from batterystore.config import StationConfig, config_to_mapping, default_config, load_config
from batterystore.errors import ConfigError, InterlockError, StoreError
from batterystore.joint.criterion import FixedRiseCriterion, RankCriterion
from batterystore.joint.dossier import ObservationDossier
from batterystore.joint.metric import JointSample, build_window_metric
from batterystore.joint.transfer import plan_even, plan_offload, plan_rebalance
from batterystore.joint.warden import JointWarden
from batterystore.store.journal import JsonlJournal

RATED_A = 140.0


def _limits():
    return default_config().joint


def _warden(racks: int = 6, capacity: float = 100.0, share: float = 60.0, journal=None, shares: dict | None = None):
    warden = JointWarden(_limits(), journal)
    for index in range(racks):
        rack_id = f"rk-{index + 1:02d}"
        warden.register(rack_id, capacity, (shares or {}).get(rack_id, share))
    return warden


def _feed_window(warden, rack_id: str, resistance: float, temp_rise: float, current: float = RATED_A) -> list[dict]:
    actions: list[dict] = []
    for _ in range(_limits().window_samples):
        report = warden.ingest_sample(rack_id, resistance, temp_rise, current)
        actions.extend(report["actions"])
    return actions


def _feed_fleet(warden, window: int, degrade: str | None = None, rate: float = 1.03, degrade_from: int = 15) -> list[dict]:
    actions: list[dict] = []
    for rack_id in sorted(warden.allocation()):
        resistance, temp_rise = 1.20, 4.5
        if rack_id == degrade and window >= degrade_from:
            factor = rate ** (window - degrade_from)
            resistance, temp_rise = 1.20 * factor, 4.5 * factor
        actions.extend(_feed_window(warden, rack_id, resistance, temp_rise))
    return actions


def test_window_metric_normalizes_temp_rise_to_rated_current():
    samples = [
        JointSample(1.21, 4.5, RATED_A),
        JointSample(1.19, 4.5 * (0.5) ** 2, RATED_A * 0.5),  # 半电流下的温升折算后应一致
        JointSample(1.22, 9.9, RATED_A * 0.1),  # 电流过低，不参与温升统计
    ]
    metric = build_window_metric("rk-01", 0, samples, RATED_A, 0.3)
    assert metric.resistance_mohm == pytest.approx(1.21, abs=1e-6)
    assert metric.temp_rise_c == pytest.approx(4.5, abs=1e-6)
    assert metric.samples == 3


def test_window_metric_rejects_empty_or_low_current_samples():
    with pytest.raises(StoreError):
        build_window_metric("rk-01", 0, [], RATED_A, 0.3)
    with pytest.raises(StoreError):
        build_window_metric("rk-01", 0, [JointSample(1.2, 4.5, RATED_A * 0.1)], RATED_A, 0.3)


def test_ambient_shift_splits_the_two_criteria():
    """寒潮让全部簇内阻同步抬升：相对位次沉默、固定幅度误报，结论相反。"""
    warden = _warden()
    for window in range(40):
        env = 1.0 if window < 20 else 1.15
        for rack_id in sorted(warden.allocation()):
            _feed_window(warden, rack_id, 1.20 * env, 4.5 * env)
    assert warden.snapshot().as_dict()["transfers"] == 0
    compare = warden.compare("rk-01")
    assert compare["rank"]["confirmed"] is False
    assert compare["fixed_rise"]["confirmed"] is True
    assert compare["diverged"] is True
    assert compare["criterion"] == "relative_rank"


def test_degrading_rack_is_retired_and_load_moves_without_gap():
    warden = _warden()
    total_before = sum(warden.allocation().values())
    confirmed = False
    for window in range(40):
        actions = _feed_fleet(warden, window, degrade="rk-03")
        if any(action["action"] == "degradation_confirmed" for action in actions):
            confirmed = True
            break
    assert confirmed, "劣化簇应被确认"
    assert warden.state_of("rk-03") == "out_of_service"
    transfer = next(action for action in actions if action["action"] == "load_transfer")
    assert transfer["min_total_ratio"] >= 1.0 - 1e-6
    allocation = warden.allocation()
    assert allocation["rk-03"] == 0.0
    assert sum(allocation.values()) == pytest.approx(total_before)
    assert all(kw <= 100.0 for kw in allocation.values())


def test_transfer_steps_never_open_a_gap():
    now = {"a": 150.0, "b": 150.0, "c": 150.0, "d": 150.0}
    target = {"a": 0.0, "b": 200.0, "c": 200.0, "d": 200.0}
    steps = plan_rebalance(now, target, 30.0)
    assert steps, "应当产生转移轨迹"
    for step in steps:
        assert sum(step.values()) >= sum(now.values()) - 1e-6
    assert steps[-1] == target
    with pytest.raises(InterlockError):
        plan_rebalance(now, {"a": 0.0, "b": 100.0, "c": 100.0, "d": 100.0}, 30.0)


def test_plan_offload_partial_when_headroom_short():
    allocation = {"a": 60.0, "b": 95.0, "c": 95.0}
    capacity = {"a": 100.0, "b": 100.0, "c": 100.0}
    plan = plan_offload(allocation, capacity, "a", ["b", "c"], 30.0)
    assert plan["partial"] is True
    assert plan["targets"]["a"] == pytest.approx(50.0)
    assert plan["shortage_kw"] == pytest.approx(50.0)
    with pytest.raises(InterlockError):
        plan_offload(allocation, capacity, "a", [], 30.0)


def test_partial_offload_holds_then_completes_after_dispatch_frees_headroom():
    shares = {"rk-01": 95.0, "rk-02": 95.0, "rk-04": 95.0, "rk-05": 95.0, "rk-06": 95.0, "rk-03": 60.0}
    warden = _warden(shares=shares)
    for window in range(40):
        actions = _feed_fleet(warden, window, degrade="rk-03")
        if any(action["action"] == "degradation_confirmed" for action in actions):
            break
    assert warden.state_of("rk-03") == "degraded_hold"
    transfer = next(action for action in actions if action["action"] == "load_transfer")
    assert transfer["partial"] is True
    assert transfer["severity"] == "alarm"
    report = warden.adjust_allocation("rk-01", 40.0)
    assert any(action["action"] == "load_transfer" and not action["partial"] for action in report["actions"])
    assert warden.state_of("rk-03") == "out_of_service"
    assert warden.allocation()["rk-03"] == 0.0


def test_judgment_pauses_when_peer_group_too_small():
    """同型健康簇不足时拒绝判定（相对位次不可信），快照标明判定停摆。"""
    warden = _warden(racks=4)
    for window in range(40):
        actions = _feed_fleet(warden, window, degrade="rk-04")
        if any(action["action"] == "degradation_confirmed" for action in actions):
            break
    assert warden.state_of("rk-04") == "out_of_service"
    states = warden.snapshot().as_dict()["states"]
    assert states["rk-01"]["judging"] is False
    for window in range(40, 80):
        _feed_fleet(warden, window, degrade="rk-03")
    assert warden.state_of("rk-03") == "in_service"


def test_observation_pass_recommissions_with_even_allocation():
    warden = _warden()
    for window in range(40):
        actions = _feed_fleet(warden, window, degrade="rk-03")
        if any(action["action"] == "degradation_confirmed" for action in actions):
            break
    assert warden.state_of("rk-03") == "out_of_service"
    warden.begin_observation("rk-03")
    assert warden.state_of("rk-03") == "observing"
    recommissioned = False
    for _ in range(_limits().observation_windows):
        actions = _feed_fleet(warden, 99)  # 维修后指标回到群体带内
        recommissioned = recommissioned or any(action["action"] == "recommissioned" for action in actions)
    assert recommissioned, "观察合格应当复投"
    assert warden.state_of("rk-03") == "in_service"
    dossier = warden.dossier_of("rk-03")
    assert dossier["verdict"] == "pass"
    assert dossier["sealed"] == _limits().observation_windows
    assert dossier["integrity"] is True
    for kw in warden.allocation().values():
        assert kw == pytest.approx(60.0)


def test_observation_fail_returns_to_out_of_service():
    warden = _warden()
    for window in range(40):
        actions = _feed_fleet(warden, window, degrade="rk-03")
        if any(action["action"] == "degradation_confirmed" for action in actions):
            break
    warden.begin_observation("rk-03")
    failed = False
    for window in range(41, 46):
        actions = _feed_fleet(warden, window, degrade="rk-03", rate=1.4, degrade_from=40)  # 维修无效
        failed = failed or any(action["action"] == "observation_failed" for action in actions)
        if failed:
            break
    assert failed, "维修无效应当在观察期内被拦下"
    assert warden.state_of("rk-03") == "out_of_service"
    assert warden.dossier_of("rk-03")["verdict"] == "fail"


def test_dossier_sealed_against_overwrite_and_tamper():
    dossier = ObservationDossier("rk-01", 0, 2, 2.5)
    dossier.seal_window(0, 1.2, 4.5, 0.1, 0.2)
    dossier.seal_window(1, 1.21, 4.51, 0.2, 0.1)
    dossier.finalize("pass", 1)
    with pytest.raises(InterlockError):
        dossier.seal_window(2, 1.2, 4.5, 0.0, 0.0)
    with pytest.raises(InterlockError):
        dossier.finalize("fail", 2)
    assert dossier.verify_integrity() is True
    dossier._entries[0]["z_r"] = 99.0  # noqa: SLF001 - 故意篡改以验证哈希链
    assert dossier.verify_integrity() is False


def test_dossier_replay_from_journal_is_not_rewritten_by_recalc(tmp_path):
    journal = JsonlJournal(str(tmp_path), "joint")
    warden = _warden(journal=journal)
    for window in range(40):
        actions = _feed_fleet(warden, window, degrade="rk-03")
        if any(action["action"] == "degradation_confirmed" for action in actions):
            break
    warden.begin_observation("rk-03")
    for _ in range(_limits().observation_windows):
        _feed_fleet(warden, 99)
    original = warden.dossier_of("rk-03")
    assert original["verdict"] == "pass"

    replayed = JointWarden(_limits())  # 重算/重启只能重放，不得覆盖
    problems = replayed.restore_dossiers(journal.read_all())
    assert problems == []
    restored = replayed.dossier_of("rk-03")
    assert restored["entries"] == original["entries"]
    assert restored["integrity"] is True

    tampered = json.loads(json.dumps(journal.read_all()))
    for record in tampered:
        if record.get("type") == "observation_window":
            record["entry"]["z_r"] = 42.0
            break
    again = JointWarden(_limits())
    assert again.restore_dossiers(tampered) != []


def test_platform_joint_drill_runs_full_cycle_and_survives_restart(tmp_path):
    storage = str(tmp_path / "state")
    platform = build_platform(default_config(), clock=ManualClock(), storage_root=storage, rack_count=6)
    platform.start()
    report = platform.joint_drill()
    assert report["ok"], report.get("actions")
    assert report["min_total_ratio"] >= 1.0 - 1e-6
    assert report["dossier"]["verdict"] == "pass"
    assert report["compare"]["diverged"] is True
    assert platform.status()["joint"]["criterion"] == "relative_rank"
    actions = {entry["action"] for entry in platform.audit.query(component="joint")}
    assert {"degradation_confirmed", "load_transfer", "observation_started", "recommissioned"} <= actions
    platform.stop()

    rebuilt = build_platform(default_config(), clock=ManualClock(), storage_root=storage, rack_count=6)
    dossier = rebuilt.joint.dossier_of(report["target"])
    assert dossier["verdict"] == "pass"
    assert dossier["integrity"] is True
    assert dossier["entries"] == report["dossier"]["entries"]


def test_joint_drill_requires_peer_group(tmp_path):
    platform = build_platform(default_config(), clock=ManualClock(), storage_root=str(tmp_path / "state"), rack_count=2)
    report = platform.joint_drill()
    assert report["ran"] is False
    assert "insufficient_peers" in report["violations"]


def test_joint_interlocks_are_enforced(tmp_path):
    platform = build_platform(default_config(), clock=ManualClock(), storage_root=str(tmp_path / "state"), rack_count=6)
    with pytest.raises(InterlockError):
        platform.joint_begin_observation("rk-01")  # 在运簇不能直接进观察期
    with pytest.raises(InterlockError):
        platform.joint_compare("rk-01")  # 历史窗口不足
    with pytest.raises(KeyError):
        platform.joint_sample("rk-99")


def test_joint_config_section_roundtrip_and_validation(tmp_path):
    config = default_config()
    assert config.joint.observation_z_limit < config.joint.z_threshold
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_to_mapping(config)), encoding="utf-8")
    assert load_config(str(path)) == config
    raw = config_to_mapping(config)
    raw["limits"]["joint"]["observation_z_limit"] = 99.0
    with pytest.raises(ConfigError):
        StationConfig.from_mapping(raw)


def test_shipped_config_carries_joint_section():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config" / "batterystore.json"
    config = load_config(str(path))
    assert config.joint.min_peers >= 2
    assert config.joint.observation_windows >= 1


def test_console_joint_routes(tmp_path):
    import json as jsonlib
    import urllib.error
    import urllib.request

    import support
    from batterystore.console.server import ConsoleServer

    def _get(base_url: str, path: str):
        with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as response:
            return response.status, jsonlib.loads(response.read().decode("utf-8"))

    def _post(base_url: str, path: str, payload: dict):
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=jsonlib.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, jsonlib.loads(response.read().decode("utf-8"))

    platform = build_platform(default_config(), clock=ManualClock(), storage_root=str(tmp_path / "state"), rack_count=6)
    support.bring_online(platform)
    server = ConsoleServer(platform, port=0)
    base_url = server.start()
    try:
        code, payload = _post(base_url, "/api/joint/sample", {"rack_id": "rk-01", "resistance_mohm": 1.21})
        assert code == 200
        assert payload["rack_id"] == "rk-01"
        code, payload = _get(base_url, "/api/joint")
        assert code == 200
        assert payload["criterion"] == "relative_rank"
        assert len(payload["states"]) == 6
        code, payload = _post(base_url, "/api/joint/drill", {})
        assert code == 200
        assert payload["ok"] is True
        code, payload = _post(base_url, "/api/joint/compare", {"rack_id": "rk-01"})
        assert code == 200
        assert payload["criterion"] == "relative_rank"
        try:
            _post(base_url, "/api/joint/observe", {"rack_id": "rk-01"})
        except urllib.error.HTTPError as error:
            assert error.code == 409
        else:  # pragma: no cover - 明确失败路径
            raise AssertionError("在运簇进入观察期应当被拒绝")
    finally:
        server.stop()
