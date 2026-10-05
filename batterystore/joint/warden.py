"""连接点健康值守。

``JointWarden`` 把五件事收在一个入口后面：窗口特征积累、按相对位次口径
判定劣化、确认后无缺口停用转移、维修后限功率观察、观察合格复投。状态机
按簇独立：

``in_service →（确认劣化）→ out_of_service →（维修完成）→ observing →（合格）→ in_service``
        ↓（健康簇裕度不足）                          ↓（观察不合格）
``degraded_hold（降额保持，裕度恢复后续传）``      ``out_of_service（再次停用）``

观察期与停用簇不参与群体基线，避免把待判对象揉进对照组；同型健康簇不足
``min_peers`` 时拒绝判定（相对位次在簇数过少时不可信），并在快照里用
``judging`` 标明判定是否在工作。维修后历史清零，旧趋势不进新基线。组件
本身不碰事件总线，动作以字典列表返回，由平台发布。
"""

from __future__ import annotations

import threading

from ..config import JointLimits
from ..errors import InterlockError, StoreError
from ..store.journal import JsonlJournal
from .criterion import FixedRiseCriterion, RankCriterion
from .dossier import ObservationDossier
from .metric import JointSample, WindowMetric, build_window_metric
from .snapshot import JointSnapshot
from .stats import median
from .transfer import plan_even, plan_offload, plan_rebalance

STATES = ("in_service", "degraded_hold", "out_of_service", "observing")

_HISTORY_LIMIT = 48


class _RackWatch:
    """一簇的值守上下文。"""

    def __init__(self, rack_id: str) -> None:
        self.rack_id = rack_id
        self.state = "in_service"
        self.history: list[WindowMetric] = []
        self.pending: list[JointSample] = []
        self.window_index = 0
        self.last_verdict: dict | None = None


class JointWarden:
    """连接点健康值守组件。"""

    def __init__(self, limits: JointLimits, journal: JsonlJournal | None = None) -> None:
        self._limits = limits
        self._journal = journal
        self._rank = RankCriterion(limits)
        self._fixed = FixedRiseCriterion(limits)
        self._watch: dict[str, _RackWatch] = {}
        self._capacity: dict[str, float] = {}
        self._allocation: dict[str, float] = {}
        self._dossiers: dict[str, ObservationDossier] = {}
        self._transfers = 0
        self._lock = threading.RLock()

    @property
    def limits(self) -> JointLimits:
        return self._limits

    def register(self, rack_id: str, capacity_kw: float, share_kw: float) -> None:
        """登记一簇的容量与初始出力份额。"""

        with self._lock:
            if rack_id in self._watch:
                raise StoreError(f"重复登记的簇: {rack_id}", component="joint")
            if not 0.0 <= float(share_kw) <= float(capacity_kw):
                raise StoreError("初始份额超出簇容量", component="joint")
            self._watch[rack_id] = _RackWatch(rack_id)
            self._capacity[rack_id] = float(capacity_kw)
            self._allocation[rack_id] = float(share_kw)

    def _require(self, rack_id: str) -> _RackWatch:
        rec = self._watch.get(rack_id)
        if rec is None:
            raise StoreError(f"未知的簇: {rack_id}", component="joint")
        return rec

    def state_of(self, rack_id: str) -> str:
        """返回簇当前值守状态。"""

        with self._lock:
            return self._require(rack_id).state

    def allocation(self) -> dict[str, float]:
        """返回当前簇间功率分配台账。"""

        with self._lock:
            return dict(self._allocation)

    def dossier_of(self, rack_id: str) -> dict | None:
        """返回某簇最新观察档案（没有则为 None）。"""

        with self._lock:
            dossier = self._dossiers.get(rack_id)
            return dossier.as_dict() if dossier is not None else None

    def ingest_sample(self, rack_id: str, resistance_mohm: float, temp_rise_c: float, current_a: float) -> dict:
        """接入一条采样；攒够一个窗口就评估并执行相应动作。"""

        with self._lock:
            rec = self._require(rack_id)
            if rec.state == "out_of_service":
                return {"rack_id": rack_id, "window_closed": False, "dropped": True, "actions": []}
            rec.pending.append(JointSample(float(resistance_mohm), float(temp_rise_c), float(current_a)))
            if len(rec.pending) < self._limits.window_samples:
                return {"rack_id": rack_id, "window_closed": False, "actions": []}
            metric = build_window_metric(
                rack_id, rec.window_index, rec.pending, self._limits.rated_current_a, self._limits.min_current_ratio
            )
            rec.pending = []
            rec.window_index += 1
            rec.history.append(metric)
            if len(rec.history) > _HISTORY_LIMIT:
                del rec.history[: len(rec.history) - _HISTORY_LIMIT]
            actions = self._route_window(rec, metric)
            return {"rack_id": rack_id, "window_closed": True, "metric": metric.as_dict(), "actions": actions}

    def _route_window(self, rec: _RackWatch, metric: WindowMetric) -> list[dict]:
        if rec.state == "observing":
            return self._observe_window(rec, metric)
        if rec.state == "degraded_hold":
            return self._deactivate(rec, reason="retry")
        if rec.state != "in_service" or len(rec.history) < self._limits.trend_windows:
            return []
        peers_r, peers_t = self._peer_levels(rec.rack_id)
        if len(peers_r) < self._limits.min_peers:
            return []
        verdict = self._rank.evaluate(rec.rack_id, list(rec.history), peers_r, peers_t)
        rec.last_verdict = verdict.as_dict()
        if not verdict.confirmed:
            return []
        confirm = {
            "action": "degradation_confirmed",
            "rack_id": rec.rack_id,
            "severity": "alarm",
            "message": "连接点劣化确认，停用该簇并转移负载",
            "window_index": verdict.window_index,
            "z_resistance": round(verdict.resistance.z, 3),
            "z_temp_rise": round(verdict.temp_rise.z, 3),
            "criterion": "relative_rank",
        }
        return [confirm, *self._deactivate(rec, reason="degradation")]

    def _level(self, rec: _RackWatch, attr: str) -> float:
        return median([getattr(metric, attr) for metric in rec.history[-self._limits.level_windows :]])

    def _peer_levels(self, rack_id: str) -> tuple[list[float], list[float]]:
        peers_r: list[float] = []
        peers_t: list[float] = []
        for other_id, other in self._watch.items():
            if other_id == rack_id or other.state != "in_service":
                continue
            if len(other.history) < self._limits.level_windows:
                continue
            peers_r.append(self._level(other, "resistance_mohm"))
            peers_t.append(self._level(other, "temp_rise_c"))
        return peers_r, peers_t

    def _deactivate(self, rec: _RackWatch, reason: str) -> list[dict]:
        recipients = [
            other_id
            for other_id, other in self._watch.items()
            if other_id != rec.rack_id and other.state == "in_service"
        ]
        try:
            plan = plan_offload(self.allocation(), dict(self._capacity), rec.rack_id, recipients, self._limits.transfer_step_kw)
        except InterlockError as error:
            rec.state = "degraded_hold"
            return [
                {
                    "action": "transfer_blocked",
                    "rack_id": rec.rack_id,
                    "severity": "alarm",
                    "message": f"负载转移被联锁拦下：{error.message}",
                    "reason": reason,
                }
            ]
        for step in plan["steps"]:
            self._allocation.update(step)
        self._transfers += 1
        rec.state = "degraded_hold" if plan["partial"] else "out_of_service"
        return [
            {
                "action": "load_transfer",
                "rack_id": rec.rack_id,
                "severity": "alarm" if plan["partial"] else "notice",
                "message": "健康簇裕度不足，部分卸载后降额保持" if plan["partial"] else "负载已转移到健康簇",
                "reason": reason,
                "steps": len(plan["steps"]),
                "partial": plan["partial"],
                "shortage_kw": round(plan["shortage_kw"], 3),
                "min_total_ratio": round(plan["min_total_ratio"], 6),
                "allocation": {rack_id: round(kw, 3) for rack_id, kw in plan["targets"].items()},
            }
        ]

    def begin_observation(self, rack_id: str) -> dict:
        """维修完成后进入观察期：限功率复归，观察档案开卷。"""

        with self._lock:
            rec = self._require(rack_id)
            if rec.state != "out_of_service":
                raise InterlockError("仅停用状态的簇可进入观察期", component="joint")
            others = [other_id for other_id, other in self._watch.items() if other.state == "in_service"]
            if not others:
                raise InterlockError("无在运簇可让出观察功率", component="joint")
            total = sum(self._allocation.values())
            obs_power = self._limits.observation_power_ratio * self._capacity[rack_id]
            if total < obs_power:
                raise InterlockError("在运总量不足以安排观察功率", component="joint")
            scale = (total - obs_power) / total if total > 0 else 0.0
            targets = {
                other_id: (self._allocation[other_id] * scale if other_id in others else self._allocation[other_id])
                for other_id in self._allocation
            }
            targets[rack_id] = obs_power
            steps = plan_rebalance(dict(self._allocation), targets, self._limits.transfer_step_kw)
            for step in steps:
                self._allocation.update(step)
            dossier = ObservationDossier(
                rack_id, rec.window_index, self._limits.observation_windows, self._limits.observation_z_limit
            )
            self._dossiers[rack_id] = dossier
            self._append_journal(
                {
                    "type": "observation_begin",
                    "rack_id": rack_id,
                    "started_window": dossier.started_window,
                    "required_windows": dossier.required_windows,
                    "z_limit": dossier.z_limit,
                }
            )
            rec.state = "observing"
            rec.history = []  # 维修后重新积累，旧趋势不进新基线
            self._rank.reset(rack_id)
            return {
                "rack_id": rack_id,
                "obs_power_kw": round(obs_power, 3),
                "steps": len(steps),
                "actions": [
                    {
                        "action": "observation_started",
                        "rack_id": rack_id,
                        "severity": "notice",
                        "message": "维修完成，限功率进入复投观察期",
                        "obs_power_kw": round(obs_power, 3),
                        "required_windows": dossier.required_windows,
                    }
                ],
            }

    def _observe_window(self, rec: _RackWatch, metric: WindowMetric) -> list[dict]:
        dossier = self._dossiers.get(rec.rack_id)
        if dossier is None or dossier.finalized:
            return []
        peers_r, peers_t = self._peer_levels(rec.rack_id)
        if len(peers_r) < self._limits.min_peers:
            return []
        r_level = self._level(rec, "resistance_mohm")
        t_level = self._level(rec, "temp_rise_c")
        z_r = self._rank.score(r_level, peers_r, self._limits.resistance_floor_mohm)
        z_t = self._rank.score(t_level, peers_t, self._limits.temp_rise_floor_c)
        entry = dossier.seal_window(metric.window_index, r_level, t_level, z_r, z_t)
        self._append_journal({"type": "observation_window", "rack_id": rec.rack_id, "entry": entry})
        if not entry["passed"]:
            final = dossier.finalize("fail", metric.window_index)
            self._append_journal({"type": "observation_final", "rack_id": rec.rack_id, "final": final})
            failed = {
                "action": "observation_failed",
                "rack_id": rec.rack_id,
                "severity": "alarm",
                "message": "观察期内指标再次越限，重新停用",
                "z_resistance": round(z_r, 3),
                "z_temp_rise": round(z_t, 3),
            }
            return [failed, *self._deactivate(rec, reason="observation_failed")]
        if not dossier.ready():
            return []
        final = dossier.finalize("pass", metric.window_index)
        self._append_journal({"type": "observation_final", "rack_id": rec.rack_id, "final": final})
        rec.state = "in_service"
        even = plan_even(self.allocation(), dict(self._capacity), self._active_ids(), self._limits.transfer_step_kw)
        for step in even["steps"]:
            self._allocation.update(step)
        self._transfers += 1
        return [
            {
                "action": "recommissioned",
                "rack_id": rec.rack_id,
                "severity": "notice",
                "message": "观察期合格，簇复投并按容量重新均分负载",
                "windows": dossier.sealed,
                "min_total_ratio": round(even["min_total_ratio"], 6),
                "allocation": {rack_id: round(kw, 3) for rack_id, kw in even["targets"].items()},
            }
        ]

    def _active_ids(self) -> list[str]:
        return [rack_id for rack_id, rec in self._watch.items() if rec.state == "in_service"]

    def adjust_allocation(self, rack_id: str, power_kw: float) -> dict:
        """调度回读：同步某簇实际出力；裕度释放后自动完成滞留的卸载。"""

        with self._lock:
            rec = self._require(rack_id)
            if rec.state == "out_of_service":
                raise InterlockError("停用簇不参与功率分配", component="joint")
            if not 0.0 <= float(power_kw) <= self._capacity[rack_id]:
                raise InterlockError("出力超出簇容量", component="joint")
            previous = self._allocation[rack_id]
            self._allocation[rack_id] = float(power_kw)
            actions: list[dict] = []
            for other in list(self._watch.values()):
                if other.state == "degraded_hold":
                    actions.extend(self._deactivate(other, reason="retry_after_adjust"))
            return {
                "rack_id": rack_id,
                "previous_kw": round(previous, 3),
                "power_kw": round(float(power_kw), 3),
                "actions": actions,
            }

    def compare(self, rack_id: str) -> dict:
        """两种判定口径对照（本站停用判决只采用相对位次）。"""

        with self._lock:
            rec = self._require(rack_id)
            if len(rec.history) < self._limits.trend_windows:
                raise InterlockError("历史窗口不足，无法对照判定口径", component="joint")
            peers_r, peers_t = self._peer_levels(rack_id)
            rank = None
            if len(peers_r) >= self._limits.min_peers:
                rank = self._rank.assess(rec.rack_id, list(rec.history), peers_r, peers_t)
            fixed = self._fixed.evaluate(rec.rack_id, list(rec.history))
            diverged = rank is not None and rank.confirmed != fixed.confirmed
            return {
                "rack_id": rack_id,
                "criterion": "relative_rank",
                "rank": rank.as_dict() if rank is not None else None,
                "fixed_rise": fixed.as_dict(),
                "diverged": diverged,
                "note": "环境温度整体漂移时两种口径结论可能相反；本站停用判决只采用相对位次口径",
            }

    def restore_dossiers(self, records: list[dict]) -> list[str]:
        """从流水重放观察档案；返回完整性校验未通过的簇。"""

        with self._lock:
            problems: list[str] = []
            for record in records:
                kind = record.get("type")
                rack_id = str(record.get("rack_id", ""))
                if kind == "observation_begin":
                    self._dossiers[rack_id] = ObservationDossier(
                        rack_id,
                        int(record.get("started_window", 0)),
                        int(record.get("required_windows", 1)),
                        float(record.get("z_limit", 1.5)),
                    )
                elif kind == "observation_window" and rack_id in self._dossiers:
                    dossier = self._dossiers[rack_id]
                    if dossier.finalized:
                        continue
                    entry = record.get("entry", {})
                    sealed = dossier.seal_window(
                        int(entry.get("window_index", 0)),
                        float(entry.get("resistance_level", 0.0)),
                        float(entry.get("temp_rise_level", 0.0)),
                        float(entry.get("z_r", 0.0)),
                        float(entry.get("z_t", 0.0)),
                    )
                    if sealed["hash"] != entry.get("hash"):
                        problems.append(f"{rack_id}:观察窗口哈希不一致")
                elif kind == "observation_final" and rack_id in self._dossiers:
                    dossier = self._dossiers[rack_id]
                    if dossier.finalized:
                        continue
                    final = record.get("final", {})
                    sealed_final = dossier.finalize(str(final.get("verdict", "fail")), int(final.get("window_index", 0)))
                    if sealed_final["hash"] != final.get("hash"):
                        problems.append(f"{rack_id}:观察结案哈希不一致")
            for rack_id, dossier in self._dossiers.items():
                if not dossier.verify_integrity():
                    problems.append(f"{rack_id}:观察档案哈希链断裂")
            return sorted(set(problems))

    def _append_journal(self, record: dict) -> None:
        if self._journal is not None:
            self._journal.append(record)

    def snapshot(self) -> JointSnapshot:
        """返回值守状态快照。"""

        with self._lock:
            allocation = self.allocation()
            states = {}
            for rack_id, rec in sorted(self._watch.items()):
                peers_r, _peers_t = self._peer_levels(rack_id)
                states[rack_id] = {
                    "state": rec.state,
                    "allocation_kw": round(allocation[rack_id], 3),
                    "capacity_kw": round(self._capacity[rack_id], 3),
                    "windows": len(rec.history),
                    "streak": self._rank.streak_of(rack_id),
                    "peers": len(peers_r),
                    "judging": rec.state == "in_service" and len(peers_r) >= self._limits.min_peers,
                    "last_verdict": rec.last_verdict,
                }
            return JointSnapshot(
                criterion="relative_rank",
                states=states,
                dossiers={rack_id: dossier.as_dict() for rack_id, dossier in sorted(self._dossiers.items())},
                total_kw=round(sum(allocation.values()), 3),
                transfers=self._transfers,
            )
