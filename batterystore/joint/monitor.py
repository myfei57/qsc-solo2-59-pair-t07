"""连接点健康监控组件。

一条完整的处置链：日特征 → 同型簇相对位次判定 → 停用与负载转移 → 处理后
限功率观察 → 复投评估。判定口径、状态机、无缺口转移与只增台账分别在
ranking / lifecycle / transfer / ledger 四个模块里，本组件负责把它们接成
一条链，并把每一步写进事件总线（审计据此回放）。

两条贯穿全链的规矩：

- **重算不覆盖**：同一天重复评估时，特征只追加更高版本的记录；判定结论与
  观察记录以当天首次落台账的为准，之后怎么重算都不改写历史。
- **无缺口**：停用、限功率并入、复投、退回，凡是改变功率分配的动作都走
  transfer 模块的同步反向爬坡计划，全站合计在每一步保持不变。
"""

from __future__ import annotations

from ..clock import Clock
from ..config import JointLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus
from .features import DayFeature, aggregate_day
from .ledger import JointLedger
from .lifecycle import (
    STATE_DEGRADED,
    STATE_IN_SERVICE,
    STATE_OBSERVATION,
    STATE_OUT_OF_SERVICE,
    STATE_WATCH,
    JointLifecycle,
)
from .ranking import (
    FLAG_INVALID,
    VERDICT_DEGRADED,
    VERDICT_NORMAL,
    VERDICT_SUSPENDED,
    VERDICT_WATCH,
    DayScore,
    evaluate_persistence,
    rank_day,
    trend_slope,
)
from .sample import JointSample
from .snapshot import JointSnapshot
from .transfer import ClusterPower, allocate, execute_plan, plan_outage, plan_reentry

# 参与功率分配与判定的运行态；停用与观察期簇的功率保持恒定，不参与重分摊。
_RUNNING_STATES = (STATE_IN_SERVICE, STATE_WATCH, STATE_DEGRADED)


class JointHealthMonitor:
    """簇级连接点健康监控。"""

    def __init__(self, limits: JointLimits, ledger: JointLedger, clock: Clock, bus: EventBus, station: str) -> None:
        self._limits = limits
        self._ledger = ledger
        self._clock = clock
        self._bus = bus
        self._station = station
        self._lifecycle = JointLifecycle()
        self._types: dict[str, str] = {}
        self._capability_kw: dict[str, float] = {}
        self._allocation_kw: dict[str, float] = {}
        self._features: dict[str, dict[str, DayFeature]] = {}
        self._versions: dict[tuple[str, str], int] = {}
        self._flags: dict[str, list[str]] = {}
        self._verdicts: dict[str, str] = {}
        self._judged_days: dict[str, set[str]] = {}
        self._obs_counters: dict[str, int] = {}

    def register_rack(self, rack_id: str, type_id: str, capability_kw: float) -> None:
        """登记一簇：同型分组、可用容量与初始状态。"""

        self._lifecycle.register(rack_id)
        self._types[rack_id] = type_id
        self._capability_kw[rack_id] = float(capability_kw)
        self._flags.setdefault(rack_id, [])
        self._verdicts.setdefault(rack_id, VERDICT_SUSPENDED)
        self._judged_days.setdefault(rack_id, set())

    # ------------------------------------------------------------------
    # 功率分配（协调视图）
    # ------------------------------------------------------------------

    def dispatch(self, total_kw: float) -> dict:
        """设定全站总出力，并在在运簇间按能力分摊。

        观察期簇不参与分摊：它的限功率是复投评估的一部分，不能被调度
        重分摊冲掉；停用簇份额恒为零。
        """

        fixed = {
            rack_id: kw
            for rack_id, kw in self._allocation_kw.items()
            if self._lifecycle.state(rack_id) == STATE_OBSERVATION
        }
        eligible = [rack_id for rack_id in self._types if self._lifecycle.state(rack_id) in _RUNNING_STATES]
        caps = {rack_id: (0.0, self._capability_kw[rack_id]) for rack_id in eligible}
        share = allocate(float(total_kw) - sum(fixed.values()), caps)
        self._allocation_kw = {**share, **fixed}
        self._publish("dispatch", total_kw=round(float(total_kw), 3))
        return {"total_kw": float(total_kw), "allocation_kw": dict(self._allocation_kw)}

    # ------------------------------------------------------------------
    # 每日评估
    # ------------------------------------------------------------------

    def review(self, day: str, telemetry: dict[str, list[dict]]) -> dict:
        """跑一天的评估：特征 → 判定 → 必要的停用转移，返回当日报告。"""

        protection = self._protection_scan(telemetry)
        day_features: dict[str, DayFeature] = {}
        unknown: list[str] = []
        for rack_id, raw_samples in telemetry.items():
            if rack_id not in self._types:
                unknown.append(rack_id)
                continue
            samples = [JointSample.from_dict(rack_id, raw) for raw in raw_samples]
            version = self._versions.get((rack_id, day), 0) + 1
            feature = aggregate_day(rack_id, day, samples, self._limits, version=version)
            self._versions[(rack_id, day)] = version
            self._features.setdefault(rack_id, {})[day] = feature
            day_features[rack_id] = feature
            self._ledger.append("feature", rack_id, day, feature.as_dict())

        verdicts: dict[str, str] = {}
        scores_out: dict[str, dict] = {}
        transfers: list[dict] = []
        for type_id in sorted(set(self._types.values())):
            members = [rack_id for rack_id, kind in self._types.items() if kind == type_id]
            group = {
                rack_id: day_features[rack_id]
                for rack_id in members
                if rack_id in day_features and self._lifecycle.state(rack_id) != STATE_OUT_OF_SERVICE
            }
            if not group:
                continue
            scores = rank_day(group, self._limits)
            for rack_id, score in scores.items():
                scores_out[rack_id] = score.as_dict()
                state = self._lifecycle.state(rack_id)
                if state == STATE_OBSERVATION:
                    self._record_observation(rack_id, day, score)
                    continue
                if day in self._judged_days[rack_id]:
                    # 同一天的重算不改写已落台账的判定，只保留上面的新版本特征。
                    continue
                self._judged_days[rack_id].add(day)
                verdict = self._judge(rack_id, day, score)
                verdicts[rack_id] = verdict
        for rack_id in sorted(self._types):
            if self._lifecycle.state(rack_id) == STATE_DEGRADED:
                transfers.append(self._isolate(rack_id, day))
        return {
            "day": day,
            "features": {rack_id: feature.as_dict() for rack_id, feature in day_features.items()},
            "scores": scores_out,
            "verdicts": verdicts,
            "transfers": [item for item in transfers if item],
            "protection": protection,
            "unknown": unknown,
            "states": self._lifecycle.states(),
        }

    # ------------------------------------------------------------------
    # 处理后复投
    # ------------------------------------------------------------------

    def repair_done(self, rack_id: str, day: str) -> dict:
        """连接点处理完成：限功率并入，进入观察期。

        观察期必须带负载运行——连接点没有电流，内阻与温升特征都无从谈起，
        空转观察不算观察。
        """

        if self._lifecycle.state(rack_id) != STATE_OUT_OF_SERVICE:
            raise InterlockError(f"{rack_id} 不在停用态，不能进入观察期", component="joint")
        powers = self._power_view(extra=rack_id)
        plan = plan_reentry(powers, rack_id, join_kw=self._limits.reentry_power_kw)
        log = execute_plan(plan, self._apply, self._confirm)
        self._obs_counters[rack_id] = self._obs_counters.get(rack_id, 0) + 1
        episode = f"obs-{rack_id}-{self._obs_counters[rack_id]}"
        self._lifecycle.transition(rack_id, STATE_OBSERVATION, episode=episode)
        record = self._ledger.append(
            "decision", rack_id, day, {"action": "enter_observation", "episode": episode, "plan": plan.as_dict()}
        )
        self._publish(
            "enter_observation",
            rack_id=rack_id,
            message=f"{rack_id} 处理完成，限功率 {self._limits.reentry_power_kw:.0f}kW 进入观察期",
            episode=episode,
        )
        return {"rack_id": rack_id, "state": STATE_OBSERVATION, "episode": episode, "steps": log, "ledger_seq": record["seq"]}

    def close_observation(self, rack_id: str, day: str) -> dict:
        """观察期满：依据台账里冻结的观察记录决定复投或退回。

        评估只读台账（每天首次落台账的观察记录），特征库后来怎么重算都不
        影响结论。合格则按能力参与全量重分摊正式复投；不合格则把限功率
        同步转回健康簇（同样不能有缺口），退回停用态。
        """

        if self._lifecycle.state(rack_id) != STATE_OBSERVATION:
            raise InterlockError(f"{rack_id} 不在观察期", component="joint")
        episode = self._lifecycle.episode(rack_id)
        seen_days: set[str] = set()
        usable_peaks: list[float] = []
        for record in self._ledger.records(kind="observation", rack_id=rack_id):
            payload = record["payload"]
            if payload.get("episode") != episode or record["day"] in seen_days:
                continue
            seen_days.add(record["day"])
            score = payload.get("score") or {}
            if score.get("flag") == FLAG_INVALID or not payload.get("feature"):
                continue
            usable_peaks.append(max(float(score.get("z_r", 0.0)), float(score.get("z_k", 0.0))))
        passed = (
            len(usable_peaks) >= self._limits.observation_days
            and all(peak < self._limits.z_ok for peak in usable_peaks)
            and trend_slope(usable_peaks) <= self._limits.z_slope_max
        )
        powers = self._power_view(extra=rack_id)
        if passed:
            plan = plan_reentry(powers, rack_id)
            execute_plan(plan, self._apply, self._confirm)
            self._lifecycle.transition(rack_id, STATE_IN_SERVICE)
            decision = "reinstated"
        else:
            plan = plan_outage(powers, rack_id, self._limits.transfer_margin_ratio)
            execute_plan(plan, self._apply, self._confirm)
            self._lifecycle.transition(rack_id, STATE_OUT_OF_SERVICE)
            decision = "rejected"
        record = self._ledger.append(
            "decision",
            rack_id,
            day,
            {
                "action": decision,
                "episode": episode,
                "observed_days": len(usable_peaks),
                "peak_z": round(max(usable_peaks), 3) if usable_peaks else None,
                "plan": plan.as_dict(),
            },
        )
        self._verdicts[rack_id] = VERDICT_SUSPENDED
        self._flags[rack_id] = []
        self._judged_days[rack_id] = set()
        self._publish(
            decision,
            rack_id=rack_id,
            severity="notice" if passed else "alarm",
            message=f"{rack_id} 观察期满，复投{'合格' if passed else '不合格'}",
            episode=episode,
        )
        return {
            "rack_id": rack_id,
            "decision": decision,
            "episode": episode,
            "observed_days": len(usable_peaks),
            "state": self._lifecycle.state(rack_id),
            "ledger_seq": record["seq"],
        }

    # ------------------------------------------------------------------
    # 对外状态
    # ------------------------------------------------------------------

    def snapshot(self) -> JointSnapshot:
        """返回连接点健康快照。"""

        return JointSnapshot(
            states=self._lifecycle.states(),
            verdicts=dict(self._verdicts),
            allocation_kw=dict(self._allocation_kw),
            ledger_count=self._ledger.count(),
            ledger_ok=self._ledger.verify()["ok"],
        )

    def ledger_report(self) -> dict:
        """返回台账校验结果与最近记录，供控制台与交付自检核对。"""

        records = self._ledger.records()
        return {"count": len(records), "verify": self._ledger.verify(), "recent": records[-5:]}

    # ------------------------------------------------------------------
    # 内部：判定、隔离、观察记录
    # ------------------------------------------------------------------

    def _judge(self, rack_id: str, day: str, score: DayScore) -> str:
        flags = self._flags[rack_id]
        flags.append(score.flag)
        del flags[: max(0, len(flags) - self._limits.window_days)]
        verdict = evaluate_persistence(flags, self._limits)
        self._verdicts[rack_id] = verdict
        self._ledger.append(
            "judgment", rack_id, day, {"score": score.as_dict(), "verdict": verdict, "flags": list(flags)}
        )
        state = self._lifecycle.state(rack_id)
        if verdict == VERDICT_DEGRADED and state in (STATE_IN_SERVICE, STATE_WATCH):
            self._lifecycle.transition(rack_id, STATE_DEGRADED)
            self._publish(
                "degraded",
                rack_id=rack_id,
                severity="alarm",
                message=f"{rack_id} 连接点劣化判定成立，安排停用与负载转移",
            )
        elif verdict == VERDICT_WATCH and state == STATE_IN_SERVICE:
            self._lifecycle.transition(rack_id, STATE_WATCH)
            self._publish("watch", rack_id=rack_id, severity="warning", message=f"{rack_id} 连接点指标偏离同型簇")
        elif verdict == VERDICT_NORMAL and state == STATE_WATCH:
            self._lifecycle.transition(rack_id, STATE_IN_SERVICE)
            self._publish("recovered", rack_id=rack_id, message=f"{rack_id} 连接点指标回到同型簇区间内")
        return verdict

    def _isolate(self, rack_id: str, day: str) -> dict:
        """把一簇的出力转移到健康簇并置为停用；裕度不足时保持原状并告警。"""

        try:
            plan = plan_outage(self._power_view(), rack_id, self._limits.transfer_margin_ratio)
            log = execute_plan(plan, self._apply, self._confirm)
        except InterlockError as error:
            self._publish(
                "isolation_blocked",
                rack_id=rack_id,
                severity="alarm",
                message=f"{rack_id} 负载转移被拦下：{error.message}",
            )
            return {"rack_id": rack_id, "isolated": False, "reason": error.message}
        self._lifecycle.transition(rack_id, STATE_OUT_OF_SERVICE)
        record = self._ledger.append("decision", rack_id, day, {"action": "isolated", "plan": plan.as_dict()})
        self._publish(
            "isolated",
            rack_id=rack_id,
            severity="alarm",
            message=f"{rack_id} 连接点劣化，负载已同步转移并停用",
            plan=plan.as_dict(),
        )
        return {"rack_id": rack_id, "isolated": True, "plan": plan.as_dict(), "steps": log, "ledger_seq": record["seq"]}

    def _record_observation(self, rack_id: str, day: str, score: DayScore) -> None:
        if day in self._judged_days[rack_id]:
            # 观察期数据以当天首次落台账为准，重算不覆盖。
            return
        self._judged_days[rack_id].add(day)
        feature = self._features.get(rack_id, {}).get(day)
        self._ledger.append(
            "observation",
            rack_id,
            day,
            {
                "episode": self._lifecycle.episode(rack_id),
                "score": score.as_dict(),
                "feature": feature.as_dict() if feature else None,
            },
        )

    def _protection_scan(self, telemetry: dict[str, list[dict]]) -> list[str]:
        """绝对温度保护：连接点温度越限直接告警，与劣化判定口径无关。"""

        tripped: list[str] = []
        for rack_id, raw_samples in telemetry.items():
            peak = max((float(raw.get("t_conn_c", 0.0)) for raw in raw_samples), default=None)
            if peak is not None and peak >= self._limits.protect_t_conn_c:
                tripped.append(rack_id)
                self._publish(
                    "protection_alarm",
                    rack_id=rack_id,
                    severity="alarm",
                    message=f"{rack_id} 连接点温度 {peak:.1f}℃ 越过绝对保护限值",
                    t_conn_c=round(peak, 2),
                )
        return tripped

    # ------------------------------------------------------------------
    # 内部：功率视图与执行接线
    # ------------------------------------------------------------------

    def _power_view(self, extra: str | None = None) -> dict[str, ClusterPower]:
        included = [
            rack_id
            for rack_id in self._types
            if self._lifecycle.state(rack_id) in _RUNNING_STATES or rack_id == extra
        ]
        return {
            rack_id: ClusterPower(
                rack_id=rack_id,
                p_now_kw=self._allocation_kw.get(rack_id, 0.0),
                p_max_kw=self._capability_kw[rack_id],
                ramp_kw=self._limits.ramp_step_kw,
            )
            for rack_id in included
        }

    def _apply(self, setpoints: dict[str, float]) -> None:
        self._allocation_kw.update(setpoints)

    def _confirm(self, setpoints: dict[str, float]) -> bool:
        # 真实站控里这里读各簇遥测核对执行结果；本内核核对分配表已原子落定。
        return all(abs(self._allocation_kw.get(rack_id, 0.0) - kw) <= 1.0 for rack_id, kw in setpoints.items())

    def _publish(self, action: str, *, severity: str = "notice", message: str = "", **payload: object) -> None:
        self._bus.publish(
            Event.of("joint", "joint", action, station=self._station, severity=severity, message=message, **payload)
        )
