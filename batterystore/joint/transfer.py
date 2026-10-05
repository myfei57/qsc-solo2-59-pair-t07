"""无缺口负载转移。

停用一簇会改变在运功率分配，但全站出力不能出现缺口。做法是把转移拆成
若干步：每一步接收簇增加的功率与退出簇减少的功率严格相等，每一步的合计
都等于转移前的总出力——不变式在计划构造时成立，不靠事后校验。

接收簇按可用裕度比例分摊：``share_i = amount · h_i / Σh``，由于
``amount ≤ Σh``，必有 ``share_i ≤ h_i``，数学上不会越限，不需要迭代
水位法。执行时逐步下发并逐步确认，确认失败沿原路径回滚到转移前。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import InterlockError

TOL_KW = 1e-6


@dataclass(frozen=True)
class ClusterPower:
    """一簇的当前出力与调节能力。"""

    rack_id: str
    p_now_kw: float
    p_max_kw: float
    ramp_kw: float
    p_min_kw: float = 0.0


@dataclass(frozen=True)
class TransferPlan:
    """一次负载转移计划：初始点、逐步设定值、终点与守恒总量。"""

    initial: dict[str, float]
    steps: tuple[dict[str, float], ...]
    final: dict[str, float]
    total_kw: float

    def as_dict(self) -> dict:
        return {
            "total_kw": round(self.total_kw, 3),
            "steps": len(self.steps),
            "initial": {key: round(value, 3) for key, value in sorted(self.initial.items())},
            "final": {key: round(value, 3) for key, value in sorted(self.final.items())},
        }


def allocate(total_kw: float, caps: dict[str, tuple[float, float]]) -> dict[str, float]:
    """把总出力按可用容量比例分摊到各簇，结果不越任何簇的上下限。"""

    if not caps:
        raise InterlockError("没有可分配的簇", component="joint")
    low = {rack_id: bound[0] for rack_id, bound in caps.items()}
    high = {rack_id: bound[1] for rack_id, bound in caps.items()}
    sum_low = sum(low.values())
    sum_high = sum(high.values())
    if total_kw < sum_low - TOL_KW or total_kw > sum_high + TOL_KW:
        raise InterlockError(
            f"总出力 {total_kw:.1f}kW 超出可行域 [{sum_low:.1f}, {sum_high:.1f}]kW",
            component="joint",
        )
    total = min(max(float(total_kw), sum_low), sum_high)
    rooms = {rack_id: high[rack_id] - low[rack_id] for rack_id in caps}
    room_total = sum(rooms.values())
    result = dict(low)
    if room_total > TOL_KW:
        remaining = total - sum_low
        for rack_id in caps:
            result[rack_id] = low[rack_id] + remaining * rooms[rack_id] / room_total
    return {rack_id: min(max(value, low[rack_id]), high[rack_id]) for rack_id, value in result.items()}


def plan_reallocation(current: dict[str, float], target: dict[str, float], ramps: dict[str, float]) -> TransferPlan:
    """构造从当前分配到目标分配的同步反向爬坡计划。

    目标合计必须等于当前合计，否则"无缺口"无从谈起。步数取各簇调节量与
    爬坡能力之比的最大值，每步线性插值——任何一步的合计都等于总出力。
    """

    if set(current) != set(target):
        raise InterlockError("重分配前后簇集合不一致", component="joint")
    total = sum(current.values())
    if abs(sum(target.values()) - total) > 1e-3:
        raise InterlockError("目标合计必须等于当前合计，否则必然出现出力缺口", component="joint")
    steps_count = 1
    for rack_id, now_kw in current.items():
        ramp = max(float(ramps.get(rack_id, 0.0)), TOL_KW)
        steps_count = max(steps_count, math.ceil(abs(target[rack_id] - now_kw) / ramp - 1e-9))
    steps: list[dict[str, float]] = []
    for index in range(1, steps_count + 1):
        ratio = index / steps_count
        step = {rack_id: now_kw + (target[rack_id] - now_kw) * ratio for rack_id, now_kw in current.items()}
        if abs(sum(step.values()) - total) > 1e-3:
            raise InterlockError("转移计划破坏了总量守恒", component="joint")
        steps.append(step)
    return TransferPlan(dict(current), tuple(steps), dict(target), total)


def plan_outage(powers: dict[str, ClusterPower], out_id: str, margin_ratio: float) -> TransferPlan:
    """规划停用一簇：其出力按裕度比例转给其余簇，合计不变。

    健康簇的可用裕度必须覆盖待转移功率并留有裕度系数，否则拒绝出计划——
    宁可保持劣化簇在运并告警，也不能制造出力缺口。
    """

    if out_id not in powers:
        raise InterlockError(f"未知簇: {out_id}", component="joint")
    leaving = powers[out_id]
    move_kw = leaving.p_now_kw - leaving.p_min_kw
    headroom = sum(p.p_max_kw - p.p_now_kw for rack_id, p in powers.items() if rack_id != out_id)
    if headroom < move_kw * (1.0 + margin_ratio):
        raise InterlockError(
            f"健康簇裕度 {headroom:.1f}kW 不足以承接 {move_kw:.1f}kW（含裕度系数）",
            component="joint",
        )
    total = sum(p.p_now_kw for p in powers.values())
    caps = {rack_id: (p.p_min_kw, p.p_max_kw) for rack_id, p in powers.items() if rack_id != out_id}
    final = allocate(total - leaving.p_min_kw, caps)
    final[out_id] = leaving.p_min_kw
    current = {rack_id: p.p_now_kw for rack_id, p in powers.items()}
    ramps = {rack_id: p.ramp_kw for rack_id, p in powers.items()}
    return plan_reallocation(current, final, ramps)


def plan_reentry(powers: dict[str, ClusterPower], join_id: str, join_kw: float | None = None) -> TransferPlan:
    """规划一簇并入：其余簇同步减出，总量不变。

    ``join_kw`` 给定时并入到该功率（观察期限功率）；缺省时按能力比例参与
    全量重分摊（正式复投）。
    """

    if join_id not in powers:
        raise InterlockError(f"未知簇: {join_id}", component="joint")
    total = sum(p.p_now_kw for p in powers.values())
    caps = {rack_id: (p.p_min_kw, p.p_max_kw) for rack_id, p in powers.items()}
    if join_kw is None:
        final = allocate(total, caps)
    else:
        joiner = powers[join_id]
        if not joiner.p_min_kw <= join_kw <= joiner.p_max_kw:
            raise InterlockError(f"并入功率 {join_kw:.1f}kW 超出该簇允许区间", component="joint")
        others = {rack_id: bound for rack_id, bound in caps.items() if rack_id != join_id}
        final = allocate(total - join_kw, others)
        final[join_id] = float(join_kw)
    current = {rack_id: p.p_now_kw for rack_id, p in powers.items()}
    ramps = {rack_id: p.ramp_kw for rack_id, p in powers.items()}
    return plan_reallocation(current, final, ramps)


def execute_plan(plan: TransferPlan, apply_setpoints, confirm, tol_kw: float = 1.0) -> list[dict]:
    """逐步执行转移计划：下发 → 确认，确认失败沿原路径回滚。

    ``apply_setpoints`` 与 ``confirm`` 由调用方接线（真实站控里 confirm
    读各簇遥测，本内核里核对分配表），本函数只保证顺序与回滚路径。
    """

    log: list[dict] = []
    executed: list[dict[str, float]] = []
    for index, step in enumerate(plan.steps, start=1):
        apply_setpoints(step)
        if confirm(step):
            executed.append(step)
            log.append({"step": index, "ok": True, "total_kw": round(sum(step.values()), 3)})
            continue
        for back in list(reversed(executed)) + [plan.initial]:
            apply_setpoints(back)
            if not confirm(back):
                log.append({"step": index, "ok": False, "rolled_back": False})
                raise InterlockError("负载转移确认失败且回滚未完成，需人工核对出力", component="joint")
        log.append({"step": index, "ok": False, "rolled_back": True})
        raise InterlockError(f"负载转移第 {index} 步确认失败，已回滚到转移前", component="joint")
    return log
