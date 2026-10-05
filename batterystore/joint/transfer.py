"""簇间负载转移。

停用劣化簇会改变在运簇的功率分配；转移过程的第一不变量是**总量不得出现
缺口**：任何一步执行后，在运总量都不得低于转移前的指令总量。为此每步先
加载后减载——健康簇按裕度比例先爬坡，减载侧的下降量永远不超过本步已经
加上的量与「超出目标总量的富余」二者之一；目标总量低于当前总量时直接
联锁拒绝，不生成任何轨迹。
"""

from __future__ import annotations

from ..errors import InterlockError

_EPS = 1e-6
_MAX_STEPS = 4096


def _total(plan: dict[str, float]) -> float:
    return sum(plan.values())


def _min_total_ratio(initial: dict[str, float], steps: list[dict[str, float]]) -> float:
    base = _total(initial)
    if base <= 0 or not steps:
        return 1.0
    return min(_total(step) for step in steps) / base


def plan_rebalance(now: dict[str, float], target: dict[str, float], step_kw: float) -> list[dict[str, float]]:
    """生成从当前分配到目标分配的逐步轨迹（每步总量不降）。"""

    if _total(target) < _total(now) - _EPS:
        raise InterlockError("目标总量低于当前总量，无法保证转移无缺口", component="joint")
    initial_total = _total(now)
    current = {rack_id: float(kw) for rack_id, kw in now.items()}
    steps: list[dict[str, float]] = []
    for _ in range(_MAX_STEPS):
        ups = {r: min(step_kw, target[r] - current[r]) for r in current if target[r] - current[r] > _EPS}
        downs = {r: min(step_kw, current[r] - target[r]) for r in current if current[r] - target[r] > _EPS}
        if not ups and not downs:
            return steps
        nxt = dict(current)
        raised = 0.0
        for rack_id, delta in ups.items():
            nxt[rack_id] += delta
            raised += delta
        budget = max(raised, _total(nxt) - _total(target))
        for rack_id, delta in downs.items():
            cut = min(delta, budget)
            nxt[rack_id] -= cut
            budget -= cut
        if _total(nxt) < initial_total - _EPS:  # 不变量自检：任何一步都不得出现缺口
            raise InterlockError("转移轨迹出现出力缺口", component="joint")
        steps.append(nxt)
        current = nxt
    raise InterlockError("转移步数超限", component="joint")


def plan_offload(
    allocation: dict[str, float],
    capacity: dict[str, float],
    outgoing: str,
    recipients: list[str],
    step_kw: float,
) -> dict:
    """把一簇的负载按健康簇裕度比例转移出去；裕度不足时部分卸载。"""

    need = float(allocation[outgoing])
    headroom = {r: max(0.0, float(capacity[r]) - float(allocation[r])) for r in recipients}
    available = sum(headroom.values())
    if need > _EPS and not recipients:
        raise InterlockError("没有可接力的健康簇", component="joint")
    targets = dict(allocation)
    if available >= need - _EPS:
        for rack_id in recipients:
            share = need * headroom[rack_id] / available if available > _EPS else 0.0
            targets[rack_id] = float(allocation[rack_id]) + share
        targets[outgoing] = 0.0
        partial = False
    else:
        for rack_id in recipients:
            targets[rack_id] = float(capacity[rack_id])
        targets[outgoing] = need - available
        partial = True
    steps = plan_rebalance(allocation, targets, step_kw)
    return {
        "targets": targets,
        "steps": steps,
        "partial": partial,
        "shortage_kw": max(0.0, need - available),
        "min_total_ratio": _min_total_ratio(allocation, steps),
    }


def plan_even(
    allocation: dict[str, float],
    capacity: dict[str, float],
    active: list[str],
    step_kw: float,
) -> dict:
    """在在运簇之间按容量比例重新均分负载（复投后调用）。"""

    if not active:
        raise InterlockError("没有在运簇，无法重新分配", component="joint")
    total = _total(allocation)
    cap = sum(float(capacity[rack_id]) for rack_id in active)
    targets = {
        rack_id: (total * float(capacity[rack_id]) / cap if rack_id in active else 0.0) for rack_id in allocation
    }
    steps = plan_rebalance(allocation, targets, step_kw)
    return {"targets": targets, "steps": steps, "min_total_ratio": _min_total_ratio(allocation, steps)}
