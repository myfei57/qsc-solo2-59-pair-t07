"""连接点健康状态机。

::

    in_service → watch →（恢复）in_service
              ↘ degraded → out_of_service → observation →（合格）in_service
                                                  ↑（不合格）┘

degraded 只是判定结论，真正隔离要等负载转移完成；observation 是处理后的
限功率试运行，观察期满才允许复投。所有迁移都走 :meth:`transition`，非法
迁移直接拒绝——状态机之外没有旁路（比如跳过观察期直接复投）。
"""

from __future__ import annotations

from ..errors import InterlockError

STATE_IN_SERVICE = "in_service"
STATE_WATCH = "watch"
STATE_DEGRADED = "degraded"
STATE_OUT_OF_SERVICE = "out_of_service"
STATE_OBSERVATION = "observation"

_TRANSITIONS = {
    STATE_IN_SERVICE: {STATE_WATCH, STATE_DEGRADED},
    STATE_WATCH: {STATE_IN_SERVICE, STATE_DEGRADED},
    STATE_DEGRADED: {STATE_OUT_OF_SERVICE},
    STATE_OUT_OF_SERVICE: {STATE_OBSERVATION},
    STATE_OBSERVATION: {STATE_IN_SERVICE, STATE_OUT_OF_SERVICE},
}


class JointLifecycle:
    """按簇跟踪健康状态与当前观察期编号。"""

    def __init__(self) -> None:
        self._states: dict[str, str] = {}
        self._episodes: dict[str, str] = {}

    def register(self, rack_id: str) -> None:
        """登记一簇，初始为在运态（已登记的簇保持原状态）。"""

        self._states.setdefault(rack_id, STATE_IN_SERVICE)

    def state(self, rack_id: str) -> str:
        """返回簇当前状态，未登记按在运处理。"""

        return self._states.get(rack_id, STATE_IN_SERVICE)

    def states(self) -> dict[str, str]:
        """返回全部簇状态。"""

        return dict(self._states)

    def episode(self, rack_id: str) -> str:
        """返回簇当前观察期编号，不在观察期返回空串。"""

        return self._episodes.get(rack_id, "")

    def transition(self, rack_id: str, target: str, *, episode: str = "") -> dict:
        """执行一次状态迁移，非法迁移抛 :class:`InterlockError`。"""

        current = self.state(rack_id)
        if target not in _TRANSITIONS.get(current, set()):
            raise InterlockError(f"连接点状态不允许 {current} → {target}", component="joint")
        self._states[rack_id] = target
        if episode:
            self._episodes[rack_id] = episode
        return {"rack_id": rack_id, "from": current, "to": target, "episode": self._episodes.get(rack_id, "")}
