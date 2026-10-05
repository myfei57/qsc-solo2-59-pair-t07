"""连接点健康快照。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointSnapshot:
    """连接点健康组件的对外状态。"""

    states: dict[str, str]
    verdicts: dict[str, str]
    allocation_kw: dict[str, float]
    ledger_count: int
    ledger_ok: bool

    def as_dict(self) -> dict:
        return {
            "states": dict(self.states),
            "verdicts": dict(self.verdicts),
            "allocation_kw": {key: round(value, 3) for key, value in sorted(self.allocation_kw.items())},
            "ledger": {"count": self.ledger_count, "ok": self.ledger_ok},
        }
