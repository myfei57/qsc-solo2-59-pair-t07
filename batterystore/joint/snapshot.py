"""连接点健康快照。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointSnapshot:
    """连接点值守某一时刻的状态。"""

    criterion: str
    states: dict
    dossiers: dict
    total_kw: float
    transfers: int

    def as_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "states": self.states,
            "dossiers": self.dossiers,
            "total_kw": self.total_kw,
            "transfers": self.transfers,
        }
