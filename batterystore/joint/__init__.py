"""连接点健康管理：劣化判定、停用转移、复投观察。

判定口径选定"同型簇相对位次"（理由见 :mod:`batterystore.joint.ranking`），
负载转移保证全站出力无缺口（见 :mod:`batterystore.joint.transfer`），观察期
数据只增不改、重算不覆盖（见 :mod:`batterystore.joint.ledger`）。
"""

from __future__ import annotations

from .features import DayFeature, aggregate_day
from .ledger import JointLedger
from .lifecycle import JointLifecycle
from .monitor import JointHealthMonitor
from .ranking import DayScore, evaluate_persistence, rank_day
from .sample import JointSample
from .snapshot import JointSnapshot
from .transfer import ClusterPower, TransferPlan, execute_plan, plan_outage, plan_reentry

__all__ = [
    "ClusterPower",
    "DayFeature",
    "DayScore",
    "JointHealthMonitor",
    "JointLedger",
    "JointLifecycle",
    "JointSample",
    "JointSnapshot",
    "TransferPlan",
    "aggregate_day",
    "evaluate_persistence",
    "execute_plan",
    "plan_outage",
    "plan_reentry",
    "rank_day",
]
