"""劣化判定口径。

连接点（铜排螺栓、接触器触头、熔丝座）劣化的物理表现是接触电阻缓慢增大：
簇内阻长期上行，发热 ≈ I²·R 使温升同步抬升。两种候选判定口径：

* **固定上升幅度**：相对自身投运基线，内阻上升超过固定比例、温升超过固定
  度数即判劣化；
* **相对位次**：与同站同型簇群体比较，看该簇是否掉到群体之外。

本站采用**相对位次**，理由有三：

1. 环境温度是共模干扰。低温期全部簇的内阻一起抬升、温升基线随季节漂移，
   固定幅度口径没有精确热模型就无法区分「天冷」与「接头劣化」，而热模型
   本身随老化失准；相对位次让同站同型簇互为对照，共模变化在群体内对消，
   两种口径在环境温度不一致时结论相反，正是这个原因。
2. 连接点劣化本质是差异性故障——同一批簇里个别接头先劣化，相对位次正好
   刻画「掉队」；且负载转移的接收方本来就要选「健康簇」，相对位次同时给出
   停用名单与接力名单，判定与处置口径自洽。
3. 相对位次的固有盲区是全体同步劣化（群体一起变差时没有人掉队）。本站用
   噪声下限与最小上行斜率抑制健康群体内的排名噪声，并把固定幅度口径保留
   为 :meth:`JointWarden.compare` 的对照读数，供定期人工复核兜底。

实现：水平取最近若干窗口中位数，与同型健康簇的中位数比较，偏离以群体 MAD
为尺度折算稳健 z 分数；z 越限且自身 Theil–Sen 斜率持续上行的窗口累计到
确认窗口数，才判定连接点劣化并停用。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import JointLimits
from .metric import WindowMetric
from .stats import mad_sigma, median, theil_sen_slope


@dataclass(frozen=True)
class MetricCheck:
    """单项指标（内阻或温升）的一次核对。"""

    level: float
    z: float
    slope: float
    suspect: bool

    def as_dict(self) -> dict:
        return {
            "level": round(self.level, 4),
            "z": round(self.z, 3),
            "slope": round(self.slope, 5),
            "suspect": self.suspect,
        }


@dataclass(frozen=True)
class Verdict:
    """一个窗口的判定结果。"""

    rack_id: str
    window_index: int
    resistance: MetricCheck
    temp_rise: MetricCheck
    streak: int
    confirmed: bool

    def as_dict(self) -> dict:
        return {
            "rack_id": self.rack_id,
            "window_index": self.window_index,
            "resistance": self.resistance.as_dict(),
            "temp_rise": self.temp_rise.as_dict(),
            "streak": self.streak,
            "confirmed": self.confirmed,
        }


class RankCriterion:
    """相对位次口径：z 越限且自身趋势上行，连续若干窗口才确认。"""

    def __init__(self, limits: JointLimits) -> None:
        self._limits = limits
        self._streaks: dict[str, int] = {}

    @staticmethod
    def score(level: float, peer_levels: list[float], floor: float) -> float:
        """计算相对群体的稳健 z 分数（MAD 尺度，带噪声下限）。"""

        center = median(peer_levels)
        return (float(level) - center) / max(mad_sigma(peer_levels), float(floor))

    def assess(
        self,
        rack_id: str,
        history: list[WindowMetric],
        peers_r: list[float],
        peers_t: list[float],
    ) -> Verdict:
        """只读评估：不推进连续计数，供口径对照与观察期打分。"""

        limits = self._limits
        r_level = median([m.resistance_mohm for m in history[-limits.level_windows :]])
        t_level = median([m.temp_rise_c for m in history[-limits.level_windows :]])
        r_slope = theil_sen_slope([m.resistance_mohm for m in history[-limits.trend_windows :]])
        t_slope = theil_sen_slope([m.temp_rise_c for m in history[-limits.trend_windows :]])
        z_r = self.score(r_level, peers_r, limits.resistance_floor_mohm)
        z_t = self.score(t_level, peers_t, limits.temp_rise_floor_c)
        r_suspect = z_r >= limits.z_threshold and r_slope >= limits.resistance_slope_frac * median(peers_r)
        t_suspect = z_t >= limits.z_threshold and t_slope >= limits.temp_rise_slope_c
        streak = self._streaks.get(rack_id, 0) + 1 if r_suspect and t_suspect else 0
        return Verdict(
            rack_id=rack_id,
            window_index=history[-1].window_index,
            resistance=MetricCheck(r_level, z_r, r_slope, r_suspect),
            temp_rise=MetricCheck(t_level, z_t, t_slope, t_suspect),
            streak=streak,
            confirmed=streak >= limits.persistence_windows,
        )

    def evaluate(
        self,
        rack_id: str,
        history: list[WindowMetric],
        peers_r: list[float],
        peers_t: list[float],
    ) -> Verdict:
        """评估并推进该簇的连续越限计数。"""

        verdict = self.assess(rack_id, history, peers_r, peers_t)
        self._streaks[rack_id] = verdict.streak
        return verdict

    def reset(self, rack_id: str) -> None:
        """清零某簇的连续计数（维修后重新积累）。"""

        self._streaks.pop(rack_id, None)

    def streak_of(self, rack_id: str) -> int:
        """返回当前连续越限窗口数。"""

        return self._streaks.get(rack_id, 0)


class FixedRiseCriterion:
    """固定上升幅度口径（仅对照用，不作为停用依据）。

    该口径相对投运初期基线判定，不随环境温度修正：低温期全体簇内阻抬升会
    被误判为劣化。返回的 :class:`Verdict` 中 ``z`` 字段携带相对基线的上升
    幅度（内阻为比例、温升为摄氏度），``slope`` 恒为 0。
    """

    def __init__(self, limits: JointLimits) -> None:
        self._limits = limits

    def evaluate(self, rack_id: str, history: list[WindowMetric]) -> Verdict:
        """按固定幅度核对当前水平，不维护连续计数。"""

        limits = self._limits
        baseline_r = median([m.resistance_mohm for m in history[: limits.level_windows]])
        baseline_t = median([m.temp_rise_c for m in history[: limits.level_windows]])
        r_level = median([m.resistance_mohm for m in history[-limits.level_windows :]])
        t_level = median([m.temp_rise_c for m in history[-limits.level_windows :]])
        rise_r = r_level / baseline_r - 1.0
        rise_t = t_level - baseline_t
        r_suspect = rise_r >= limits.fixed_rise_resistance_frac
        t_suspect = rise_t >= limits.fixed_rise_temp_c
        confirmed = r_suspect and t_suspect
        return Verdict(
            rack_id=rack_id,
            window_index=history[-1].window_index,
            resistance=MetricCheck(r_level, rise_r, 0.0, r_suspect),
            temp_rise=MetricCheck(t_level, rise_t, 0.0, t_suspect),
            streak=limits.persistence_windows if confirmed else 0,
            confirmed=confirmed,
        )
