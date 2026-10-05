"""连接点遥测样本与单样本特征折算。

两个观测量从同一帧遥测里折算出来：

- 等效连接电阻 ``r = (V_簇侧 - V_母线侧) / I``，只取放电且电流足够大的样本——
  电流太小时毫欧级连接压降会淹没在测量噪声里；
- 归一化温升系数 ``k = (T_连接点 - T_环境) / I²``。稳态下连接点发热功率为
  ``I²R``，温升近似正比于 ``I²R``，除以 ``I²`` 后负载差异被归一化掉；取对
  环境温度的差值，季节与天气这个共模量也被归一化掉。

连接点劣化早期，表面看不出、温升也不明显，但这两个量会先动——它们是判定的
全部输入，因此折算时的每一道质量门（准稳态、电流下限、合理性区间）都是在
防止"坏数据算出好结论"。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import JointLimits


@dataclass(frozen=True)
class JointSample:
    """一帧簇级连接点遥测。"""

    ts: float
    rack_id: str
    current_a: float
    v_terminal_v: float
    v_bus_v: float
    t_conn_c: float
    t_amb_c: float
    online: bool = True

    @classmethod
    def from_dict(cls, rack_id: str, raw: dict) -> "JointSample":
        """把控制台／采集层送上来的映射还原成样本。"""

        return cls(
            ts=float(raw.get("ts", 0.0)),
            rack_id=rack_id,
            current_a=float(raw.get("current_a", 0.0)),
            v_terminal_v=float(raw.get("v_terminal_v", 0.0)),
            v_bus_v=float(raw.get("v_bus_v", 0.0)),
            t_conn_c=float(raw.get("t_conn_c", 0.0)),
            t_amb_c=float(raw.get("t_amb_c", 25.0)),
            online=bool(raw.get("online", True)),
        )


def steady_mask(samples: list[JointSample], rated_current_a: float, band_ratio: float) -> list[bool]:
    """标记准稳态样本：前帧或后帧电流波动不超过额定值的一定比例。

    温升有热惯性，电流剧烈变化时 k 会失真，阶跃中的样本不参与折算。
    """

    band = float(rated_current_a) * float(band_ratio)
    mask: list[bool] = []
    for index, sample in enumerate(samples):
        steady = False
        if index > 0 and abs(samples[index - 1].current_a - sample.current_a) <= band:
            steady = True
        if index + 1 < len(samples) and abs(samples[index + 1].current_a - sample.current_a) <= band:
            steady = True
        mask.append(steady)
    return mask


def sample_metrics(samples: list[JointSample], limits: JointLimits) -> list[tuple[float, float, float]]:
    """逐帧折算 ``(ts, r_mohm, k_c_per_ka2)``，不合格样本直接丢弃。"""

    ordered = sorted(samples, key=lambda item: item.ts)
    steady = steady_mask(ordered, limits.rated_current_a, limits.steady_band_ratio)
    rows: list[tuple[float, float, float]] = []
    for sample, is_steady in zip(ordered, steady):
        if not sample.online or not is_steady:
            continue
        if sample.current_a < limits.min_current_a:
            continue
        drop_v = sample.v_terminal_v - sample.v_bus_v
        r_mohm = drop_v / sample.current_a * 1000.0
        if not 0.0 <= r_mohm <= limits.r_max_mohm:
            continue
        delta_t = sample.t_conn_c - sample.t_amb_c
        if not limits.dt_min_c <= delta_t <= limits.dt_max_c:
            continue
        current_ka = sample.current_a / 1000.0
        rows.append((sample.ts, r_mohm, delta_t / (current_ka * current_ka)))
    return rows
