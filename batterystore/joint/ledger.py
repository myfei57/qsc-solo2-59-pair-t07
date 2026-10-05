"""连接点健康台账：只增不改的判定与观察记录。

观察期数据是复投决策的唯一依据，必须保证"后来的重算覆盖不了当时的记录"。
三条机制：

1. 台账只提供追加接口，没有更新与删除——覆盖在接口层面就不存在；
2. 每条记录携带前一条的哈希，事后任何改动（哪怕是直接改流水文件）都会在
   校验时断链；
3. 特征重算只会追加更高版本的新记录，观察记录里冻结的是当时的特征值与
   版本号——复投评估只读台账，不回读特征库。
"""

from __future__ import annotations

import hashlib
import json

from ..store.journal import JsonlJournal

GENESIS_HASH = "0" * 64


def _digest(record: dict) -> str:
    body = {key: record[key] for key in ("seq", "kind", "rack_id", "day", "payload", "prev_hash")}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JointLedger:
    """基于分段 JSONL 流水的哈希链台账。"""

    def __init__(self, journal: JsonlJournal) -> None:
        self._journal = journal
        self._records: list[dict] = journal.read_all()

    def append(self, kind: str, rack_id: str, day: str, payload: dict) -> dict:
        """追加一条记录，返回带哈希的完整记录。"""

        record = {
            "seq": len(self._records),
            "kind": kind,
            "rack_id": rack_id,
            "day": day,
            "payload": payload,
            "prev_hash": self._records[-1]["hash"] if self._records else GENESIS_HASH,
        }
        record["hash"] = _digest(record)
        self._journal.append(record)
        self._records.append(record)
        return dict(record)

    def records(self, kind: str | None = None, rack_id: str | None = None) -> list[dict]:
        """按类型与簇过滤记录（内存副本，与落盘流水同源）。"""

        found = self._records
        if kind is not None:
            found = [record for record in found if record["kind"] == kind]
        if rack_id is not None:
            found = [record for record in found if record["rack_id"] == rack_id]
        return [dict(record) for record in found]

    def count(self) -> int:
        """返回累计记录数。"""

        return len(self._records)

    def verify(self) -> dict:
        """重放落盘流水，核对序号、哈希与前向链接。

        校验读的是磁盘上的流水而不是内存副本——防的就是事后改文件。
        """

        records = self._journal.read_all()
        prev_hash = GENESIS_HASH
        broken_at = -1
        for expect_seq, record in enumerate(records):
            if (
                record.get("seq") != expect_seq
                or record.get("prev_hash") != prev_hash
                or record.get("hash") != _digest(record)
            ):
                broken_at = expect_seq
                break
            prev_hash = record["hash"]
        return {"ok": broken_at < 0, "records": len(records), "broken_at": broken_at}
