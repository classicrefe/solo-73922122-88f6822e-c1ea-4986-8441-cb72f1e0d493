"""外部归并排序与有界资源（内存 / 文件句柄）的流式归并核对。

设计要点：

读取与排序（内存有界）
- 结果集以服务端命名游标分批拉取（itersize 受限），不会整表进内存；
- 双路读取共享同一个 MemoryBudget：两侧驻留行的估算字节之和永远不超过
  read_memory_budget，任一侧达到 SORT_CHUNK_ROWS 行或 SORT_CHUNK_BYTES
  字节即排序落盘为一个 run，并立即把对应字节归还预算；
- 窄宽混合行按实际字节计入预算，分块大小随行宽自适应，不会因单行很宽而
  突破内存上限，也不会拒收任何合法 bigint/text 行；
- 文本逐行 pickle 流式落盘，不截断任何字符，NULL/空串原样保留。

归并（句柄有界）
- 采用自适应有界扇入的分层（多相）归并：每层最多 effective_fan_in() 个有序
  流参与，任意时刻同时打开的文件数不超过当前扇入（单流逐行 unpickle，
  堆中每流只驻留一行），run 再多也不会耗尽文件句柄；
- 中间层结果写成下一层 run，最终层直接以惰性流交给核对，不做整物化；
- 比较按行内容与“出现次数”进行，NULL 与空串天然不同（Python None != ""），
  不使用整表哈希做判等；
- 最多保留 MAX_DIFF_TYPES 种差异用于展示，计数器仍覆盖全量数据。
"""
from __future__ import annotations

import heapq
import os
import pickle
import resource
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# 保守兜底扇入：无法读取 NOFILE 或测试显式指定时使用。
# 取 8 时，即便进程 NOFILE 低至 32，两侧归并 + DB 连接 + 标准流也够用。
FAN_IN = 8
# 自适应扇入的硬上限：堆中每流只驻留一行，256 流的内存与 log(k) 成本仍可忽略
HARD_FAN_IN_CAP = 256
# 自适应下限：再低的 NOFILE 也保证每侧至少能同时归并 4 个流
MIN_FAN_IN = 4
# 预留给标准流、3 条 PG 连接、socket 与临时 mkstemp 的文件描述符余量
_FD_RESERVE = 16
# 测试钩子：置位后 effective_fan_in() 固定返回该值
_fan_in_override: int | None = None


def set_fan_in_override(value: int | None) -> None:
    """仅供测试：固定自适应扇入（None 恢复自动探测）。"""
    global _fan_in_override
    _fan_in_override = value


def effective_fan_in() -> int:
    """按进程软 NOFILE 计算两侧归并各自可用的安全扇入。

    两侧最终归并流会同时打开，因此每侧分得 (NOFILE - 预留) / 2；
    常见容器 NOFILE=1024 时得到 256（封顶），普通 run 数量下单轮即可归并、
    不产生中间层文件；NOFILE=32 的受限环境下降到 8，自动走分层归并。
    """
    if _fan_in_override is not None:
        return _fan_in_override
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return FAN_IN
    if soft < 0 or soft == resource.RLIM_INFINITY:
        return HARD_FAN_IN_CAP
    fan = (soft - _FD_RESERVE) // 2
    return max(MIN_FAN_IN, min(HARD_FAN_IN_CAP, fan))

# 一行 = tuple，元素仅允许 int / str / None
Row = tuple[Any, ...]


def sort_key(row: Row) -> tuple:
    """全序键：None 排在最前，其次 int，再次 str；同类型按自然序。"""
    key: list[tuple[int, Any]] = []
    for v in row:
        if v is None:
            key.append((0, 0))
        elif isinstance(v, int):
            key.append((1, v))
        else:  # str（含空串 ""）
            key.append((2, v))
    return tuple(key)


def rows_equal(a: Row, b: Row) -> bool:
    return a == b  # tuple 逐元素相等：None 与 "" 不相等，符合需求


def row_bytes(row: Row) -> int:
    """估算一行在进程内的驻留字节，用于内存预算记账。

    只统计不可共享的负载：str 按字符数（UCS-1/2/4 最坏每字符 4 字节，
    取 len 是保守下界方向的折中，配合预算余量使用），int/None 计固定开销。
    排序产生的键对象等额外驻留由预算余量吸收，不改变完整行值。
    """
    total = 0
    for v in row:
        if v is None:
            total += 16
        elif isinstance(v, int):
            total += 28
        else:
            # 字符串对象头 + 字符负载（按每字符至少 1 字节记账，16KiB 文本
            # 的实际驻留与该估算同量级）
            total += 49 + len(v)
    return total + 56  # tuple 本体与引用槽


class MemoryBudget:
    """双路读取共享的内存预算（字节），带条件变量阻塞获取。

    不变量：已发放字节 total_acquired - total_released 永远 <= limit。
    单次申请即便超过限额也不会被永久阻塞——当预算完全空闲时直接放行，
    保证“极宽单行”这类合法输入不会被拒收或死锁。
    """

    def __init__(self, limit_bytes: int):
        self._limit = int(limit_bytes)
        self._used = 0
        self._cond = threading.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        with self._cond:
            return self._used

    def try_acquire(self, size: int) -> bool:
        """非阻塞申请：当前空闲足够则立即记账并返回 True，否则返回 False。"""
        if size <= 0:
            return True
        with self._cond:
            if self._used + size > self._limit:
                return False
            self._used += size
            return True

    def acquire(
        self,
        size: int,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """阻塞申请预算，可由 is_cancelled() 打断并抛 JobCancelled。

        等待按 0.2s 分段，确保取消/超时信号最迟在一个分段内被受理，
        读取线程不会因卡在条件变量上而跨任务泄漏（其持有的连接也能及时关闭）。
        """
        if size <= 0:
            return
        with self._cond:
            while self._used + size > self._limit and self._used > 0:
                if is_cancelled is not None and is_cancelled():
                    raise JobCancelled()
                self._cond.wait(timeout=0.2)
            self._used += size

    def release(self, size: int) -> None:
        if size <= 0:
            return
        with self._cond:
            self._used -= size
            if self._used < 0:
                self._used = 0
            self._cond.notify_all()


@dataclass
class RunWriter:
    """把分批到达的行切成多个有序 run 文件（逐行流式 pickle）。

    受两个条件约束 flush：行数达到 chunk_rows，或当前缓冲估算字节达到
    chunk_bytes。flush 后把对应字节归还共享内存预算。
    """

    work_dir: str
    chunk_rows: int
    chunk_bytes: int
    budget: MemoryBudget
    is_cancelled: Callable[[], bool] | None = None
    prefix: str = "run-"
    rows_written: int = 0
    run_paths: list[str] = field(default_factory=list)
    _buffer: list[Row] = field(default_factory=list)
    _buffer_bytes: int = 0
    _held_bytes: int = 0

    def add(self, rows: list[Row]) -> None:
        incoming = 0
        for row in rows:
            incoming += row_bytes(row)
        # 活性保证：预算不足以非阻塞申请时，先把自身已缓冲的行落盘归还，
        # 再进入可取消的阻塞等待。否则两侧可能各持一个“未满阈值”的块互相
        # 等待，谁都进不到 flush 而形成死锁（最终只能靠看门狗超时收场）。
        if not self.budget.try_acquire(incoming):
            if self._buffer:
                self._flush(self._buffer)
                self._buffer = []
                self._buffer_bytes = 0
            self.budget.acquire(incoming, self.is_cancelled)
        self._held_bytes += incoming
        try:
            self._buffer.extend(rows)
            self.rows_written += len(rows)
            self._buffer_bytes += incoming
            # 行条件或字节条件任一满足即落盘；字节先触发时整块可能不足 chunk_rows
            while (
                len(self._buffer) >= self.chunk_rows
                or self._buffer_bytes >= self.chunk_bytes
            ):
                self._flush(self._buffer)
                self._buffer = []
                self._buffer_bytes = 0
        except Exception:
            # 出错时释放本次仍持有的全部预算，避免对侧永久饥饿/死锁
            self.budget.release(self._held_bytes)
            self._held_bytes = 0
            raise

    def finish(self) -> list[str]:
        if self._buffer:
            self._flush(self._buffer)
            self._buffer = []
            self._buffer_bytes = 0
        return self.run_paths

    def _flush(self, chunk: list[Row]) -> None:
        chunk.sort(key=sort_key)
        flushed_bytes = sum(row_bytes(r) for r in chunk)
        fd, path = tempfile.mkstemp(
            prefix=self.prefix, suffix=".pkl", dir=self.work_dir
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                # 逐行写，避免 pickle 维护跨对象引用表造成额外驻留
                for row in chunk:
                    pickle.dump(row, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            _quiet_unlink(path)
            raise
        self.run_paths.append(path)
        # 落盘完成才归还预算：排序与写盘期间这些行仍在内存中
        self._buffer_bytes = max(0, self._buffer_bytes - flushed_bytes)
        self._held_bytes -= flushed_bytes
        self.budget.release(flushed_bytes)


# 携带预计算全序键的行：(sort_key, row)
KeyedRow = tuple[tuple, Row]


class KWayMerger:
    """有界句柄的 k 路归并器：同时打开的文件数 == len(paths) <= 扇入。

    上下文退出时确保全部文件描述符关闭（包括取消/异常路径），
    不依赖生成器被 GC 才释放句柄。
    """

    def __init__(self, paths: list[str]):
        if not paths:
            raise ValueError("KWayMerger 需要至少一个 run")
        self._paths = paths
        self._files: list[Any] = []
        self._heap: list[tuple[tuple, int, Row]] = []
        self._iters: list[Iterator[KeyedRow]] = []

    def open(self) -> "KWayMerger":
        try:
            for idx, path in enumerate(self._paths):
                fh = open(path, "rb")
                self._files.append(fh)
                it = self._unpickle_iter(fh)
                self._iters.append(it)
                first = next(it, None)
                if first is not None:
                    key, row = first
                    heapq.heappush(self._heap, (key, idx, row))
        except Exception:
            self.close()
            raise
        return self

    @staticmethod
    def _unpickle_iter(fh: Any) -> Iterator[KeyedRow]:
        while True:
            try:
                row = pickle.load(fh)
            except EOFError:
                return
            yield sort_key(row), row

    def __iter__(self) -> Iterator[KeyedRow]:
        return self

    def __next__(self) -> KeyedRow:
        if not self._heap:
            raise StopIteration
        key, idx, row = heapq.heappop(self._heap)
        nxt = next(self._iters[idx], None)
        if nxt is not None:
            nkey, nrow = nxt
            heapq.heappush(self._heap, (nkey, idx, nrow))
        return key, row

    def close(self) -> None:
        for fh in self._files:
            try:
                fh.close()
            except OSError:
                pass
        self._files.clear()
        self._heap.clear()

    def __enter__(self) -> "KWayMerger":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _write_merged_stream(
    stream: Iterator[KeyedRow], work_dir: str, level: int
) -> str:
    """把一个有序惰性流逐行写成下一层 run 文件（流式，不物化）。"""
    fd, path = tempfile.mkstemp(
        prefix=f"merge-l{level}-", suffix=".pkl", dir=work_dir
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            for _key, row in stream:
                pickle.dump(row, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        _quiet_unlink(path)
        raise
    return path


def cascade_merge(
    run_paths: list[str], work_dir: str
) -> KWayMerger | None:
    """有界句柄分层归并，返回最终层的惰性归并流；无 run 时返回 None。

    扇入取 effective_fan_in()：每层把 run 按 fan_in 个一组归并成下一层文件，
    任意时刻同时打开的文件描述符 <= fan_in。run 数不超过扇入时零中间落盘，
    与单轮 k 路归并成本一致；run 数远超可用句柄时自动多相分层。
    输入 run（含中间层）统一由任务收尾时清空 work_dir 回收，本函数不删除
    任何输入文件——同一份 run 列表可能被再次归并或用于两侧核对，提前删除
    会破坏“完整行值可重复读取”的前提。
    """
    fan_in = effective_fan_in()
    current = list(run_paths)
    level = 0
    while len(current) > fan_in:
        next_level: list[str] = []
        try:
            for start in range(0, len(current), fan_in):
                group = current[start : start + fan_in]
                with KWayMerger(group) as merger:
                    out_path = _write_merged_stream(merger, work_dir, level)
                next_level.append(out_path)
        except Exception:
            for p in next_level:
                _quiet_unlink(p)
            raise
        current = next_level
        level += 1
    if not current:
        return None
    return KWayMerger(current).open()


def _quiet_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@dataclass
class ReconcileStats:
    left_row_total: int = 0
    right_row_total: int = 0
    matched_group_count: int = 0      # 两侧该组次数相等的去重行数
    diff_group_total: int = 0         # 两侧次数不等（含仅一侧存在）的去重行数
    diffs: list[dict[str, Any]] = field(default_factory=list)


class _PeekStream:
    """按全序键聚合等值组的有序流，支持回看一个元素。"""

    def __init__(self, stream: Iterator[KeyedRow] | None):
        self._stream: Iterator[KeyedRow] = stream if stream is not None else iter(())
        self._peek: KeyedRow | None = None
        self._has_peek = False

    def next_group(self) -> tuple[tuple | None, Row | None, int]:
        first = self._peek if self._has_peek else next(self._stream, None)
        self._has_peek = False
        if first is None:
            return None, None, 0
        first_key, first_row = first
        count = 1
        while True:
            nxt = next(self._stream, None)
            if nxt is None:
                return first_key, first_row, count
            if nxt[0] == first_key:
                count += 1
            else:
                self._peek = nxt
                self._has_peek = True
                return first_key, first_row, count


def reconcile_sorted(
    left_runs: list[str],
    right_runs: list[str],
    *,
    left_total: int,
    right_total: int,
    max_diff_types: int,
    is_cancelled: Callable[[], bool],
    work_dir: str | None = None,
) -> ReconcileStats:
    """对两侧已落盘的有序 run 做归并连接式多重集核对。

    任一组两侧次数不同即记为一种差异；差异明细最多保留 max_diff_types 种，
    但 diff_group_total / 总行数等计数始终覆盖全量。
    比较全程复用预计算键，NULL 与空串因键不同而自然区分。
    归并句柄有界：每侧同时打开的 run 文件数不超过 effective_fan_in()。
    """
    stats = ReconcileStats(left_row_total=left_total, right_row_total=right_total)
    if work_dir is None:
        raise ValueError("work_dir 为分层归并必需参数")

    left_merger = cascade_merge(left_runs, work_dir)
    right_merger = cascade_merge(right_runs, work_dir)
    try:
        left = _PeekStream(left_merger)
        right = _PeekStream(right_merger)

        l_key, l_row, l_cnt = left.next_group()
        r_key, r_row, r_cnt = right.next_group()

        while l_key is not None or r_key is not None:
            if is_cancelled():
                raise JobCancelled()

            if l_key is not None and (r_key is None or l_key < r_key):
                _record_diff(stats, l_row, l_cnt, 0, max_diff_types)
                l_key, l_row, l_cnt = left.next_group()
            elif r_key is not None and (l_key is None or r_key < l_key):
                _record_diff(stats, r_row, 0, r_cnt, max_diff_types)
                r_key, r_row, r_cnt = right.next_group()
            else:
                # 键相同即行内容相同（sort_key 对固定列数是单射全序）
                assert l_key is not None and l_key == r_key and rows_equal(l_row, r_row)
                if l_cnt == r_cnt:
                    stats.matched_group_count += 1
                else:
                    _record_diff(stats, l_row, l_cnt, r_cnt, max_diff_types)
                l_key, l_row, l_cnt = left.next_group()
                r_key, r_row, r_cnt = right.next_group()
    finally:
        if left_merger is not None:
            left_merger.close()
        if right_merger is not None:
            right_merger.close()

    return stats


def _record_diff(
    stats: ReconcileStats,
    row: Row,
    left_count: int,
    right_count: int,
    max_diff_types: int,
) -> None:
    stats.diff_group_total += 1
    if len(stats.diffs) < max_diff_types:
        stats.diffs.append(
            {
                "values": list(row),
                "right_count": right_count,
                "left_count": left_count,
            }
        )


class JobCancelled(Exception):
    """核对过程中检测到取消请求。"""
