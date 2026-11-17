"""external_merge 的纯逻辑测试（不依赖数据库）。

覆盖：
- 全序键、NULL 与空串区分、多重集次数统计、差异展示上限等原有语义；
- 宽文本行（16KiB/行）在字节预算下不截断、不混淆；
- 窄宽混合行按字节自适应分块，行值/次数/NULL 语义不变；
- 暂存段（run）远多于可用文件句柄时，有界扇入分层归并仍正确；
- 共享内存预算的硬上限与阻塞获取语义。
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

import pytest

from app.external_merge import (
    FAN_IN,
    JobCancelled,
    KWayMerger,
    MemoryBudget,
    RunWriter,
    cascade_merge,
    effective_fan_in,
    reconcile_sorted,
    row_bytes,
    set_fan_in_override,
    sort_key,
)

WIDE = 16 * 1024  # 需求场景中的宽文本：16KiB/行


def _make_writer(work_dir: str, chunk: int, budget: MemoryBudget,
                 chunk_bytes: int | None = None, prefix: str = "run-") -> RunWriter:
    return RunWriter(
        work_dir=work_dir,
        chunk_rows=chunk,
        chunk_bytes=chunk_bytes if chunk_bytes is not None else 16 * 1024 * 1024,
        budget=budget,
        prefix=prefix,
    )


def _write_runs(work_dir: str, rows: list[tuple], chunk: int,
                budget: MemoryBudget | None = None,
                chunk_bytes: int | None = None) -> list[str]:
    w = _make_writer(
        work_dir, chunk,
        budget if budget is not None else MemoryBudget(256 * 1024 * 1024),
        chunk_bytes,
    )
    for i in range(0, len(rows), max(1, chunk // 2)):
        w.add(rows[i : i + max(1, chunk // 2)])
    return w.finish()


def test_sort_key_null_before_empty_string_before_text() -> None:
    assert sort_key((None,)) < sort_key(("",))
    assert sort_key(("",)) < sort_key(("a",))
    assert sort_key((1,)) < sort_key(("1",))
    assert sort_key((None, 0)) < sort_key((1, 0))


def test_null_and_empty_string_are_different_groups() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        # 同列位置：None 与 "" 必须被视为不同的行内容
        left = [(1, None), (2, "")]
        right = [(1, ""), (2, None)]
        lr = _write_runs(d1, left, 2)
        rr = _write_runs(d2, right, 2)
        stats = reconcile_sorted(
            lr, rr, left_total=2, right_total=2,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        # 4 个等值组互不相同，每个组仅单侧存在
        assert stats.diff_group_total == 4
        assert stats.matched_group_count == 0
        contents = {tuple(x["values"]) for x in stats.diffs}
        assert contents == {(1, None), (2, ""), (1, ""), (2, None)}


def test_null_vs_empty_same_key_column_duplicate_counts() -> None:
    """两侧行内容相同位置上 NULL 与空串的次数分别统计，不会被合并。"""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        left = [(1, None), (1, None), (1, "")]
        right = [(1, None), (1, ""), (1, "")]
        lr = _write_runs(d1, left, 2)
        rr = _write_runs(d2, right, 2)
        stats = reconcile_sorted(
            lr, rr, left_total=3, right_total=3,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        assert stats.diff_group_total == 2
        by_value = {tuple(d["values"]): d for d in stats.diffs}
        assert by_value[(1, None)]["left_count"] == 2
        assert by_value[(1, None)]["right_count"] == 1
        assert by_value[(1, "")]["left_count"] == 1
        assert by_value[(1, "")]["right_count"] == 2


def test_duplicates_counted_as_multiset_order_ignored() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        left = [(1, "a"), (1, "a"), (2, "b"), (3, "c")]
        right = [(3, "c"), (2, "b"), (1, "a")]  # 左侧 (1,a) 重复计入
        lr = _write_runs(d1, left, 2)
        rr = _write_runs(d2, right, 2)
        stats = reconcile_sorted(
            lr, rr, left_total=4, right_total=3,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        assert stats.matched_group_count == 2
        assert stats.diff_group_total == 1
        diff = stats.diffs[0]
        assert diff["values"] == [1, "a"]
        assert diff["left_count"] == 2 and diff["right_count"] == 1


def test_identical_multisets_with_multiple_runs() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        rows = [(i, f"m{i % 3}") for i in range(1000)]
        lr = _write_runs(d1, rows, 37)   # 制造很多 run，强制分层归并
        rr = _write_runs(d2, list(reversed(rows)), 37)
        assert len(lr) > FAN_IN  # 确实需要多相归并
        # 固定小扇入，确保走分层路径而非自适应大单轮
        set_fan_in_override(FAN_IN)
        try:
            stats = reconcile_sorted(
                lr, rr, left_total=1000, right_total=1000,
                max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
            )
        finally:
            set_fan_in_override(None)
        assert stats.diff_group_total == 0
        assert stats.matched_group_count == 1000
        # 排序结果可复现：cascade_merge 产出 (键, 行)，取行后与基准排序一致
        set_fan_in_override(FAN_IN)
        try:
            merger = cascade_merge(lr, d1)
            assert merger is not None
            merged = [row for _key, row in merger]
            merger.close()
        finally:
            set_fan_in_override(None)
        assert merged == sorted(rows, key=sort_key)


def test_diff_display_cap_keeps_full_counts() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        left = [(i, "x") for i in range(50)]
        right: list[tuple] = []
        lr = _write_runs(d1, left, 7)
        rr = _write_runs(d2, right, 7)
        stats = reconcile_sorted(
            lr, rr, left_total=50, right_total=0,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        assert stats.diff_group_total == 50       # 全量计数
        assert len(stats.diffs) == 20             # 展示最多 20 种


def test_cancel_during_reconcile_raises() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        lr = _write_runs(d1, [(i, "a") for i in range(100)], 5)
        rr = _write_runs(d2, [(-i, "b") for i in range(100)], 5)
        with pytest.raises(JobCancelled):
            reconcile_sorted(
                lr, rr, left_total=100, right_total=100,
                max_diff_types=20, is_cancelled=lambda: True, work_dir=d1,
            )


def test_empty_result_sets_are_identical() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        lr = _write_runs(d1, [], 10)
        rr = _write_runs(d2, [], 10)
        stats = reconcile_sorted(
            lr, rr, left_total=0, right_total=0,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        assert stats.diff_group_total == 0
        assert stats.diffs == []


# ---------------------------------------------------------------------------
# 宽行 / 字节预算回归
# ---------------------------------------------------------------------------

def test_wide_16kib_rows_round_trip_without_truncation() -> None:
    """12000 行 x 16KiB 文本：完整行值、重复次数在分块与归并后保持不变。"""
    n_rows = 12000
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        budget = MemoryBudget(40 * 1024 * 1024)  # 与生产默认一致
        # 每行文本带唯一后缀，确保没有任何截断/串号能蒙混过关
        marker_a = "A" * (WIDE - 8)
        marker_b = "B" * (WIDE - 8)
        left = [(i, f"{i:08d}{marker_a}") for i in range(n_rows)]
        right = [(i, f"{i:08d}{marker_b}") for i in range(n_rows)]
        wl = _make_writer(d1, 20000, budget, prefix="l-")
        wr = _make_writer(d2, 20000, budget, prefix="r-")
        for i in range(0, n_rows, 500):
            wl.add(left[i : i + 500])
            wr.add(right[i : i + 500])
        lr, rr = wl.finish(), wr.finish()
        # 16KiB/行 -> 16MiB 字节阈值先于 20000 行阈值触发
        assert all(len(p) > 0 for p in lr + rr)
        assert wl.rows_written == wr.rows_written == n_rows
        stats = reconcile_sorted(
            lr, rr, left_total=n_rows, right_total=n_rows,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        # 文本逐字节不同：左 12000 个仅左侧组 + 右 12000 个仅右侧组，
        # diff_group_total 全量计数；展示的文本长度必须完整
        assert stats.diff_group_total == 2 * n_rows
        assert stats.matched_group_count == 0
        for diff in stats.diffs:
            assert len(diff["values"][1]) == WIDE


def test_wide_identical_rows_match_with_counts_and_null_distinction() -> None:
    """相同宽行（含重复、NULL、空串）经字节分块后多重集核对仍精确。"""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        wide = "W" * WIDE
        # 差异组数控制在展示上限内，确保 NULL/空串都能被直接检视
        left = [
            (1, wide), (1, wide), (1, wide),     # 宽行重复 3 次
            (2, None),                            # NULL
            (3, ""),                              # 空串，与 NULL 不同
            (4, "z"),
        ]
        right = [
            (1, wide), (1, wide),                # 宽行重复 2 次 -> 差异
            (2, ""),                              # 位置类型互换
            (3, None),
            (4, "z"),
        ]
        # 极小 chunk_bytes，强制每一两行就落盘，制造大量段
        lr = _write_runs(d1, left, chunk=2,
                         budget=MemoryBudget(4 * 1024 * 1024), chunk_bytes=4096)
        rr = _write_runs(d2, right, chunk=2,
                         budget=MemoryBudget(4 * 1024 * 1024), chunk_bytes=4096)
        stats = reconcile_sorted(
            lr, rr, left_total=6, right_total=5,
            max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
        )
        by_key = {d["values"][0]: d for d in stats.diffs}
        assert stats.matched_group_count == 1  # 仅 (4, "z") 完全一致
        # 宽行次数差 1 组 + NULL/空串互换的 4 个单侧组 = 5 个差异组
        assert stats.diff_group_total == 5
        assert by_key[1]["left_count"] == 3 and by_key[1]["right_count"] == 2
        assert len(by_key[1]["values"][1]) == WIDE
        # 同一 id 上 NULL 与空串是两个不同的差异组
        null_empty_pairs = {
            (d["values"][0], d["values"][1])
            for d in stats.diffs
            if d["values"][0] in (2, 3)
        }
        assert null_empty_pairs == {(2, None), (2, ""), (3, None), (3, "")}


def test_narrow_wide_mixed_rows_flush_by_bytes() -> None:
    """窄宽混合：字节阈值先触发时按小切块落盘，全部行值与次数无损。"""
    with tempfile.TemporaryDirectory() as d:
        budget = MemoryBudget(8 * 1024 * 1024)
        w = RunWriter(work_dir=d, chunk_rows=20000,
                      chunk_bytes=64 * 1024, budget=budget)
        rows: list[tuple] = []
        # 交替：窄行 + 16KiB 宽行，共 2000 行
        wide = "Q" * WIDE
        for i in range(1000):
            rows.append((i, 0))
            rows.append((10_000 + i, wide))
        for i in range(0, len(rows), 100):
            w.add(rows[i : i + 100])
        runs = w.finish()
        assert len(runs) > 10  # 字节阈值把混合数据切成很多段
        merger = cascade_merge(runs, d)
        assert merger is not None
        merged = [row for _key, row in merger]
        merger.close()
        assert merged == sorted(rows, key=sort_key)
        # 宽行内容未截断
        wide_rows = [r for r in merged if isinstance(r[1], str) and len(r[1]) == WIDE]
        assert len(wide_rows) == 1000


def test_shared_budget_never_exceeds_limit_under_two_writers() -> None:
    """双路并行写入时共享预算的已发放字节永不超过上限。"""
    limit = 2 * 1024 * 1024
    budget = MemoryBudget(limit)
    with tempfile.TemporaryDirectory() as d:
        wl = RunWriter(d, chunk_rows=10_000, chunk_bytes=256 * 1024,
                       budget=budget, prefix="l-")
        wr = RunWriter(d, chunk_rows=10_000, chunk_bytes=256 * 1024,
                       budget=budget, prefix="r-")
        peak = 0
        stop = threading.Event()

        def monitor() -> None:
            nonlocal peak
            while not stop.is_set():
                peak = max(peak, budget.used)
                time.sleep(0.0005)

        wide = "M" * 4096
        def feed(w: RunWriter) -> None:
            for i in range(0, 4000, 200):
                w.add([(i, wide) for _ in range(200)])

        tmon = threading.Thread(target=monitor)
        tmon.start()
        tl = threading.Thread(target=feed, args=(wl,))
        tr = threading.Thread(target=feed, args=(wr,))
        tl.start(); tr.start(); tl.join(); tr.join()
        stop.set(); tmon.join()
        peak = max(peak, budget.used)
        assert budget.used == 0           # flush 后全部归还
        assert peak <= limit              # 硬上限从未被突破


def test_budget_blocks_until_released() -> None:
    budget = MemoryBudget(1000)
    budget.acquire(800)
    acquired = threading.Event()

    def waiter() -> None:
        budget.acquire(500)  # 剩余 200 不够，必须阻塞
        acquired.set()

    t = threading.Thread(target=waiter)
    t.start()
    t.join(timeout=0.3)
    assert not acquired.is_set()          # 预算未归还前拿不到
    budget.release(400)                   # 剩余变 600，满足 500
    t.join(timeout=0.3)
    assert acquired.is_set()
    # 超宽单行：预算完全空闲时直接放行，不会死锁或拒收
    budget.release(900)
    budget.acquire(10_000_000)
    assert budget.used == 10_000_000


def test_budget_wait_is_cancellable() -> None:
    """阻塞在预算上的申请必须能被取消标志打断，读取线程不跨任务泄漏。"""
    budget = MemoryBudget(1000)
    budget.acquire(900)  # 占满，后续申请必然进入等待
    cancel = threading.Event()
    outcome: dict[str, object] = {}

    def waiter() -> None:
        try:
            budget.acquire(500, is_cancelled=cancel.is_set)
            outcome["ok"] = True
        except JobCancelled:
            outcome["cancelled"] = True

    t = threading.Thread(target=waiter)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive()                  # 未取消前持续等待
    cancel.set()
    t.join(timeout=1.0)                  # 一个等待分段（0.2s）内受理
    assert not t.is_alive()
    assert outcome == {"cancelled": True}
    assert budget.used == 900            # 被取消的申请未记账


def test_two_writers_do_not_deadlock_under_tight_budget() -> None:
    """预算小于两个满块之和时，双侧并行写入不得互相等待到超时（活性回归）。

    复现历史缺陷：两侧各持一个未达字节阈值的缓冲、同时申请下一批时，
    旧实现双双阻塞在预算上且无人能 flush，最终看门狗超时。
    """
    # 预算仅能容纳约一个块多一点，且块阈值小于“两侧缓冲之和”
    limit = 200 * 1024
    budget = MemoryBudget(limit)
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        wl = RunWriter(d1, chunk_rows=1_000_000, chunk_bytes=128 * 1024,
                       budget=budget, prefix="l-")
        wr = RunWriter(d2, chunk_rows=1_000_000, chunk_bytes=128 * 1024,
                       budget=budget, prefix="r-")
        wide = "D" * 4096
        errors: list[Exception] = []

        def feed(w: RunWriter) -> None:
            try:
                for _ in range(40):
                    w.add([(i, wide) for i in range(32)])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        tl = threading.Thread(target=feed, args=(wl,))
        tr = threading.Thread(target=feed, args=(wr,))
        tl.start(); tr.start()
        tl.join(timeout=10); tr.join(timeout=10)
        assert not tl.is_alive() and not tr.is_alive(), "双侧写入发生死锁"
        assert errors == []
        assert wl.rows_written == wr.rows_written == 40 * 32
        assert len(wl.run_paths) > 1 and len(wr.run_paths) > 1
        assert budget.used == 0  # 全部落盘归还
        # 落盘行值完整：归并后行数与字节数可核对
        ml = cascade_merge(wl.run_paths, d1)
        mr = cascade_merge(wr.run_paths, d2)
        assert ml is not None and mr is not None
        left_rows = [row for _k, row in ml]
        right_rows = [row for _k, row in mr]
        ml.close(); mr.close()
        assert len(left_rows) == len(right_rows) == 40 * 32
        assert all(len(r[1]) == 4096 for r in left_rows)


def test_row_bytes_accounts_for_type_shapes() -> None:
    assert row_bytes((None,)) > 0
    assert row_bytes((1,)) > row_bytes((None,))
    wide = "x" * WIDE
    assert row_bytes((1, wide)) > WIDE


# ---------------------------------------------------------------------------
# 多段归并 + 文件句柄有界回归
# ---------------------------------------------------------------------------

def test_kway_merger_opens_at_max_fan_in_files() -> None:
    """KWayMerger 同时打开的文件描述符数 == 输入流数，且退出即关闭。"""
    with tempfile.TemporaryDirectory() as d:
        runs = _write_runs(d, [(i, f"v{i}") for i in range(FAN_IN * 3)],
                           chunk=3)
        before = len(os.listdir("/proc/self/fd"))
        with KWayMerger(runs[:FAN_IN]) as merger:
            during = len(os.listdir("/proc/self/fd"))
            assert during - before == FAN_IN
            assert len(list(merger)) == FAN_IN * 3
        after = len(os.listdir("/proc/self/fd"))
        assert after <= before


def test_more_runs_than_handles_merge_correctly() -> None:
    """run 数远超扇入：分层归并结果与直接全序排序一致，计数不丢。"""
    n = FAN_IN * FAN_IN * 3 + 7  # 三层归并的数据量
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        rows = [(i, f"line-{i % 13}") for i in range(n)]
        # 每 3 行一个 run -> 每侧约 170 个段
        lr = _write_runs(d1, rows, chunk=3,
                         budget=MemoryBudget(16 * 1024 * 1024), chunk_bytes=4096)
        rr = _write_runs(d2, list(reversed(rows)), chunk=3,
                         budget=MemoryBudget(16 * 1024 * 1024), chunk_bytes=4096)
        assert len(lr) > FAN_IN * FAN_IN
        set_fan_in_override(FAN_IN)  # 固定 8 路扇入，强制三层归并
        try:
            stats = reconcile_sorted(
                lr, rr, left_total=n, right_total=n,
                max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
            )
            assert stats.diff_group_total == 0
            assert stats.matched_group_count == n
            merger = cascade_merge(lr, d1)
            assert merger is not None
            merged = [row for _key, row in merger]
            merger.close()
        finally:
            set_fan_in_override(None)
        assert merged == sorted(rows, key=sort_key)


def test_effective_fan_in_bounds_and_override() -> None:
    """自适应扇入始终落在 [MIN_FAN_IN, HARD_FAN_IN_CAP]，且受覆盖钩子控制。"""
    from app.external_merge import HARD_FAN_IN_CAP, MIN_FAN_IN

    auto = effective_fan_in()
    assert MIN_FAN_IN <= auto <= HARD_FAN_IN_CAP
    set_fan_in_override(4)
    try:
        assert effective_fan_in() == 4
    finally:
        set_fan_in_override(None)
    assert effective_fan_in() == auto


def test_single_level_merge_when_runs_within_fan_in() -> None:
    """run 数不超过自适应扇入时不应产生任何中间层文件。"""
    with tempfile.TemporaryDirectory() as d:
        runs = _write_runs(d, [(i, f"v{i}") for i in range(500)], chunk=50)
        before = set(os.listdir(d))
        merger = cascade_merge(runs, d)
        assert merger is not None
        assert len(list(merger)) == 500
        merger.close()
        after = set(os.listdir(d))
        assert before == after  # 无 merge-l* 中间文件


def test_merge_under_low_nofile_subprocess() -> None:
    """在 NOFILE=32 的子进程里跑两侧各 300 段的归并，不得 EMFILE。

    用子进程隔离 rlimit，避免影响同进程其他用例；沙箱不允许降低时则跳过。
    """
    script = textwrap.dedent(
        """
        import resource, sys, tempfile
        sys.path.insert(0, "/workspace")
        from app.external_merge import RunWriter, MemoryBudget, reconcile_sorted

        try:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (32, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        except (ValueError, OSError):
            print("SKIP")
            raise SystemExit(0)

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            w1 = RunWriter(d1, 100, 4 * 1024 * 1024, MemoryBudget(8 * 1024 * 1024))
            w2 = RunWriter(d2, 100, 4 * 1024 * 1024, MemoryBudget(8 * 1024 * 1024))
            for i in range(0, 30000, 100):
                rows = [(j, "v" * 64) for j in range(i, min(i + 100, 30000))]
                w1.add(rows)
                w2.add(list(reversed(rows)))
            r1, r2 = w1.finish(), w2.finish()
            assert len(r1) == 300 and len(r2) == 300
            stats = reconcile_sorted(
                r1, r2, left_total=30000, right_total=30000,
                max_diff_types=20, is_cancelled=lambda: False, work_dir=d1,
            )
            assert stats.diff_group_total == 0
            assert stats.matched_group_count == 30000
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in {"OK", "SKIP"}
