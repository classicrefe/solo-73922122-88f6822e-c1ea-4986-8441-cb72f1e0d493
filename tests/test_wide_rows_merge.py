"""宽行与多段归并的端到端回归（FastAPI + 真实 PostgreSQL）。

覆盖需求中的资源控制场景：
- 16KiB 级长文本（产线说明）结果集，双路并行共守 128MiB，精确核对不截断；
- 窄宽混合列、NULL 与空串并存，行值/次数/NULL 语义保持；
- 小分块制造多于可用句柄的暂存段，自适应扇入 + 分层归并正常出结论；
- 失败/取消不发布结论，暂存纳入原 rmtree 清理流程。
"""
from __future__ import annotations

import os
import time

from fastapi.testclient import TestClient

from app import external_merge as em

WIDE = 16 * 1024  # 16KiB 长文本，对应“带长产线说明”的日结结果


def _wait_terminal(client: TestClient, job_id: str, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("succeeded", "failed", "cancelled", "timeout"):
            return body
        time.sleep(0.1)
    raise AssertionError("任务未在时限内到达终态")


def test_wide_text_rows_reconciled_exactly(client: TestClient) -> None:
    """两侧各 12000 行、每行 16KiB 文本：默认配置下精确核对、文本不截断。"""
    base = (
        "SELECT g::bigint AS id, repeat('x', 16384)::text AS note "
        "FROM generate_series(1, 12000) AS g"
    )
    # 左侧第 7 行换成等长不同文本，制造一个精确差异
    left = (
        "SELECT g::bigint AS id, "
        "CASE WHEN g = 7 THEN repeat('z', 16384)::text "
        "ELSE repeat('x', 16384)::text END AS note "
        "FROM generate_series(1, 12000) AS g"
    )
    out = _wait_terminal(
        client,
        client.post("/jobs", json={"left_sql": left, "right_sql": base}).json()["job_id"],
    )
    assert out["status"] == "succeeded", out
    result = out["result"]
    assert result["left_row_total"] == result["right_row_total"] == 12000
    assert result["column_types"] == ["bigint", "text"]
    # id=7 的两个等长不同文本各成一个单侧差异组
    assert result["diff_group_total"] == 2
    for diff in result["diffs"]:
        assert len(diff["values"][1]) == WIDE  # 无截断
    ids = {d["values"][0] for d in result["diffs"]}
    assert ids == {7}


def test_wide_identical_sets_report_identical(client: TestClient) -> None:
    sql = (
        "SELECT (g % 97)::bigint AS id, repeat('w', 16384)::text AS note "
        "FROM generate_series(1, 6000) AS g"
    )
    out = _wait_terminal(
        client,
        client.post("/jobs", json={"left_sql": sql, "right_sql": sql}).json()["job_id"],
    )
    assert out["status"] == "succeeded", out
    result = out["result"]
    assert result["identical"] is True
    assert result["diff_group_total"] == 0
    assert result["left_row_total"] == 6000


def test_narrow_wide_mixed_with_null_and_empty_string(client: TestClient) -> None:
    """窄宽混合 + NULL/空串并存：差异计数与值语义全部保留。"""
    left = (
        "SELECT v.id::bigint, v.note FROM (VALUES "
        "(1::bigint, repeat('Q', 16384)::text), "
        "(2, repeat('Q', 16384)::text), "
        "(3, NULL::text), "
        "(4, ''::text), "
        "(5, 'short')"
        ") AS v(id, note)"
    )
    right = (
        "SELECT v.id::bigint, v.note FROM (VALUES "
        "(1::bigint, repeat('Q', 16384)::text), "   # 完全一致
        "(2, repeat('P', 16384)::text), "            # 宽文本内容不同
        "(3, ''::text), "                            # NULL -> 空串
        "(4, NULL::text), "                          # 空串 -> NULL
        "(5, 'short'), (5, 'short')"                 # 右侧重复一次
        ") AS v(id, note)"
    )
    out = _wait_terminal(
        client,
        client.post("/jobs", json={"left_sql": left, "right_sql": right}).json()["job_id"],
    )
    assert out["status"] == "succeeded", out
    result = out["result"]
    assert result["matched_group_count"] == 1  # 仅 id=1 完全一致
    pairs = {(d["values"][0], d["values"][1]): (d["left_count"], d["right_count"])
             for d in result["diffs"]}
    # 宽文本差异保持完整长度
    assert all(len(k[1]) == WIDE for k in pairs if k[0] == 2)
    # NULL 与空串分别成组，不被合并
    assert (3, None) in pairs and (3, "") in pairs
    assert (4, None) in pairs and (4, "") in pairs
    # 重复次数精确
    assert pairs[(5, "short")] == (1, 2)


def test_many_runs_beyond_handles_merges_with_small_fan_in(
    client: TestClient, monkeypatch
) -> None:
    """小分块产生大量暂存段，并强制小扇入走分层归并，结论仍正确。"""
    from app import config as config_mod

    # 每 500 行一个 run：每侧 24 个段，扇入固定 8 -> 两层归并
    monkeypatch.setattr(config_mod.settings, "chunk_rows", 500)
    monkeypatch.setattr(em, "_fan_in_override", 8)
    try:
        sql = (
            "SELECT g::bigint AS id, "
            "('line-' || (g % 17))::text AS name "
            "FROM generate_series(1, 12000) AS g"
        )
        out = _wait_terminal(
            client,
            client.post("/jobs", json={"left_sql": sql, "right_sql": sql}).json()["job_id"],
        )
        assert out["status"] == "succeeded", out
        assert out["result"]["identical"] is True
        assert out["result"]["left_row_total"] == 12000
        assert out["result"]["diff_group_total"] == 0
    finally:
        monkeypatch.setattr(em, "_fan_in_override", None)


def test_cancelled_wide_job_publishes_no_conclusion_and_cleans_staging(
    client: TestClient, monkeypatch
) -> None:
    """宽行任务被取消：无结论、无暂存残留、名额释放（清理流程覆盖新暂存）。"""
    from app import db as db_mod
    from app import manager as manager_mod

    def flag_only_cancel(self):
        for job in manager_mod.manager._jobs.values():
            if job.status in ("running", "cancelling"):
                job.cancel_requested = True
                job.status = "cancelling"

    monkeypatch.setattr(db_mod.SnapshotSession, "cancel_all", flag_only_cancel)

    real_add = em.RunWriter.add

    def slow_add(self, rows):
        time.sleep(0.1)
        return real_add(self, rows)

    monkeypatch.setattr(em.RunWriter, "add", slow_add)

    sql = (
        "SELECT g::bigint AS id, repeat('c', 16384)::text AS note "
        "FROM generate_series(1, 12000) AS g"
    )
    resp = client.post("/jobs", json={"left_sql": sql, "right_sql": sql})
    job_id = resp.json()["job_id"]
    for _ in range(100):
        if client.get(f"/jobs/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.02)
    cancelled = client.post(f"/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200

    out = _wait_terminal(client, job_id, timeout=30)
    assert out["status"] == "cancelled"
    assert out["result"] is None  # 失败/取消不发布结论

    # 新增暂存（含分层中间段）必须纳入原清理流程
    from app import config as config_mod

    job_dir = os.path.join(config_mod.settings.work_dir, job_id)
    assert not os.path.exists(job_dir)
    # 名额已释放，查询与新建任务保持可用
    assert client.get("/jobs").status_code == 200
    again = client.post(
        "/jobs",
        json={
            "left_sql": "SELECT 1::bigint, 'x'::text",
            "right_sql": "SELECT 1::bigint, 'x'::text",
        },
    )
    assert again.status_code == 201
    assert _wait_terminal(client, again.json()["job_id"])["status"] == "succeeded"


def test_cancel_while_reader_blocked_on_budget(
    client: TestClient, monkeypatch
) -> None:
    """读取线程阻塞在共享内存预算上时取消：线程须及时退出、不泄漏连接。

    用极小预算让宽行读取频繁进入条件变量等待，验证取消标志能打断等待，
    终态在秒级到达且读取线程不跨任务残留。
    """
    from app import config as config_mod
    from app import db as db_mod
    from app import manager as manager_mod

    monkeypatch.setattr(config_mod.settings, "read_memory_budget", 2 * 1024 * 1024)

    def flag_only_cancel(self):
        for job in manager_mod.manager._jobs.values():
            if job.status in ("running", "cancelling"):
                job.cancel_requested = True
                job.status = "cancelling"

    monkeypatch.setattr(db_mod.SnapshotSession, "cancel_all", flag_only_cancel)

    sql = (
        "SELECT g::bigint AS id, repeat('c', 16384)::text AS note "
        "FROM generate_series(1, 12000) AS g"
    )
    resp = client.post("/jobs", json={"left_sql": sql, "right_sql": sql})
    job_id = resp.json()["job_id"]
    for _ in range(100):
        if client.get(f"/jobs/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.02)
    time.sleep(0.3)  # 让读取线程进入预算阻塞窗口
    client.post(f"/jobs/{job_id}/cancel")

    t0 = time.time()
    out = _wait_terminal(client, job_id, timeout=15)
    assert time.time() - t0 < 5  # 取消被及时受理，未等满超时
    assert out["status"] == "cancelled"
    assert out["result"] is None

    job = manager_mod.manager.get_job(job_id)
    for t in job.reader_threads:
        t.join(timeout=3)
        assert not t.is_alive()  # 阻塞在预算上的线程也已退出，无连接泄漏
    job_dir = os.path.join(config_mod.settings.work_dir, job_id)
    assert not os.path.exists(job_dir)
    assert manager_mod.manager._active is None
