"""运行期配置（全部可通过环境变量覆盖）。"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # 数据库连接
    pg_dsn: str = _env(
        "PG_DSN",
        "postgresql://recon:recon@postgres:5432/recon",
    )
    connect_timeout: int = _env_int("PG_CONNECT_TIMEOUT", 5)
    statement_timeout_ms: int = _env_int("PG_STATEMENT_TIMEOUT_MS", 28000)
    lock_timeout_ms: int = _env_int("PG_LOCK_TIMEOUT_MS", 2000)
    idle_in_tx_timeout_ms: int = _env_int("PG_IDLE_IN_TX_TIMEOUT_MS", 60000)

    # 每轮核对的硬性时限（秒）
    job_timeout_seconds: int = _env_int("JOB_TIMEOUT_SECONDS", 30)

    # 流式归并的内存/磁盘参数
    chunk_rows: int = _env_int("SORT_CHUNK_ROWS", 20000)
    # 单个排序块的估算字节上限：宽行（如 16KiB text）按字节提前切块，
    # 避免只按行数切块时单块撑爆 128MiB 容器上限
    chunk_bytes: int = _env_int("SORT_CHUNK_BYTES", 16 * 1024 * 1024)
    # 双路读取共享的驻留内存预算（两侧缓冲估算字节之和的硬上限）
    read_memory_budget: int = _env_int(
        "READ_MEMORY_BUDGET_BYTES", 40 * 1024 * 1024
    )
    # 服务端游标单次网络往返的最大行数；配合内存预算限制宽行批次的瞬时驻留。
    # 取 200 时 16KiB 宽行单批约 3.2MiB，两侧预算外的网络缓冲也在 128MiB 余量内；
    # 窄行场景下批次变小带来的往返增加对 50 万行总耗时影响约 1 秒。
    fetch_batch_rows: int = _env_int("FETCH_BATCH_ROWS", 200)
    max_diff_types: int = _env_int("MAX_DIFF_TYPES", 20)

    # 暂存目录（容器内），独立挂载 tmpfs / 卷
    work_dir: str = _env("WORK_DIR", "/tmp/recon-work")

    # 允许出现在结果集中的列类型（PostgreSQL 类型名 -> psycopg 加载后的 Python 类型）
    allowed_pg_types: tuple[str, ...] = ("bigint", "text")


settings = Settings()
