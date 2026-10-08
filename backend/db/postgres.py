import os
import threading
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


_pools: dict[str, ConnectionPool] = {}
_lock = threading.Lock()


def get_pool(database_url: str) -> ConnectionPool:
    with _lock:
        pool = _pools.get(database_url)
        if pool is None:
            pool = ConnectionPool(
                database_url,
                min_size=1,
                max_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
                kwargs={"row_factory": dict_row},
                open=False,
            )
            pool.open(wait=True, timeout=30)
            _pools[database_url] = pool
        return pool


@contextmanager
def postgres_connection(database_url: str):
    with get_pool(database_url).connection() as connection:
        yield connection


def close_pools() -> None:
    with _lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.close()
