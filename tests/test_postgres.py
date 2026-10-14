import unittest
from unittest.mock import MagicMock, patch

from backend.db import postgres


class PostgresPoolTests(unittest.TestCase):
    def tearDown(self) -> None:
        postgres._pools.clear()

    def test_checks_connections_before_reuse(self) -> None:
        pool = MagicMock()
        with patch("backend.db.postgres.ConnectionPool") as pool_class:
            pool_class.return_value = pool
            postgres.get_pool("postgresql://test")

        self.assertIs(
            pool_class.call_args.kwargs["check"],
            pool_class.check_connection,
        )
        pool.open.assert_called_once_with(wait=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
