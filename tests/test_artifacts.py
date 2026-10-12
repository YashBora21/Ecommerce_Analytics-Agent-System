import unittest
from unittest.mock import MagicMock, patch

from backend.tools.artifacts import cleanup_artifacts, setup_artifacts


class ArtifactTests(unittest.TestCase):
    def test_schema_keeps_only_used_metadata(self) -> None:
        connection = MagicMock()

        setup_artifacts(connection)

        create_table = connection.execute.call_args_list[1].args[0]
        migration = connection.execute.call_args_list[2].args[0]
        self.assertNotIn("source_sql text", create_table)
        self.assertNotIn("profile_json jsonb", create_table)
        self.assertIn("DROP COLUMN IF EXISTS source_sql", migration)
        self.assertIn("DROP COLUMN IF EXISTS profile_json", migration)

    def test_cleanup_reads_named_postgres_columns(self) -> None:
        connection = MagicMock()
        expired_query = MagicMock()
        expired_query.fetchall.return_value = [
            {"artifact_id": "artifact-1", "relation_name": "result_old"}
        ]
        connection.execute.side_effect = [expired_query, MagicMock(), MagicMock()]

        with patch("backend.tools.artifacts.setup_artifacts"):
            removed = cleanup_artifacts(connection)

        self.assertEqual(removed, 1)
        self.assertEqual(connection.execute.call_args_list[-1].args[1], (["artifact-1"],))


if __name__ == "__main__":
    unittest.main()
