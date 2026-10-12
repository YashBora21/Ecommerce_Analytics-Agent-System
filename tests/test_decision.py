import unittest

from backend.agent.decision import decision_json_schema, parse_action, parse_action_json


class DecisionTests(unittest.TestCase):
    def test_parses_answer(self) -> None:
        action = parse_action({
            "action": "answer",
            "reason": "Enough evidence",
            "arguments": {"final_answer": "  Five orders.  "},
        })
        self.assertEqual(action, {"name": "answer", "args": {"final_answer": "Five orders."}})

    def test_validates_statistics_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "correlation requires second_column"):
            parse_action({
                "action": "calculate_statistics",
                "reason": "Correlation requested",
                "arguments": {"operation": "correlation", "column": "revenue"},
            })

        with self.assertRaisesRegex(ValueError, "linear_regression requires second_column"):
            parse_action({
                "action": "calculate_statistics",
                "reason": "Trend requested",
                "arguments": {"operation": "linear_regression", "column": "orders"},
            })

        action = parse_action({
            "action": "calculate_statistics",
            "reason": "Percentile requested",
            "arguments": {
                "operation": "percentile",
                "column": "revenue",
                "second_column": None,
                "percentile": 90,
                "artifact_id": "artifact-1",
            },
        })
        self.assertEqual(action["args"]["second_column"], "")

    def test_validates_chart_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "heatmap requires"):
            parse_action({
                "action": "create_chart",
                "reason": "Chart requested",
                "arguments": {
                    "chart_type": "heatmap",
                    "x_column": "region",
                    "y_column": "category",
                    "final_answer": "Chart prepared.",
                },
            })

    def test_json_parser_rejects_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_action_json("{not valid json}")

    def test_rejects_oversized_final_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 5000 characters"):
            parse_action({
                "action": "answer",
                "reason": "Too much output",
                "arguments": {"final_answer": "x" * 5_001},
            })

    def test_generation_schema_is_strict(self) -> None:
        objects = []

        def collect(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    objects.append(value)
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        schema = decision_json_schema()
        collect(schema)

        self.assertEqual(schema["type"], "object")
        self.assertIn("anyOf", schema["properties"]["decision"])
        self.assertTrue(objects)
        self.assertTrue(all(item["additionalProperties"] is False for item in objects))
        self.assertTrue(all(
            set(item.get("properties", {})) == set(item["required"])
            for item in objects
        ))

    def test_query_action_requires_sql(self) -> None:
        action = parse_action({
            "action": "query_sql",
            "reason": "Need data",
            "arguments": {"sql": " SELECT COUNT(*) FROM orders "},
        })
        self.assertEqual(action["args"]["sql"], "SELECT COUNT(*) FROM orders")

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid next action"):
            parse_action({"action": "delete_data", "reason": "bad", "arguments": {}})



if __name__ == "__main__":
    unittest.main()

