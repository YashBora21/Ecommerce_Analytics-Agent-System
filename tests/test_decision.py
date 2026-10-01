import unittest

from backend.agent.decision import parse_action


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

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid next action"):
            parse_action({"action": "delete_data", "reason": "bad", "arguments": {}})


if __name__ == "__main__":
    unittest.main()
