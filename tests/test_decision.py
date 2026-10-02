import unittest

from backend.agent.decision import chart_required, parse_action


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

    def test_chart_requirement_uses_evidence_not_question_templates(self) -> None:
        state = {
            "question": "Summarize this result",
            "evidence": [{
                "kind": "sql",
                "result": "region,revenue\nA,10\nB,20\nC,30",
            }],
        }
        self.assertTrue(chart_required(state))
        state["question"] = "Summarize this result, text only"
        self.assertFalse(chart_required(state))

        state = {
            "question": "Are these measures related?",
            "evidence": [{
                "kind": "statistics",
                "result": '{"operation":"correlation","value":0.4,"count":100}',
            }],
        }
        self.assertTrue(chart_required(state))


if __name__ == "__main__":
    unittest.main()
