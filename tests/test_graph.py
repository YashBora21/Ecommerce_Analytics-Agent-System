import unittest

from backend.agent.graph import begin_turn


class GraphTests(unittest.TestCase):
    def test_begin_turn_resets_only_per_turn_state(self) -> None:
        reset = begin_turn({
            "question": "Follow-up",
            "history": [{"role": "user", "content": "Earlier"}],
            "evidence": [{"sql": "old", "result": "old"}],
            "chart": "old chart",
            "error": "old error",
            "steps": 4,
        })
        self.assertEqual(reset["evidence"], [])
        self.assertEqual(reset["chart"], "")
        self.assertEqual(reset["error"], "")
        self.assertEqual(reset["steps"], 0)
        self.assertNotIn("history", reset)


if __name__ == "__main__":
    unittest.main()
