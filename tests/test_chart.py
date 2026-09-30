import json
import unittest

from backend.tools.chart import create_chart, run_chart


CSV_DATA = "state,revenue\nSao Paulo,100\nRio de Janeiro,60\n"


class ChartTests(unittest.TestCase):
    def test_creates_plotly_bar_chart(self) -> None:
        figure = json.loads(create_chart(CSV_DATA, "bar", "state", "revenue", "Revenue"))

        self.assertEqual(figure["data"][0]["type"], "bar")
        self.assertEqual(figure["data"][0]["x"], ["Sao Paulo", "Rio de Janeiro"])
        self.assertEqual(figure["data"][0]["y"], [100.0, 60.0])
        self.assertEqual(figure["layout"]["title"]["text"], "Revenue")

    def test_rejects_truncated_results(self) -> None:
        output = run_chart.invoke(
            {
                "csv_data": CSV_DATA + "[Results truncated]",
                "chart_type": "bar",
                "x_column": "state",
                "y_column": "revenue",
            }
        )

        self.assertEqual(output, "Rejected: Charts require complete SQL results")


if __name__ == "__main__":
    unittest.main()
