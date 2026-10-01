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

    def test_creates_heatmap_from_long_rows(self) -> None:
        csv_data = "region,category,revenue\nSouth,A,10\nSouth,B,20\nNorth,A,30\n"
        figure = json.loads(
            create_chart(
                csv_data,
                "heatmap",
                "region",
                "category",
                value_column="revenue",
            )
        )

        self.assertEqual(figure["data"][0]["type"], "heatmap")
        self.assertEqual(figure["data"][0]["z"], [[10.0, 30.0], [20.0, None]])

    def test_creates_hierarchical_treemap(self) -> None:
        csv_data = "category,state,revenue\nA,South,10\nA,North,20\nB,South,5\n"
        figure = json.loads(
            create_chart(
                csv_data,
                "treemap",
                value_column="revenue",
                path_columns=["category", "state"],
            )
        )

        trace = figure["data"][0]
        self.assertEqual(trace["type"], "treemap")
        self.assertEqual(trace["branchvalues"], "total")
        self.assertIn("A / South", trace["ids"])
        self.assertEqual(trace["values"][trace["ids"].index("A")], 30.0)

    def test_heatmap_preserves_numeric_looking_labels(self) -> None:
        csv_data = "year,score,revenue\n2017,5,10\n2018,5,20\n"
        trace = json.loads(
            create_chart(
                csv_data,
                "heatmap",
                "year",
                "score",
                value_column="revenue",
            )
        )["data"][0]

        self.assertEqual(trace["x"], ["2017", "2018"])
        self.assertEqual(trace["z"], [[10.0, 20.0]])

    def test_rejects_missing_or_nonnumeric_measure(self) -> None:
        missing = run_chart.invoke(
            {
                "csv_data": CSV_DATA,
                "chart_type": "heatmap",
                "x_column": "state",
                "y_column": "revenue",
                "value_column": "missing",
            }
        )
        nonnumeric = run_chart.invoke(
            {
                "csv_data": "state,revenue\nSouth,nope\n",
                "chart_type": "bar",
                "x_column": "state",
                "y_column": "revenue",
            }
        )

        self.assertEqual(missing, "Rejected: Column not found: missing")
        self.assertEqual(nonnumeric, "Rejected: Column must be numeric: revenue")

    def test_rejects_empty_and_oversized_charts(self) -> None:
        empty = run_chart.invoke(
            {
                "csv_data": "state,revenue\n",
                "chart_type": "bar",
                "x_column": "state",
                "y_column": "revenue",
            }
        )
        rows = "state,revenue\n" + "\n".join(f"S{number},{number}" for number in range(21))
        oversized = run_chart.invoke(
            {
                "csv_data": rows,
                "chart_type": "pie",
                "x_column": "state",
                "y_column": "revenue",
            }
        )

        self.assertEqual(empty, "Rejected: Chart requires at least one result row")
        self.assertEqual(oversized, "Rejected: Pie chart exceeds 20 slices")

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
