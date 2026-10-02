import unittest

from backend.tools.statistics import calculate_statistics, run_statistics


CSV_DATA = "category,revenue,orders\nA,10,2\nB,20,4\nC,40,8\n"


class StatisticsTests(unittest.TestCase):
    def test_mean_and_median(self) -> None:
        self.assertEqual(calculate_statistics(CSV_DATA, "mean", "revenue")["value"], 23.333333)
        self.assertEqual(calculate_statistics(CSV_DATA, "median", "revenue")["value"], 20.0)

    def test_percentile_uses_linear_interpolation(self) -> None:
        result = calculate_statistics(CSV_DATA, "percentile", "revenue", percentile=75)
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], 30.0)

    def test_correlation_uses_matching_columns(self) -> None:
        result = calculate_statistics(CSV_DATA, "correlation", "revenue", "orders")
        self.assertTrue(result["ok"])
        self.assertGreater(result["value"], 0.99)

    def test_linear_regression_returns_slope_and_intercept(self) -> None:
        result = calculate_statistics(CSV_DATA, "linear_regression", "orders", "revenue")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], {"slope": 5.0, "intercept": 0.0})
        self.assertEqual(result["count"], 3)

    def test_rejects_nonnumeric_data(self) -> None:
        result = calculate_statistics("name,value\nA,nope\n", "mean", "value")
        self.assertFalse(result["ok"])
        self.assertIn("nonnumeric", result["error"])

    def test_rejects_truncated_sql_results(self) -> None:
        result = calculate_statistics(
            "value\n1\n2\n[Results truncated]",
            "mean",
            "value",
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "Statistics require complete SQL results")
    def test_langchain_tool_returns_actual_result(self) -> None:
        output = run_statistics.invoke(
            {"csv_data": CSV_DATA, "operation": "max", "column": "revenue"}
        )
        self.assertIn('"value":40.0', output)


if __name__ == "__main__":
    unittest.main()


