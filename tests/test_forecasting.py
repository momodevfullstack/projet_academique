import unittest

import pandas as pd

from src.forecasting import add_features, temporal_split


class ForecastingTests(unittest.TestCase):
    def test_add_features_extracts_calendar_fields(self):
        data = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2015-07-31"]),
                "DayOfWeek": [5],
            }
        )

        result = add_features(data)

        self.assertEqual(result.loc[0, "Day"], 31)
        self.assertEqual(result.loc[0, "Month"], 7)
        self.assertEqual(result.loc[0, "Year"], 2015)
        self.assertEqual(result.loc[0, "IsWeekend"], 0)

    def test_temporal_split_keeps_latest_days_for_test(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5),
                "Sales": [10, 20, 30, 40, 50],
            }
        )

        train, test = temporal_split(data, test_days=2)

        self.assertLess(train["Date"].max(), test["Date"].min())
        self.assertEqual(
            test["Date"].tolist(), list(pd.date_range("2024-01-04", periods=2))
        )

    def test_temporal_split_rejects_empty_data(self):
        with self.assertRaisesRegex(ValueError, "vide"):
            temporal_split(pd.DataFrame(columns=["Date"]))


if __name__ == "__main__":
    unittest.main()
