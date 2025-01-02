import pandas as pd
import logging
class SalesAnalyzer:
    def __init__(self):
        """
        Initialize with the dataset.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized Seasonality Sales Analysis class.")
    def tag_holiday_periods(self, df):
        self.logger.info("Starting to Holiday Sales analysis.")
        df["Date"] = pd.to_datetime(df["Date"])

        # Define State holidays
        df["StateHoliday"] = df["StateHoliday"].replace({"0": None})
        holiday_dates = df.loc[df["StateHoliday"].notnull(), "Date"]

        # Create columns for "Before Holiday", "During Holiday", "After Holiday"
        df["HolidayPeriod"] = "Normal"
        for holiday in holiday_dates:
            df.loc[df["Date"] == holiday, "HolidayPeriod"] = "During Holiday"
            df.loc[df["Date"] == holiday - pd.Timedelta(days=1), "HolidayPeriod"] = "Before Holiday"
            df.loc[df["Date"] == holiday + pd.Timedelta(days=1), "HolidayPeriod"] = "After Holiday"    
        # School Holidays
        df["SchoolHoliday"] = df["SchoolHoliday"].replace({"0": None})
        holiday_dates1 = df.loc[df["SchoolHoliday"].notnull(), "Date"]
        
        
        # Create columns for "Before Holiday", "During Holiday", "After Holiday"
        df["SchoolHolidayPeriod"] = "Normal"
        for holiday in holiday_dates1:
            df.loc[df["Date"] == holiday, "SchoolHolidayPeriod"] = "During Holiday"
            df.loc[df["Date"] == holiday - pd.Timedelta(days=1), "SchoolHolidayPeriod"] = "Before Holiday"
            df.loc[df["Date"] == holiday + pd.Timedelta(days=1), "SchoolHolidayPeriod"] = "After Holiday"

        # Analyze sales behavior
        behavior_summary = df.groupby("HolidayPeriod")["Sales"].agg(["mean", "sum", "median"])
        school_behavior_summary = df.groupby("SchoolHolidayPeriod")["Sales"].agg(["mean", "sum", "median"])
        self.logger.info("Sale to Holiday Seasonality analysis.")
        return behavior_summary, school_behavior_summary