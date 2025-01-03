import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        return  df
    def holiday_analysis(self, df):
        behavior_summary = df.groupby("HolidayPeriod")["Sales"].agg(["mean", "sum", "median"])
        return  behavior_summary
    
    
    def analyze_seasonal_behavior(self,df):
        """
        Analyze seasonal purchase behaviors based on inferred holidays.
        
        Parameters:
        - df (pd.DataFrame): Input dataset with at least 'Date', 'Sales', and 'HolidayPeriod'.
        
        Returns:
        - holiday_stats (pd.DataFrame): Aggregated sales and customer metrics by season.
        - seasonal_plot: Displays seasonal trends in sales.
        """
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Infer holiday periods based on dates
        def assign_holiday(row):
            if row['Date'].month == 12:  # December for Christmas
                return 'Christmas'
            elif row['Date'].month in [3, 4]:  # March or April for Easter
                return 'Easter'
            elif row['Date'].month in [6, 7, 8]:  # June-August for Summer
                return 'Summer'
            elif row['Date'].month in [1, 2]:  # January-February for New Year
                return 'New Year'
            else:
                return 'Normal'
        
        df['InferredHoliday'] = df.apply(assign_holiday, axis=1)
        
        # Group by InferredHoliday and aggregate sales and customers
        holiday_stats = df.groupby('InferredHoliday').agg({
            'Sales': ['sum', 'mean'],
            'Customers': ['sum', 'mean']
        }).reset_index()
        
        # Rename columns for clarity
        holiday_stats.columns = ['InferredHoliday', 'TotalSales', 'AvgSales', 'TotalCustomers', 'AvgCustomers']
        
        # Print holiday stats
        return holiday_stats, df
    

    # Preprocessing for PCA
    def preprocess_pca(self, df):
        # Select only numeric columns
        numeric_features = df.select_dtypes(include=['number'])
        
        # Drop columns with constant values (like `Open` if all are 1 or 0)
        numeric_features = numeric_features.loc[:, (numeric_features != numeric_features.iloc[0]).any()]
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        return scaled_features, numeric_features.columns

    # Apply PCA
    def apply_pca(self, data, n_components=None):
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        
        return pca, pca_result
   