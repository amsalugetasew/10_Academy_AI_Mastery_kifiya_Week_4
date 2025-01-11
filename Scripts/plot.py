import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class Plot:
    def __init__(self):
        """Initializes the Ploting."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized Plot class.")
        self.df = {}
    def check_promotion_distribution_with_plot(self, df_train, df_test):
        """
        Compares the distribution of promotions in training and test sets and plots the results.
        
        :param df_train: DataFrame containing training data.
        :param df_test: DataFrame containing test data.
        :return: None
        """
        self.logger.info("Starting to ploting promotion distribution on training and test dataset.")
        # Calculate percentage of rows with active promotions in training data
        promo_train = df_train['Promo'].mean() * 100
        promo2_train = df_train['Promo2'].mean() * 100
        
        # Calculate percentage of rows with active promotions in test data
        promo_test = df_test['Promo'].mean() * 100
        promo2_test = df_test['Promo2'].mean() * 100
        
        # Plotting the distribution of Promo and Promo2 in both train and test sets
        categories = ['Promo', 'Promo2']
        train_values = [promo_train, promo2_train]
        test_values = [promo_test, promo2_test]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.35
        index = range(len(categories))

        # Plot the bars for both training and test sets
        ax.bar(index, train_values, bar_width, label='Train', color='b')
        ax.bar([i + bar_width for i in index], test_values, bar_width, label='Test', color='r')

        # Labeling the plot
        ax.set_xlabel('Promotion Type')
        ax.set_ylabel('Percentage of Active Promotions')
        ax.set_title('Comparison of Promotion Distribution in Train and Test Sets')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(categories)
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

        if promo_train:
            self.logger.info("Successfully Ploting promotion distribution of training and testing of store data.")
        else:
            self.logger.error("Failed to  Ploting promotion distribution of training and testing of store data.", exc_info=True)
            
            
    def plot_sales_by_period(self, df, column, title, group_by):
        """
        Plot a bar chart for average sales by a specified period.
        The chart will include bar values and no y-axis ticks.
        
        :param df: Pandas DataFrame containing the sales data.
        :param column: Column name for the metric to plot (e.g., "Sales").
        :param title: Title of the chart.
        :param group_by: Column name for grouping the data (e.g., "HolidayPeriod").
        """
        # Calculate the average sales
        avg_sales = df.groupby(group_by)[column].mean()

        # Plot the bar chart
        ax = avg_sales.plot(kind='bar', title=title, color='skyblue', figsize=(10, 6))
        plt.xticks(rotation=45)
        plt.xlabel(group_by)
        plt.ylabel(f"Average {column}")
        
        # Remove y-axis ticks
        ax.yaxis.set_ticks([])
        
        # Add bar values on top
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        plt.show()
        
    def plot_for_holiday_effect(self, df):
        # Visualize sales trends across holidays
        plt.figure(figsize=(12, 6))
        df.groupby('InferredHoliday')['Sales'].mean().sort_values().plot(
            kind='bar', color='skyblue', edgecolor='black'
        )
        plt.title('Average Sales by Inferred Holiday Period', fontsize=14)
        plt.xlabel('Holiday Period', fontsize=12)
        plt.ylabel('Average Sales', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    def plot_seasonal_trends_line(self, df):
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by Date to compute daily total sales
        daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
        
        # Plot the time series
        plt.figure(figsize=(14, 6))
        plt.plot(daily_sales['Date'], daily_sales['Sales'], label='Sales', color='blue', linewidth=1.5)
        plt.title('Seasonal Trends Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Sales', fontsize=12)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    def plot_seasonal_trends_box(self,df):
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Assign inferred holidays
        df['InferredHoliday'] = df['Date'].apply(lambda x: 'Christmas' if x.month == 12 else 
                                                ('Easter' if x.month in [3, 4] else 
                                                ('New Year' if x.month in [1, 2] else 
                                                ('Summer' if x.month in [6, 7, 8] else 'Normal'))))
        
        # Create a box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='InferredHoliday', y='Sales', data=df, palette='Set2')
        plt.title('Sales Distribution by Inferred Holiday Period', fontsize=14)
        plt.xlabel('Holiday Period', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.5)
        plt.show()

    def plot_seasonal_trends_heatmap(self, df):
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract month and day of the week
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
        
        # Create a pivot table for heatmap
        pivot_table = df.pivot_table(
            values='Sales', 
            index='Month', 
            columns='DayOfWeek', 
            aggfunc='mean'
        )
        
        # Plot the heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlGnBu', cbar=True)
        plt.title('Average Sales by Month and Day of the Week', fontsize=14)
        plt.xlabel('Day of the Week', fontsize=12)
        plt.ylabel('Month', fontsize=12)
        plt.xticks(ticks=np.arange(7)+0.5, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
        plt.show()

    
    def plot_seasonal_trends_subplots(self, df):
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Assign inferred holidays
        df['InferredHoliday'] = df['Date'].apply(lambda x: 'Christmas' if x.month == 12 else 
                                                ('Easter' if x.month in [3, 4] else 
                                                ('New Year' if x.month in [1, 2] else 
                                                ('Summer' if x.month in [6, 7, 8] else 'Normal'))))
        
        # Aggregate sales by holiday
        holiday_sales = df.groupby(['InferredHoliday', 'Date'])['Sales'].sum().reset_index()
        
        # Get unique holidays
        holidays = holiday_sales['InferredHoliday'].unique()
        
        # Create subplots
        fig, axes = plt.subplots(nrows=len(holidays), figsize=(12, 6 * len(holidays)))
        
        for i, holiday in enumerate(holidays):
            holiday_data = holiday_sales[holiday_sales['InferredHoliday'] == holiday]
            axes[i].bar(holiday_data['Date'], holiday_data['Sales'], color='skyblue')
            axes[i].set_title(f'Sales Trends During {holiday}', fontsize=14)
            axes[i].set_xlabel('Date', fontsize=12)
            axes[i].set_ylabel('Sales', fontsize=12)
            axes[i].grid(alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
    def plot_sales_vs_customers(self, df):
        self.logger.info("Starting to promotion Analysis on sales")
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Customers'], df['Sales'], alpha=0.5, color='skyblue')
        plt.title('Sales vs. Number of Customers', fontsize=14)
        plt.xlabel('Number of Customers', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.grid(alpha=0.5)
        plt.show()
    def plot_sales_comparison(self, df):
        self.logger.info("Starting plotting promotion Analysis on sales")
        promo_sales = df[df['Promo'] == 1]['Sales'].mean()
        non_promo_sales = df[df['Promo'] == 0]['Sales'].mean()
        
        plt.figure(figsize=(8, 6))
        plt.bar(['With Promo', 'Without Promo'], [promo_sales, non_promo_sales], color=['teal', 'skyblue'])
        plt.title('Average Sales with and without Promotions', fontsize=14)
        plt.ylabel('Average Sales', fontsize=12)
        plt.show()
        self.logger.info("Successfully plotting promotion Analysis on sales")
    def plot_customer_comparison(self,df):
        self.logger.info("Starting plotting promotion Analysis on Customers")
        promo_customers = df[df['Promo'] == 1]['Customers'].mean()
        non_promo_customers = df[df['Promo'] == 0]['Customers'].mean()
        
        plt.figure(figsize=(8, 6))
        plt.bar(['With Promo', 'Without Promo'], [promo_customers, non_promo_customers], color=['orange', 'purple'])
        plt.title('Average Customers with and without Promotions', fontsize=14)
        plt.ylabel('Average Customers', fontsize=12)
        plt.show()
        self.logger.info("Successfully plotting promotion Analysis on Customers")
    
    def plot_sales_per_customer_comparison(self, df):
        self.logger.info("Starting plotting promotion Analysis on Sales vs Customers")
        df['Sales_per_Customer'] = df['Sales'] / df['Customers']
        promo_spc = df[df['Promo'] == 1]['Sales_per_Customer'].mean()
        non_promo_spc = df[df['Promo'] == 0]['Sales_per_Customer'].mean()
        
        plt.figure(figsize=(8, 6))
        plt.bar(['With Promo', 'Without Promo'], [promo_spc, non_promo_spc], color=['cyan', 'red'])
        plt.title('Sales per Customer with and without Promotions', fontsize=14)
        plt.ylabel('Sales per Customer', fontsize=12)
        plt.show()
        self.logger.info("Successfully plotting promotion Analysis on Sales vs Customers")
    
    def plot_time_series(self, df):
        self.logger.info("Starting plotting promotion Analysis Sales Over Time")
        df['Date'] = pd.to_datetime(df['Date'])
        promo_sales = df[df['Promo'] == 1].groupby('Date')['Sales'].mean()
        non_promo_sales = df[df['Promo'] == 0].groupby('Date')['Sales'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(promo_sales, label='With Promo', color='green', alpha=0.7)
        plt.plot(non_promo_sales, label='Without Promo', color='blue', alpha=0.7)
        plt.title('Sales Over Time: With and Without Promotions', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Sales', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
        self.logger.info("Successfully plotting promotion Analysis Sales Over Time")

    # Plot PCA explained variance
    def plot_explained_variance(self, pca):
        self.logger.info("Starting PCA plotting")
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.title('Explained Variance by PCA Components', fontsize=14)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontsize=12)
        plt.grid(alpha=0.5)
        plt.show()
        self.logger.info("Successfully PCA plotting")
    
    # PCA Biplot
    def plot_pca_biplot(self, pca_result, pca, feature_names):
        self.logger.info("Starting PCA Variance plotting")
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, c='blue', edgecolor='k')
        plt.title('PCA Biplot', fontsize=14)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        
        # Add feature vectors
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, pca.components_[0, i] * max(pca_result[:, 0]), 
                    pca.components_[1, i] * max(pca_result[:, 1]), 
                    color='red', alpha=0.8, head_width=0.05)
            plt.text(pca.components_[0, i] * max(pca_result[:, 0]) * 1.1, 
                    pca.components_[1, i] * max(pca_result[:, 1]) * 1.1, 
                    feature, color='black', ha='center', va='center', fontsize=10)
        
        plt.grid(alpha=0.5)
        plt.show()
        self.logger.info("Successfully PCA VariAnce plotting")
        
    def promo_effect_analysis(self, df):
        self.logger.info("Starting to Checking promotion effect Analysis")
        # Filter promo and non-promo data
        promo_data = df[df['Promo'] == 1]
        non_promo_data = df[df['Promo'] == 0]

        # 1. Bar Chart: Average Sales, Customers, and Sales per Customer
        promo_means = promo_data[['Sales', 'Customers', 'Sales_per_Customer']].mean()
        non_promo_means = non_promo_data[['Sales', 'Customers', 'Sales_per_Customer']].mean()

        promo_comparison = pd.DataFrame({'Promo': promo_means, 'Non-Promo': non_promo_means})
        promo_comparison.plot(kind='bar', figsize=(10, 6), rot=0)
        plt.title('Average Sales, Customers, and Sales per Customer (Promo vs. Non-Promo)', fontsize=14)
        plt.ylabel('Average Value', fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()
        self.logger.info("Successfully completing  promotion effect Analysis")
        return  promo_data, non_promo_data
    # Box Plot: Promo Effectiveness Across Stores
    def Promo_Effectiveness_Across_Stores(self, promo_data):
        self.logger.info("Starting to promotion effectiness Analysis across store")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=promo_data, x='Store', y='Sales', showfliers=False)
        plt.title('Sales Distribution Across Stores (During Promos)', fontsize=14)
        plt.ylabel('Sales', fontsize=12)
        plt.xlabel('Store', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(alpha=0.3)
        plt.show()
        self.logger.info("Successfully completing  promotion effectiness Analysis across store")
    #Heatmap: Promo Performance Across Stores
    def Promo_Performance_Across_Stores(self, promo_data):
        self.logger.info("Starting to promotion Performance Analysis across store")
        store_promo_sales = promo_data.groupby('Store')['Sales'].mean()
        store_promo_sales = store_promo_sales.reset_index().pivot(index='Store', columns=None, values='Sales')

        plt.figure(figsize=(12, 8))
        sns.heatmap(store_promo_sales, annot=True, fmt='.1f', cmap='coolwarm', cbar_kws={'label': 'Average Sales'})
        plt.title('Promo Effectiveness Across Stores', fontsize=14)
        plt.ylabel('Store', fontsize=12)
        plt.xlabel('Promo Effectiveness', fontsize=12)
        plt.show()
        self.logger.info("Successfully completing  promotion Performance Analysis across store")
    # Seasonal Promo Analysis (Line Chart)
    def Seasonal_Promo_Analysis_Line_Chart(self, promo_data, non_promo_data):
        self.logger.info("Starting plotting Seasonal promotion Analysis")
        promo_seasonal = promo_data.groupby('Month')['Sales'].mean()
        non_promo_seasonal = non_promo_data.groupby('Month')['Sales'].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(promo_seasonal.index, promo_seasonal, marker='o', label='Promo')
        plt.plot(non_promo_seasonal.index, non_promo_seasonal, marker='o', label='Non-Promo')
        plt.title('Seasonal Promo Effectiveness', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Sales', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        self.logger.info("Successfully completing plotting Seasonal promotion Analysis")
    
     # Separate data based on store open or closed
    def separate_by_store_open_close(self, df):
        self.logger.info("Starting Store open and close Analysis")
        open_days = df[df['Open'] == 1]
        closed_days = df[df['Open'] == 0]

        # 1. Average Metrics Comparison
        avg_open = open_days[['Sales', 'Customers', 'Sales_per_Customer']].mean()
        avg_closed = closed_days[['Sales', 'Customers', 'Sales_per_Customer']].mean()

        comparison_df = pd.DataFrame({'Open': avg_open, 'Closed': avg_closed})
        comparison_df.plot(kind='bar', figsize=(10, 6), rot=0)
        plt.title('Average Metrics: Open vs. Closed Days', fontsize=14)
        plt.ylabel('Average Value', fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()
        self.logger.info("Successfully completing Store open and close Analysis")
        return open_days, closed_days
    # Temporal Trends
    def Temporal_Trends(self, open_days):
        self.logger.info("Starting Store open days trend Analysis")
        # Group by Month to analyze trends
        monthly_sales = open_days.groupby('Month')['Sales'].mean()
        monthly_customers = open_days.groupby('Month')['Customers'].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(monthly_sales.index, monthly_sales, marker='o', label='Sales')
        plt.plot(monthly_customers.index, monthly_customers, marker='o', label='Customers')
        plt.title('Monthly Trends: Sales and Customers (Open Days)', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Value', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        self.logger.info("Successfully completing Store open days trend Analysis")
    
    # Heatmap of Store Activity
    def Heatmap_of_Store_Activity(self, open_days):
        store_open_sales = open_days.groupby('Store')['Sales'].mean().reset_index()
        store_open_sales = store_open_sales.pivot(index='Store', columns=None, values='Sales')

        plt.figure(figsize=(12, 8))
        sns.heatmap(store_open_sales, annot=False, cmap='coolwarm', cbar_kws={'label': 'Average Sales'})
        plt.title('Customer Behavior Across Stores (Open Days)', fontsize=14)
        plt.ylabel('Store', fontsize=12)
        plt.xlabel('Average Sales', fontsize=12)
        plt.show()
    # Scatter Plot: Holiday Periods and Behavior on Open Days
    def Scatter_Plot_Holiday_Behavior_open_days(self, open_days):
        self.logger.info("Starting Scatter Plot: Holiday Periods and Behavior on Open Days Analysis")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=open_days, x='HolidayPeriod', y='Sales', hue='StateHoliday')
        plt.title('Impact of Holiday Periods on Sales (Open Days)', fontsize=14)
        plt.xlabel('Holiday Period', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.legend(title='StateHoliday')
        plt.grid(alpha=0.3)
        plt.show()
        self.logger.info("Successfully completing Scatter Plot: Holiday Periods and Behavior on Open Days Analysis")
    
     # Identify stores open on all weekdays
    def Identify_stores_open_all_weekdays(self, df):
        self.logger.info("Strating  Week day open Stors Analysis")
        # Filter for weekdays (DayOfWeek 1 to 5) and Open = 1
        weekdays = df[(df['DayOfWeek'] >= 1) & (df['DayOfWeek'] <= 5) & (df['Open'] == 1)]

        # Group by Store and count unique weekdays
        weekday_open_counts = weekdays.groupby('Store')['DayOfWeek'].nunique()

        # Stores open all weekdays (1 through 5)
        stores_open_all_weekdays = weekday_open_counts[weekday_open_counts == 5].index.tolist()

        # Analyze weekend sales
        # Filter for weekend data (DayOfWeek 6 and 7)
        weekend = df[(df['DayOfWeek'] >= 6) & (df['DayOfWeek'] <= 7) & (df['Open'] == 1)]

        # Separate weekend data for stores open all weekdays and other stores
        weekend_sales_all_weekdays = weekend[weekend['Store'].isin(stores_open_all_weekdays)]
        weekend_sales_other_stores = weekend[~weekend['Store'].isin(stores_open_all_weekdays)]

        # Calculate average weekend sales
        avg_weekend_sales_all_weekdays = weekend_sales_all_weekdays['Sales'].mean()
        avg_weekend_sales_other_stores = weekend_sales_other_stores['Sales'].mean()

        # Print results
        print("Stores open on all weekdays:", stores_open_all_weekdays)
        print("Average weekend sales (stores open on all weekdays):", avg_weekend_sales_all_weekdays)
        print("Average weekend sales (other stores):", avg_weekend_sales_other_stores)
        # Step 3: Plot the results
        labels = ['Open All Weekdays', 'Other Stores']
        average_sales = [avg_weekend_sales_all_weekdays, avg_weekend_sales_other_stores]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, average_sales, color=['teal', 'orange'])
        plt.xlabel('Store Type')
        plt.ylabel('Average Weekend Sales')
        plt.title('Average Weekend Sales Comparison')
        plt.show()
        self.logger.info("Successfully Completing Week day open Stors Analysis")
    
    # Boxplot of Sales by HolidayPeriod
    def Boxplot_of_Sales_by_HolidayPeriod(self,df):
        self.logger.info("Strating  Box Plot sales over Holiday Analysis")
        plt.figure(figsize=(10,6))
        sns.boxplot(x='HolidayPeriod', y='Sales', data=df)
        plt.title('Sales Distribution by Holiday Period')
        plt.xlabel('Holiday Period')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.show()
        self.logger.info("Successfully Completing Box Plot sales over Holiday Analysis")
        
    # Bar plot for mean sales by HolidayPeriod
    def Bar_plot_mean_sales_by_HolidayPeriod(self,df):
        self.logger.info("Strating  Bar Plot sales over Holiday Analysis")
        plt.figure(figsize=(10,6))
        sns.barplot(x='HolidayPeriod', y='Sales', data=df, estimator='mean')
        plt.title('Average Sales by Holiday Period')
        plt.xlabel('Holiday Period')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)
        plt.show()
        self.logger.info("Successfully Completing Bar Plot sales over Holiday Analysis")
    
    # Violin plot of Sales by Holiday Period
    def Violin_plot_Sales_by_HolidayPeriod(self,df):
        self.logger.info("Strating  Violin Plot sales over Holiday Analysis")
        plt.figure(figsize=(10,6))
        sns.violinplot(x='HolidayPeriod', y='Sales', data=df)
        plt.title('Sales Distribution and Density by Holiday Period')
        plt.xlabel('Holiday Period')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.show()
        self.logger.info("Successfully Completing Violin Plot sales over Holiday Analysis")

    # Scatter plot for Sales vs. Customers by HolidayPeriod
    def Scatter_plot_Sales_Customers_by_HolidayPeriod(self,df):
        self.logger.info("Strating  Scatter Plot Sales vs. Customers by HolidayPeriod Analysis")
        plt.figure(figsize=(10,6))
        sns.scatterplot(x='Customers', y='Sales', hue='HolidayPeriod', data=df, palette='Set1')
        plt.title('Sales vs Customers by Holiday Period')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()
        self.logger.info("Successfully Completing Scatter Plot Sales vs. Customers by HolidayPeriod Analysis")
    
    # Calculate the correlation matrix for the selected columns
    def heatmap_correlation_analysis(self, df):
        self.logger.info("Strating heatmap for sales open, customer and promotion Analysis")
        correlation = df[['Sales', 'Customers', 'Open', 'Promo']].corr()

        # Set up the matplotlib figure
        plt.figure(figsize=(8,6))

        # Generate a heatmap of the correlation matrix
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, vmin=-1, vmax=1)

        # Title and labels
        plt.title('Correlation Heatmap of Sales, Customers, Open, and Promo', fontsize=14)
        plt.show()
        self.logger.info("Successfully Completing heatmap for sales open, customer and promotion Analysis")

    
    # Create a sample distance column (e.g., simulate random values for illustration purposes)
    def Create_sample_distance(self, df):
        np.random.seed(42)  # For reproducibility
        df['Distance_to_Competitor'] = np.random.uniform(0, 10, size=len(df))  # Simulated distances

        # Scatter plot of Sales vs Distance to Competitor
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='Distance_to_Competitor', y='Sales')
        plt.title('Sales vs Distance to Competitor')
        plt.xlabel('Distance to Competitor (km)')
        plt.ylabel('Sales')
        plt.show()
        return df 
    def create_sample_city(self, df):
        df['City_Center'] = np.random.choice([1, 0], size=len(df))  # 1 for city center, 0 for non-city center

        # Compare sales in city center vs non-city center
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='City_Center', y='Sales')
        plt.title('Sales Distribution by Store Location Type')
        plt.xlabel('City Center (1: Yes, 0: No)')
        plt.ylabel('Sales')
        plt.show()
        return df
    # Correlation between sales and distance for city center stores
    def city_center_stores(self, df):
        city_center_stores = df[df['City_Center'] == 1]
        distance_sales_corr = city_center_stores[['Sales', 'Distance_to_Competitor']].corr()
        print(distance_sales_corr)

        # Heatmap of correlation for city center stores
        plt.figure(figsize=(8, 6))
        sns.heatmap(city_center_stores[['Sales', 'Distance_to_Competitor']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap for City Center Stores')
        plt.show()
    
    def plot_feature_importance(self, X_train,feature_importances):
        self.logger.info("Starting ploting feature Importance")
        # Visualizing Feature Importance
        features = X_train.columns
        plt.figure(figsize=(8, 6))
        plt.bar(features, feature_importances, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance from RandomForestRegressor')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        self.logger.info("Successfully ploting feature Importance")
    def plot_confidence_interval(self,y_train,y_pred, lower_bound, upper_bound):
        self.logger.info("Starting ploting Predictions with Confidence Intervals")
        # Visualize Predictions with Confidence Intervals
        plt.figure(figsize=(10, 6))
        plt.plot(y_train.values, label='True Values', color='blue')
        plt.plot(y_pred, label='Predicted Values', color='orange')
        plt.fill_between(range(len(y_pred)), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
        plt.xlabel('Index')
        plt.ylabel('Sales')
        plt.title('Predictions with 95% Confidence Intervals')
        plt.legend()
        plt.tight_layout()
        plt.show()
        self.logger.info("Successfully ploting Predictions with Confidence Intervals")