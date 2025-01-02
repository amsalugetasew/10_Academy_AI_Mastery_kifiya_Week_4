import logging
import pandas as pd
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