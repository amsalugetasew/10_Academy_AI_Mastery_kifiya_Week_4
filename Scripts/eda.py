import logging
class EDA:
    def __init__(self):
        """Initializes the FileLoader with a list of file paths."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized EDA class.")
        self.df = {}
    def check_promotion_distribution(self, df_train, df_test):
        """
        Compares the distribution of promotions in training and test sets.
        
        :param df_train: DataFrame containing training data.
        :param df_test: DataFrame containing test data.
        :return: None
        """
        self.logger.info("Starting to checking promotion distribution on training and test dataset.")
        # Calculate percentage of rows with active promotions in training data
        promo_train = df_train['Promo'].mean() * 100
        promo2_train = df_train['Promo2'].mean() * 100
        
        # Calculate percentage of rows with active promotions in test data
        promo_test = df_test['Promo'].mean() * 100
        promo2_test = df_test['Promo2'].mean() * 100
        
        # Print the results
        print(f"Promotion distribution in training set:")
        print(f" - Promo: {promo_train:.2f}% active")
        print(f" - Promo2: {promo2_train:.2f}% active")
        
        print(f"\nPromotion distribution in test set:")
        print(f" - Promo: {promo_test:.2f}% active")
        print(f" - Promo2: {promo2_test:.2f}% active")
        
        # Compare the distributions
        if abs(promo_train - promo_test) < 5:
            print("\nThe Promo distribution between train and test sets is similar.")
        else:
            print("\nThe Promo distribution between train and test sets is significantly different.")
        
        if abs(promo2_train - promo2_test) < 5:
            print("The Promo2 distribution between train and test sets is similar.")
        else:
            print("The Promo2 distribution between train and test sets is significantly different.")
        if promo_train:
            self.logger.info("Successfully checking promotion distribution of training and testing of store data.")
        else:
            self.logger.error("Failed to  checking promotion distribution of training and testing of store data.", exc_info=True)