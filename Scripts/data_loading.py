import pandas as pd
import logging

class LoadData:
    def __init__(self):
        """Initializes the FileLoader with a list of file paths."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized LoadData class.")
        self.df = {}

    def read_data(self):
        """Reads data from multiple CSV files."""
        self.logger.info("Starting to read data files.")
        try:
            self.df = pd.read_csv("../src/data/rossmann-store-sales/sample_submission.csv", low_memory=False)
            df = pd.read_csv("../src/data/rossmann-store-sales/store.csv", low_memory=False)
            df_test = pd.read_csv("../src/data/rossmann-store-sales/test.csv", low_memory=False)
            df_train = pd.read_csv("../src/data/rossmann-store-sales/train.csv", low_memory=False)
            self.logger.info("Successfully read all data files.")
            return self.df, df, df_test, df_train
        except Exception as e:
            self.logger.error("Failed to read data files.", exc_info=True)
            raise e
    
    def merge_train_with_store(self, df_train, df_store):
        """
        Merges the training data with store information based on the Store column.

        :param df_train: DataFrame containing training data.
        :param df_store: DataFrame containing store data.
        :return: Merged DataFrame.
        """
        self.logger.info("Starting to merge training or testing data with store data.")
        try:
            # Merge df_train and df_store on Store column
            merged_df = pd.merge(df_train, df_store, on="Store", how="left")
            self.logger.info("Successfully merged training or testing data with store data.")
            return merged_df
        except Exception as e:
            self.logger.error("Failed to merge training data with store data.", exc_info=True)
            raise e
        
