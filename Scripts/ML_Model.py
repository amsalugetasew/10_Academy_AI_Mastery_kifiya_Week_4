import logging
import joblib
from joblib import Parallel, delayed
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
class ML_Model:
    def __init__(self):
        """
        Initialize Regression model.
        """
        # self.df = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized Random Forest Regressor model class.")
        
    def train_and_evaluate_pipeline(self,df, target_column, test_size=0.2, random_state=42, n_estimators=100, max_depth=None):
        """
        Train and evaluate a regression model using a pipeline.

        Parameters:
            df (pd.DataFrame): The input data frame containing features and target.
            target_column (str): The name of the target column.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.
            n_estimators (int): Number of trees in the Random Forest.
            max_depth (int or None): The maximum depth of the trees. Default is None.

        Returns:
            dict: A dictionary containing evaluation metrics (MSE, MAE, R2) and the trained pipeline.
        """
        self.logger.info("Starting model Trainging.")
        # Splitting features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Define the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
        ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        
        self.logger.info("Successfully Train Random Forest Regressor model")
        # Return metrics and the trained pipeline
        return {
            'pipeline': pipeline,
            'X_train':X_train, 
            'X_test':X_test, 
            'y_train':y_train, 
            'y_test':y_test, 
            'metrics': {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        }

    def feature_importance(self,pipeline):
        self.logger.info("Start Computing Feature Importance")
        # Feature Importance
        # Extract feature importance from the RandomForestRegressor
        regressor = pipeline.named_steps['regressor']  # Accessing the RandomForestRegressor from the pipeline
        feature_importances = regressor.feature_importances_
        self.logger.info("Successfully Computing Feature Importance")
        return feature_importances
    def bootstraping_confidence_interval(self, X_train, y_train):
        self.logger.info("Start Computing Confidence Interval Summary")
        #Confidence Interval Estimation
        # Bootstrapping to calculate confidence intervals
        n_bootstraps = 1000  # Number of bootstrap samples
        bootstrap_preds = np.zeros((n_bootstraps, len(y_train)))

        for i in range(n_bootstraps):
            # Bootstrap sampling
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices]
            
            # Train a new pipeline on the bootstrap sample
            bootstrap_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(random_state=42))
            ])
            bootstrap_pipeline.fit(X_bootstrap, y_bootstrap)
            
            # Predict on the original data
            bootstrap_preds[i, :] = bootstrap_pipeline.predict(X_train)

        # Calculate confidence intervals (e.g., 95%)
        lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
        self.logger.info("Successfully Computing Confidence Interval Summary")
        return lower_bound, upper_bound
    def bootstraping_confidence_interval(self, X_train, y_train, n_bootstraps=500, sample_size=0.8):
        # Precompute scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train = y_train.values  # Convert to NumPy if it's a pandas Series
        
        def train_bootstrap(i):
            indices = np.random.choice(len(X_train_scaled), int(len(X_train_scaled) * sample_size), replace=True)
            X_bootstrap = X_train_scaled[indices]
            y_bootstrap = y_train[indices]
            
            # Train model
            model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_bootstrap, y_bootstrap)
            
            return model.predict(X_train_scaled)

        # Parallelize the bootstrapping process
        bootstrap_preds = Parallel(n_jobs=-1)(
            delayed(train_bootstrap)(i) for i in range(n_bootstraps)
        )

        # Calculate confidence intervals
        bootstrap_preds = np.array(bootstrap_preds)
        lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
        
        return lower_bound, upper_bound
    def serialize_model(self, pipeline):
        self.logger.info("Starting Serialize the model")
        # Generate a timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")[:-3]  

        # Construct the filename
        filename = f"../../week_4/regression_pipeline-{timestamp}.pkl"

        # Serialize the model
        joblib.dump(pipeline, filename)

        print(f"Model serialized and saved as: {filename}")
        self.logger.info("Successfully Completing Serialize the model")

