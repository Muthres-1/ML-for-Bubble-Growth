import numpy as np                                  # Numerical computations
import pandas as pd                                 # Data handling (DataFrames)
import matplotlib.pyplot as plt                     # Plotting and visualization
import sklearn                                      # Machine learning utilities

import xgboost                                      # XGBoost library
from xgboost import XGBRegressor                    # XGBoost regressor model

from sklearn.metrics import mean_absolute_error     # MAE metric
from sklearn.metrics import median_absolute_error   # Median AE metric
from sklearn.metrics import mean_poisson_deviance   # Poisson deviance metric
from sklearn.metrics import r2_score                # R² score
from sklearn.metrics import explained_variance_score# Explained variance metric

from sklearn.model_selection import train_test_split# Train–test split
import pickle                                       # Save/load trained model

from sklearn.model_selection import GridSearchCV    # Hyperparameter tuning

df_ex = pd.read_excel("data set.xlsx")     # Load experimental dataset

df_ex.head()                                        # Display first 5 rows
df_ex.describe()                                    # Statistical summary
df_ex.info()                                        # Data types and null info
df_ex.isnull().sum()                                # Count null values
df_ex.duplicated().sum()                            # Count duplicate rows
df_ex.drop_duplicates(inplace=True)                 # Remove duplicate rows

df_ex.dropna(inplace=True)                          # Remove rows with null values

xe = df_ex.drop("d/dMax", axis=1)                   # Features (input variables)
ye = df_ex["d/dMax"]                                # Target variable

np.random.seed(42)                                  # Fix random seed for reproducibility

x_tre, x_tse, y_tre, y_tse = train_test_split(      # Split data into train/test
    xe, ye, train_size=0.85
)

model_ex = XGBRegressor()                           # Initialize XGBoost model
model_ex.fit(x_tre, y_tre)                          # Train the model
y_pred_ex = model_ex.predict(x_tse)                 # Predict on test data

mae_ex = mean_absolute_error(y_tse, y_pred_ex)      # Compute MAE (test set)
print(f"Experimental Data Mean Absolute Error: {mae_ex}")

mae = mean_absolute_error(y_tre, y_pred_ex)         # MAE (incorrectly using train y)
mdae = median_absolute_error(y_tse, y_pred_ex)      # Median absolute error
mpd = mean_poisson_deviance(y_tse, y_pred_ex)       # Mean Poisson deviance
r2 = r2_score(y_tse, y_pred_ex)                     # R² score
evs = explained_variance_score(y_tse, y_pred_ex)    # Explained variance score

print(f"Mean Absolute Error: {mae}")                 # Print MAE
print(f"Median Absolute Error: {mdae}")              # Print Median AE
print(f"Mean Poisson Deviance: {mpd}")               # Print MPD
print(f"R-squared: {r2}")                            # Print R²
print(f"Explained Variance Score: {evs}")            # Print EVS

param_grid = {                                      # GridSearch hyperparameters
    'n_estimators': [100, 300, 500],                 # Number of trees
    'learning_rate': [0.01, 0.1, 0.3],               # Learning rate
    'max_depth': [3, 5, 7],                          # Tree depth
    'min_child_weight': [1, 3, 5],                   # Min child weight
    'subsample': [0.5, 0.7, 0.9],                    # Row subsampling
    'colsample_bytree': [0.5, 0.7, 0.9],             # Column subsampling
    'gamma': [0, 0.1, 0.3]                           # Split regularization
}

reg = GridSearchCV(                                 # GridSearch setup
    estimator=m_ex,                                 # Base model (as defined)
    param_grid=param_grid,                          # Parameter grid
    cv=10,                                          # 10-fold cross-validation
    verbose=2                                       # Verbose output
)

reg.fit(x_tre, y_tre)                               # Train GridSearch model
reg.score(x_tse, y_tse)                             # Evaluate best R² on test set

best_model = reg.best_estimator_                    # Extract best model
y_pred = best_model.predict(x_tse)                  # Predict using best model
mae_best = mean_absolute_error(y_tse, y_pred)       # MAE of best model
print(f"Best Model Mean Absolute Error: {mae_best}")

testingx1 = xe[:30]                                 # Test subset 1 (features)
testingy1 = ye[:30]                                 # Test subset 1 (labels)
testingx2 = xe[30:59]                               # Test subset 2 (features)
testingy2 = ye[30:59]                               # Test subset 2 (labels)
testingx3 = xe[59:92]                               # Test subset 3 (features)
testingy3 = ye[59:92]                               # Test subset 3 (labels)
testingx4 = xe[92:105]                              # Test subset 4 (features)
testingy4 = ye[92:105]                              # Test subset 4 (labels)
testingx5 = xe[105:115]                             # Test subset 5 (features)
testingy5 = ye[105:115]                             # Test subset 5 (labels)

yp1 = reg.predict(testingx1)                        # Predictions for subset 1
yp2 = reg.predict(testingx2)                        # Predictions for subset 2
yp3 = reg.predict(testingx3)                        # Predictions for subset 3
yp4 = reg.predict(testingx4)                        # Predictions for subset 4
yp5 = reg.predict(testingx5)                        # Predictions for subset 5

plt.plot(np.linspace(0, 1, len(yp1)), yp1,           # Predicted diameter vs time
         label="Predicted Diameter")
plt.plot(np.linspace(0, 1, len(yp1)), testingy1,     # True diameter vs time
         label="True Diameter")
plt.legend()                                        # Show legend
plt.xlabel("Time")                                  # X-axis label
plt.ylabel("Diameter")                              # Y-axis label
plt.title("True and Predicted Diameters")            # Plot title
plt.grid(True)                                      # Enable grid
plt.show()                                          # Display plot

plt.scatter(testingy1, yp1, label="Plot")            # True vs predicted scatter
plt.xlabel('True Diameter')                          # X-axis label
plt.ylabel('Predicted Diameter')                     # Y-axis label
plt.plot(np.linspace(0, 1), np.linspace(0, 1),        # Reference line y = x
         color="orange", label="x = y")
plt.legend()                                        # Show legend
plt.title('Predicted vs True Diameter')              # Plot title
plt.show()                                          # Display plot

# (Same plotting logic repeated for yp2, yp3, yp4, yp5)

pickle.dump(reg, open('model_final.pkl', 'wb'))      # Save trained model to file

def evaluation(y_true, y_preds):                     # Evaluation function
    mean_absolute = mean_absolute_error(y_true, y_preds)   # MAE
    median_absolute = median_absolute_error(y_true, y_preds)# Median AE
    mean_poisson = mean_poisson_deviance(y_true, y_preds)  # MPD
    r2_scor = r2_score(y_true, y_preds)              # R² score
    explained_variance = explained_variance_score(   # Explained variance
        y_true, y_preds
    )

    print(f"the mean absolute error is  {mean_absolute}")   # Print MAE
    print(f"the median absolute error is  {median_absolute}")# Print Median AE
    print(f"the mean poisson deviance is  {mean_poisson}")  # Print MPD
    print(f"the r2_score is  {r2_scor*100: .2f}")           # Print R² (%)
    print(f"the explained_variance is  {explained_variance}")# Print EVS

evaluation(testingy1, yp1)                           # Evaluate subset 1
evaluation(testingy2, yp2)                           # Evaluate subset 2
evaluation(testingy3, yp3)                           # Evaluate subset 3
evaluation(testingy4, yp4)                           # Evaluate subset 4
evaluation(testingy5, yp5)                           # Evaluate subset 5


