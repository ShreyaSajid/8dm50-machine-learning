from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Reusable function for polynomial regression grid search and plotting
def polynomial_regression_experiment(X_train, y_train, X_test, y_test, degrees, cv=5):
    """
    Perform polynomial regression using grid search on the given dataset, and plot MSE for training and validation.
    
    Parameters:
    - X_train, y_train: training data
    - X_test, y_test: test data
    - degrees: list of polynomial degrees to try in the grid search
    - cv: number of cross-validation folds (default: 5)
    """
    # Set up the pipeline with PolynomialFeatures, StandardScaler, and Linear Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalize the input features
        ('poly', PolynomialFeatures()),  
        ('linear', LinearRegression())  
    ])

    # Define the hyperparameters for grid search: the degree of the polynomial
    param_grid = {
        'poly__degree': degrees  # Testing polynomial degrees
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)

    # Fit the model using grid search
    grid_search.fit(X_train, y_train)

    # Get the cross-validation results
    cv_results = grid_search.cv_results_

    # Extract mean test scores (neg_mean_squared_error) and polynomial degrees
    mean_test_scores = -cv_results['mean_test_score']  # Negate the negative MSE to get positive MSE
    mean_train_scores = -cv_results['mean_train_score']  # Extract mean train scores and negate the negative MSE

    # Plot the learning curve (validation MSE as a function of polynomial degree)
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, mean_test_scores, marker='o', color='b', label='Validation MSE')
    plt.plot(degrees, mean_train_scores, marker='o', color='r', label='Training MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training and Validation MSE vs. Polynomial Degree')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print the best polynomial degree and validation MSE score
    print(f"Best polynomial degree: {grid_search.best_params_['poly__degree']}")
    print(f"Validation MSE of best model: {-grid_search.best_score_}")  # Negative MSE was used, so negate it

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {test_mse}")