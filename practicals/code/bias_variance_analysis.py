import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def bias_variance_analysis(X, y, alpha_values, n_bootstrap=100, top_n_features=10):
    """
    Performs bias-variance analysis on Lasso regression coefficients using bootstrap resampling and visualizes the effect of regularization.

    Parameters:
    - X: Feature Dataframe.
    - y: Response vector Dataframe.
    - alpha_values: Array of regularization (alpha) values to use for Lasso regression.
    - n_bootstrap: Number of bootstrap samples to use for bias-variance analysis (default=100).
    - top_n_features: Number of features to visualize in the plot based on their mean coefficient magnitude (default = 10).
 
    Returns:
    None. It plots the Lasso regression coefficients of the top features for different alpha values, including error bars representing variability across bootstrap samples.
    """
    
    # Store the coefficients for each bootstrap sample and for different alpha values
    coefficients = np.zeros((n_bootstrap, len(alpha_values), X.shape[1]))

     # Pipeline to perform bootstrap sampling and fit Lasso for different alphas after scaling features
    for i in range(n_bootstrap):
        X_resampled, y_resampled = resample(X, y)
        for j, alpha in enumerate(alpha_values):
            lasso_pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Step to scale the features
                ('lasso', Lasso(alpha=alpha))  # Lasso regression step
            ])
            lasso_pipeline.fit(X_resampled, y_resampled)
            coefficients[i, j, :] = lasso_pipeline.named_steps['lasso'].coef_

    # Calculate the mean and standard deviation of the coefficients across bootstrap samples
    coef_mean = np.mean(coefficients, axis=0)
    coef_std = np.std(coefficients, axis=0)

    # Get top N features based on mean coefficient magnitude
    top_features_indices = np.argsort(np.abs(coef_mean).mean(axis=0))[-top_n_features:]
    top_feature_names = X.columns[top_features_indices]

    # Plot the profile of the top N Lasso coefficients over the grid of alphas
    plt.figure(figsize=(10, 6))
    for k in top_features_indices:
        feature_name = X.columns[k]
        plt.errorbar(alpha_values, coef_mean[:, k], yerr=coef_std[:, k], label=feature_name, capsize=5)

    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title(f'Lasso Regression Coefficients for Top {top_n_features} Features Over Alpha with Error Bars')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

