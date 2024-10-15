import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score

def train_random_forest(X_train, y_train, param_grid, random_state=333):
    """
    Train a Random Forest model with GridSearchCV and return the best model.
    :param X_train: Training data features
    :param y_train: Training data target values
    :param param_grid: Dictionary with parameters for grid search
    :param random_state: Seed for reproducibility
    :return: GridSearchCV fitted model
    """
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_grid = GridSearchCV(rf_model, param_grid, refit=True, verbose=2, cv=5)
    rf_grid.fit(X_train, y_train.ravel())  # Flatten y_train to avoid DataConversionWarning
    return rf_grid

def evaluate_random_forest(rf_grid, X_test, y_test):
    """
    Evaluate the trained Random Forest model and print classification report and precision.
    :param rf_grid: GridSearchCV fitted model
    :param X_test: Test data features
    :param y_test: Test data target values
    """
    rf_y_pred = rf_grid.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_y_pred))
    rf_precision_manual = precision_score(y_test, rf_y_pred)
    print("Random Forest Precision (Manual Calculation):", rf_precision_manual)

def plot_feature_importances(rf_grid, feature_names, top_n=50):
    """
    Plot the top N feature importances for the Random Forest model.
    :param rf_grid: GridSearchCV fitted model
    :param feature_names: List of feature names
    :param top_n: Number of top features to plot
    """
    feature_importances = rf_grid.best_estimator_.feature_importances_
    
    top_n = min(top_n, len(feature_importances))
    
    # Get indices of the top features
    top_indices = np.argsort(feature_importances)[-top_n:]
    top_importances = feature_importances[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances, align='center')
    plt.yticks(range(top_n), top_feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title(f"Top {top_n} Informative Features - Random Forest")
    plt.show()

