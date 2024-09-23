from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# for plot_roc function
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt


def create_and_train_knn_pipeline(X_train, y_train, param_grid, n_cv=10, random_state= None):
    """
    Creates and trains a KNeighborsClassifier model using GridSearchCV.
    
    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - param_grid (dict): The hyperparameter grid to search over. 
                         Example: {'knn__n_neighbors': [1, 3, 5, 7]}
    - n_cv (int): Number of cross-validation folds. Default is 10.
    - random_state (int or None): Random seed for reproducibility.
    
    Return: model_grid (GridSearchCV object): The trained GridSearchCV object.
            cv_results (DataFrame): The cross-validation results in a DataFrame.
    """
    
    # Initialize the KNeighborsClassifier
    knn = KNeighborsClassifier()

    scaler = StandardScaler()
    
    # Create the pipeline
    model_pipeline = Pipeline([
                    ("scaler", scaler),
                    ("knn", knn)
                    ])

    # Cross-validation with a random state for reproducibility
    cv_strategy = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state= random_state)

    # Create the GridSearchCV object
    model_grid = GridSearchCV(estimator=model_pipeline, 
                              param_grid=param_grid, 
                              cv=cv_strategy,  # Use the custom cross-validation strategy
                              scoring='roc_auc',   # the default would be accuracy, but roc_auc is more related to the question
                            )
    
    # Fit the model using GridSearchCV
    model_grid.fit(X_train, y_train)
    
    # Cross-validation results
    cv_results = pd.DataFrame(model_grid.cv_results_)
    
    return model_grid, cv_results



# Function to plot ROC curves and visualize FPR, TPR, and Thresholds
def plot_roc(grid_search, X_bc_train, y_bc_train, X_test, y_test):
    """
    Plots ROC curves for different values of neighbors used in grid search and visualizes FPR, TPR, and Thresholds.

    Parameters:
    grid_search: The grid search object containing the results of the cross-validation.
    X_bc_train: Training feature set.
    y_bc_train: Training target set.
    X_test: Test feature set.
    y_test: Test target set.

    Return:None
    """
    # Get the list of neighbors used in grid search
    neighbors_list = grid_search.param_grid['knn__n_neighbors']
    
    plt.figure(figsize=(8, 8))
    
    for i, neighbors in enumerate(neighbors_list):
        # Get the model corresponding to this neighbors value
        model = grid_search.best_estimator_.set_params(knn__n_neighbors=neighbors)
        model.fit(X_bc_train, y_bc_train)  # Refit with this neighbor value
        
        # Predict probabilities for the test set
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve, AUC, and Thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        
        # Print as table for each neighbors value using pandas DataFrame
        data = {
            'False Positive Rate (FPR)': fpr,
            'True Positive Rate (TPR)': tpr,
            'Thresholds': thresholds
        }
        df = pd.DataFrame(data)
        print(f'n_neighbors={neighbors}')
        print(df)
        print("-" * 60)
        
        # Plot the ROC curve using RocCurveDisplay
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=f'n_neighbors={neighbors}')
        display.plot(ax=plt.gca())  # Plot on the current axis
    
    # Plot the diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Different n_neighbors')
    plt.show()

