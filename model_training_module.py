from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class ModelTrainingModule:
    models = ['SVC']
    def svm_cross_validation(x, y, kernel='rbf', C_values=[0.1, 1, 10, 50, 100, 200, 500], gamma_values=['scale', 'auto', 0.001, 0.01, 0.1, 1]):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel=kernel))
        ])
        
        param_grid = {
            'classifier__C': C_values,
            'classifier__gamma': gamma_values
        }
        
        gs = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=5)
        gs.fit(x, y)
        
        print("Best Params:", gs.best_params_)
        print("Best F1 Score:", gs.best_score_)

        return gs.best_estimator_, gs.best_params_, gs.best_score_
    
    def train_svm(x, y, kernel='rbf', C=10, gamma=0.01, test_set_size=0.2, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_size, random_state=random_state)
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        clf.fit(x_train, y_train)
    
        y_pred = clf.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
        return clf

    def undersample(x, y, random_state=42):
        undersampler = RandomUnderSampler(random_state=random_state)
        x_resampled, y_resampled = undersampler.fit_resample(x, y)
        return x_resampled, y_resampled
    
    def tune_svm_hyperparameters(x, y,
                              C_range=[0.1, 1, 10],
                              gamma_range=[0.001, 0.01, 'scale', 'auto'],
                              kernel_options=['linear', 'poly', 'rbf'],
                              cv=5):
        """
        Performs GridSearchCV on SVM with specified C, gamma, and kernel values, and visualizes results.
    
        Parameters:
        - X: Features (numpy array or pandas DataFrame)
        - y: Labels
        - C_range: list of C values to try
        - gamma_range: list of gamma values to try (only used for certain kernels)
        - kernel_options: list of kernel types to try ['linear', 'poly', 'rbf', etc.]
        - cv: number of cross-validation folds
    
        Returns:
        - best_model: Trained SVM model with best hyperparameters
        - results_df: DataFrame of all grid search results
        """
    
        # Define pipeline (scaling + SVM)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC())
        ])
    
        # Define parameter grid
        param_grid = [
            {'svm__kernel': ['linear'], 'svm__C': C_range},
            {'svm__kernel': ['rbf', 'poly'], 'svm__C': C_range, 'svm__gamma': gamma_range}
        ]
    
        # Perform Grid Search
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=0)
        grid.fit(x, y)
    
        # Store results in DataFrame
        results_df = pd.DataFrame(grid.cv_results_)
    
        # Output
        print("Best Parameters:", grid.best_params_)
        print("Best Cross-Validation Accuracy:", round(grid.best_score_ * 100, 2), "%")
    
        return grid.best_estimator_, results_df
    
    def train_random_forest(x , y, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, n_jobs, test_size, random_state, test_split_random_state):
        # Assuming X and y are defined
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=test_split_random_state)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,      # Balance between speed and accuracy
            max_depth=max_depth,          # Prevents overfitting
            min_samples_split=min_samples_split,   # Reduces overfitting
            min_samples_leaf=min_samples_leaf,    # Avoids noisy leaves
            max_features=max_features,   # Default for classification
            bootstrap=bootstrap,        # Better generalization
            n_jobs=n_jobs,             # Parallel processing
            random_state=random_state        # Reproducibility
        )
        model.fit(x_train, y_train)
        y_test = model.predict(x_test)
        
        train_accuracy = model.score(x_train, y_train)
        test_accuracy = model.score(x_test, y_test)
        print(f"Train Accuracy: {train_accuracy:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}")
