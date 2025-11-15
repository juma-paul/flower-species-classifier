import mlflow
import joblib
import pandas as pd
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Training function
def train_model(X_train, X_test, y_train, y_test, n_neighbors=5, run_name=None):
    # start MLflow tracking
    with mlflow.start_run(run_name=run_name):
        # create KNN model
        model = KNeighborsClassifier(n_neighbors)

        # Train the model on the data
        model.fit(X_train, y_train)

        # Evaluate the model and calculate metrics
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
        conf_matrix = confusion_matrix(y_test, predictions)

        # Log parameters
        mlflow.log_param('n_neighbors', n_neighbors)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)

        # Log confusion matrix as an artifact(image)
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        # Log model
        mlflow.sklearn.log_model(
            model, 
            name='knn_model', 
            input_example=X_train[:1]
        )

        # Print results
        print(f"""
            Accuracy: {accuracy:.4f},
            Precision: {precision:.4f},
            Recall: {recall:.4f},
            F1-Score: {f1:.4f}
        """)

        # Save the model locally
        joblib.dump(model, 'models/knn_model.pkl')

        return model
    
# Hyperparameter tuning
def _tune_hyperparameters(X_train, y_train):
    model = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print(f'Best parameters: {grid.best_params_}')
    print(f'Best score: {grid.best_score_:.4f}')

    return grid.best_params_
    
# Test script
def main():
    # Setup MLflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('iris_classification')

    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    # Step 1: Train baseline model
    print("\n------- Training Baseline Model -------")
    train_model(X_train, X_test, y_train, y_test, n_neighbors=5, run_name='baseline')

    # Step 2: Tune hyperparameters
    print("\n------- Tuning Hyperparameters -------")
    best_params = _tune_hyperparameters(X_train, y_train)

    # Step 3: Train with best parameters
    print("\n------- Training Tuned Model -------")
    train_model(X_train, X_test, y_train, y_test, n_neighbors=best_params['n_neighbors'], run_name='tuned')

if __name__ == '__main__':
    main()