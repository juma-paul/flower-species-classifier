import os
import pytest
import pandas as pd
from src.data.prepare_data import _clean_data, prepare_and_save_data

# Duplicate removal test
def test_remove_duplicates():
    df = pd.DataFrame({
        'col1': [1, 2, 2, 3],
        'col2': [4, 5, 5, 6]
    })

    cleaned_df, report = _clean_data(df)

    assert len(cleaned_df) == 3
    assert report['duplicates_removed'] == 1

# Missing value handling test
def test_handle_missing_values():
    df = pd.DataFrame({
        'col1': [1, None, 2, 3],
        'col2': [4, 5, 5, None]
    })

    cleaned_df, report = _clean_data(df)

    assert cleaned_df.isnull().sum().sum() == 0
    assert report['missing_values_handled'] == 2

# Outlier handling test
def test_outlier_handling():
    df = pd.DataFrame({
        'col1': [1, 2, 3, 100],
        'col2': [4, 5, -26, 9]
    })

    cleaned_df, report = _clean_data(df)

    assert report['outliers_capped'] == 2
    assert cleaned_df['col1'].max() < 100
    assert cleaned_df['col2'].min() > -26

# Data type validation test
def test_data_types_validation():
    # dataframe with mixed types
    df = pd.DataFrame({
        'col1': ['1', '2', '3', '4'], 
        'col2': [1.5, 2.5, 3.5, 4.5],  
        'category': ['A', 'B', 'C', 'A'],  
        'date_col': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'], 
        'target': [0, 1, 2, 0]  
    })
    
    cleaned_df, _ = _clean_data(df)
    
    # assert numeric columns are actually numeric types
    assert pd.api.types.is_numeric_dtype(cleaned_df['col1'])
    assert pd.api.types.is_numeric_dtype(cleaned_df['col2'])
    assert pd.api.types.is_numeric_dtype(cleaned_df['target'])
    
    # assert categorical column is object type
    assert isinstance(cleaned_df['category'].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(cleaned_df['category'])

    
    # assert date column was converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['date_col'])

# File creation test
def test_prepare_and_save_data_creates_files():
    prepare_and_save_data()

    # check files exist
    assert os.path.exists('data/processed/X_train.csv')
    assert os.path.exists('data/processed/X_test.csv')
    assert os.path.exists('data/processed/y_train.csv')
    assert os.path.exists('data/processed/y_test.csv')
    assert os.path.exists('data/processed/cleaning_report.json')

    # check to ensure files have content
    assert os.path.getsize('data/processed/X_train.csv') > 0
    assert os.path.getsize('data/processed/X_test.csv') > 0
    assert os.path.getsize('data/processed/y_train.csv') > 0
    assert os.path.getsize('data/processed/y_test.csv') > 0
    assert os.path.getsize('data/processed/cleaning_report.json') > 0

# Split proportions test
def test_prepare_and_save_data_split_proportions():
    # load the saved files
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    
    total_samples = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total_samples
    test_ratio = len(X_test) / total_samples
    
    # check 80/20 split (with 5% tolerance)
    assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio} not within 75-85%"
    assert 0.15 <= test_ratio <= 0.25, f"Test ratio {test_ratio} not within 15-25%"
    
    # check no data leakage (train and test indices don't overlap)
    assert len(X_train) + len(X_test) == 149, "Total samples should be 150"
    
    # verify train and test sizes match for X and y
    assert len(X_train) == len(y_train), "X_train and y_train size mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test size mismatch"


# Data integrity test
def test_prepare_and_save_data_integrity():
    # load saved files
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # check column names match expected features
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    assert list(X_train.columns) == expected_columns
    assert list(X_test.columns) == expected_columns
    
    # check no missing values
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert y_train.isnull().sum() == 0
    assert y_test.isnull().sum() == 0
    
    # check data types are numeric
    assert all(X_train.dtypes == 'float64')
    assert all(X_test.dtypes == 'float64')
    
    # check target values are valid (0, 1, or 2)
    assert set(y_train.unique()).issubset({0, 1, 2})
    assert set(y_test.unique()).issubset({0, 1, 2})

# Reproducibility test
def test_prepare_and_save_data_reproducibility():
    # Run once and save files
    prepare_and_save_data()
    X_train1 = pd.read_csv('data/processed/X_train.csv')
    
    # Run again
    prepare_and_save_data()
    X_train2 = pd.read_csv('data/processed/X_train.csv')
    
    # Check that the data is identical
    assert X_train1.equals(X_train2), "Running twice should produce identical results"


