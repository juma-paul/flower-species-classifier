import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def _clean_data(dataframe, outlier_method='iqr'):
    """Comprehensive data cleaning"""
    report = {}
    df = dataframe.copy()

    # 1. remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    dups_removed  = initial_rows - len(df)
    report['duplicates_removed'] = dups_removed

    # 2. identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # try converting categorical columns to numeric/datetime
    for col in categorical_cols.copy():
        # Try numeric first
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
            categorical_cols.remove(col)
            continue
        except:
            pass
        
        # Try datetime if numeric failed
        try:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='raise')
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass
    
    report['column_types'] = {
        'numeric': len(numeric_cols),
        'categorical': len(categorical_cols),
        'datetime': len(datetime_cols)
    }

    # 3. handle missing values by type
    missing_before = df.isnull().sum().sum()
    
    # numeric: fill with median
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # categorical: fill with mode or 'unknown'
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')

    # datetime: forward fill and backfill(index 0)
    for col in datetime_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(method='bfill')

    missing_after = df.isnull().sum().sum()
    report['missing_values_handled'] = int(missing_before) - int(missing_after)

    # 4. detect and handle outliers using IQR method (numeric only)
    outliers_handled = 0
    for col in numeric_cols:
        if col not in ['target', 'id', 'labels']:
            if outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                df[col]

                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif outlier_method == 'zscore':
                lower_bound = df[col].mean() - 3 * df[col].std()
                upper_bound = df[col].mean() + 3 * df[col].std()

            # cap outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            df[col] = df[col].clip(lower_bound, upper_bound)
            outliers_handled += outliers
    
    report['outliers_capped'] = outliers_handled

    # 5. validate and enforce data types
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # 6. remove any remaining rows with NaN (from type conversion failures)
    rows_before_final = len(df)
    df = df.dropna()
    report['invalid_rows_removed'] = rows_before_final - len(df)

    # 7. final report
    report['final_rows'] = len(df)
    report['final_cols'] = len(df.columns)

    print('\n--- Data Cleaning Report ---')
    for key, value in report.items():
        print(f'  {key}: {value}')
    print('-'*30 + '\n')

    return df, report

# Prepare and save data
def prepare_and_save_data():
    # load iris dataset
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data['target'] = iris.target

    # rename columns
    name_map = {
        'sepal length (cm)': 'sepal_length', 
        'sepal width (cm)': 'sepal_width', 
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width'
    }

    cleaned_data, report = _clean_data(iris_data)

    cleaned_data.rename(columns=name_map, inplace=True)

    # separate features from target
    X = cleaned_data.drop(columns=['target'])
    y = cleaned_data['target']

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=7)

    # save splits into a csv format into data/processed
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # Save data cleaning report
    with open('data/processed/cleaning_report.json', 'w') as f:
        json.dump(report, f, indent=4, default=int)


# Test script
def main():
    prepare_and_save_data()

if __name__ == '__main__':
    main()

    

