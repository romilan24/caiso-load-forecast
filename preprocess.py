import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def import_data(load_dir, temp_dir):
    # Load data
    load = pd.read_csv(load_dir + 'hourly_caiso_load.csv')
    temp = pd.read_csv(temp_dir + 'combined_weather_data.csv')

    return load, temp

def merge_data(load, temp):
    # Merge dataframes on the datetime column
    merged_df = pd.merge(load, temp, how='inner', left_on='Time', right_on='Datetime')

    # Drop the duplicate 'Time' column
    merged_df = merged_df.drop('Datetime', axis=1)

    return merged_df

def swap_missing_data(merged_df, sf_columns, sj_columns):
    # Replace NaN values in San Francisco columns with values from San Jose columns
    for col_sf, col_sj in zip(sf_columns, sj_columns):
        merged_df[col_sf].fillna(merged_df[col_sj], inplace=True)
        merged_df[col_sj].fillna(merged_df[col_sf], inplace=True)

    return merged_df

def rename_columns(df, column_mapping):
    # Use the rename method to rename the columns
    df = df.rename(columns=column_mapping)
    return df

def interpolate_missing_values(df):
    """
    Interpolate missing values within each hour for numeric columns.

    Parameters:
    - df: DataFrame containing the time series data.

    Returns:
    - DataFrame with NaN values replaced using linear interpolation within each hour.
    """

    result_df = df.copy()

    # Ensure 'Time' column is in datetime format
    result_df['Time'] = pd.to_datetime(result_df['Time'])

    for column in df.columns:
        if column not in ['Time'] and pd.api.types.is_numeric_dtype(result_df[column]):
            try:
                # Convert the column to numeric (if not already)
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce')

                # Extract hour component and interpolate within each hour
                result_df[column] = result_df.groupby(result_df['Time'].dt.hour)[column].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both'))

            except ValueError:
                print(f"Skipping interpolation for non-numeric column: {column}")

    return result_df

def split_data(df, test_size=0.2, random_state=35):

    X = df.drop(['caiso_load_actuals', 'datetime'], axis=1)
    y = df['caiso_load_actuals']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def date_and_hour(df):
    # Convert 'datetime' column to datetime type if not already
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract 'date' and 'hour' columns
    df['date'] = df['datetime'].dt.date
    df['he'] = (df['datetime'].dt.hour + 1) % 25  # Add 1 to hour, ensuring it stays within 0-23 range

    return df

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100