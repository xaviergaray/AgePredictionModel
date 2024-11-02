# Model used for age prediction using the Human Age Prediction Synthetic Dataset on Kaggle (link below)
# Created as a project for course project at New Jersey Institute of Technology
# Data link https://www.kaggle.com/datasets/abdullah0a/human-age-prediction-synthetic-dataset/data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

ATTRIBUTES = []


def fetch_data() -> [any, any]:
    """
    Reads the data from the CSV files and returns them as Pandas DataFrames

    Parameters:
    None

    Returns:
    df_train (pd.DataFrame): The raw training data DataFrame.
    df_test (pd.DataFrame): The raw test data DataFrame.
    """
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')

    return df_train, df_test


def prune_data(df) -> [pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Cleans and preprocesses the given DataFrame.

    This function performs several operations on the DataFrame:
    1. Replaces null values with the string 'None'.
    2. Splits the 'Blood Pressure (s/d)' column into 'Systolic_BP' and 'Diastolic_BP' columns.
    4. Converts non-numerical (categorical) values into numerical values and maps them.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be pruned.

    Returns:
    df (pd.DataFrame): The cleaned and preprocessed DataFrame.
    category_values (dict): A dictionary mapping each non-numerical column to its corresponding value mapping.
    """

    # Replace null values with string None
    df.fillna('None', inplace=True)

    # Separate 'Blood Pressure (s/d)'
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure (s/d)'].str.split('/', expand=True).astype(float)
    df.drop(columns='Blood Pressure (s/d)', inplace=True)

    # Correct order of columns to have Blood Pressure values back where the original was
    current_columns = df.columns.tolist()
    new_order = current_columns[:3] + ['Systolic_BP', 'Diastolic_BP'] + current_columns[3:-2]
    df = df[new_order]

    # Convert non-numerical values to numerical
    category_values = {}
    for col in df.select_dtypes(exclude=[np.number]).columns:
        categories = df[col].unique()
        value_map = {value: idx for idx, value in enumerate(categories)}
        df[col] = df[col].map(value_map).astype('category')
        category_values[col] = value_map

    return df, category_values


def normalize_data(df) -> pd.DataFrame:
    """
    Normalizes all columns in the DataFrame using MinMaxScaler.
    ENSURE ALL NUMBERS ARE NUMERICAL

    Parameters:
    df (pd.DataFrame): The input DataFrame to be normalized.

    Returns:
    df (pd.DataFrame): The normalized DataFrame.
    """

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    return df


def visualize_data_distribution(df, category: {map}) -> None:
    """
    Visualizes data distribution through various plots.

    This function generates several visualizations for the given DataFrame, including:
    1. Bar charts for categorical columns based on the provided mapping.
    2. Histograms with KDE plots for all numerical columns.
    3. A correlation matrix heatmap.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be visualized.
    category (dict): A dictionary mapping column names to their categorical value mappings.

    Returns:
    None
    """
    plt.figure(figsize=(15, 15))

    for i, category_name in enumerate(category):
        # Flip category map
        category_values = {v: k for k, v in category[category_name].items()}

        plt.subplot(4, 4, i+1)
        df[category_name].map(category_values).value_counts().plot(kind='bar')
        plt.title(category_name)
        plt.xlabel(category_name)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right', fontsize=8)

    plt.tight_layout(pad=3)
    plt.show()

    # Plotting histograms for all numerical columns
    plt.figure(figsize=(20, 20))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns):
        plt.subplot(4, 4, i + 1)
        sns.histplot(df[column], kde=True, bins=10)
        plt.title(f'Histogram and KDE of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right', fontsize=8)

    plt.tight_layout(pad=3)
    plt.show()

    # Display correlation matrix
    plt.figure(figsize=(16, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout(pad=3)
    plt.show()

    # View Outliers
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=df.select_dtypes(include=[np.number]))
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout(pad=3)
    plt.show()


def main():
    # Get dataset as dataframe and their corresponding category values
    df = {}
    df['train'], df['test'] = fetch_data()

    # Prune dataset and store the category value mappings
    category_values = {}
    df['train'], category_values['train'] = prune_data(df['train'])
    df['test'], category_values['test'] = prune_data(df['test'])

    visualize_data_distribution(df['train'], category_values['train'])

    # Normalize the datasets
    df_norm = {}
    df_norm['train'] = normalize_data(df['train'])
    df_norm['test'] = normalize_data(df['test'])


if __name__ == '__main__':
    main()
