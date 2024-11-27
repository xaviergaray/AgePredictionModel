# Model used for age prediction using the Human Age Prediction Synthetic Dataset on Kaggle (link below)
# Created as a project for course project at New Jersey Institute of Technology
# Data link https://www.kaggle.com/datasets/abdullah0a/human-age-prediction-synthetic-dataset/data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow messages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.inspection import permutation_importance
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.svm import SVR

ATTRIBUTES = []


def fetch_data() -> [any, any]:
    """
    Reads the data from the CSV file and returns it as a Pandas DataFrame

    Parameters:
    None

    Returns:
    df_train (pd.DataFrame): The raw training data DataFrame.
    """
    df_train = pd.read_csv('../data/Train.csv')
    return df_train


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
    normalized_array = scaler.fit_transform(df)

    return pd.DataFrame(normalized_array, columns=df.columns, index=df.index)


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


def KNN_model(X_train, X_test, y_train, y_test):
    rmse_val = [] #to store rmse values for different k
    for K in range(20):
        K = K+1
        model = KNeighborsRegressor(n_neighbors = K)
        model.fit(X_train, y_train)  #fit the model
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values

    #plotting the rmse values against k values
    curve = pd.DataFrame(rmse_val) #elbow curve 
    curve.plot()
    plt.show()

    #print the lowest rmse value and its K
    rmse_val = np.array(rmse_val)
    print(rmse_val.min())
    print(rmse_val.argmin())
    return model

def LinearRegression_model(df_norm):

    X = df_norm.drop(columns=['Age (years)'])
    y = df_norm['Age (years)']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Linear Regression using all features:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    print("Linear Regression using only 5 most correlated features:")
    XCor5 = df_norm[['Systolic_BP', 'Diastolic_BP', 'Cholesterol Level (mg/dL)', 'Blood Glucose Level (mg/dL)', 'Hearing Ability (dB)']]
    XCor5_train, XCor5_test, y_train, y_test = train_test_split(XCor5, y)
    model = LinearRegression()
    model.fit(XCor5_train, y_train)
    print(model.score(XCor5_test, y_test))

    return model

def train_neural_network_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tf.keras.Sequential:
    """
    Trains a sequential neural network.

    Parameters:
    X_train (pd.DataFrame): The training features DataFrame.
    y_train (pd.DataFrame): The training labels DataFrame.

    Returns:
    model (tensorflow.keras.Sequential): The trained model.
    """
    model = tf.keras.Sequential([
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', 'r2_score'])
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=0)

    return model


def neural_network_tasks(X, X_train, y_train, X_test, y_test):
    """
    Performs all functions related to the neural network

    Parameters:
    X (pd.DataFrame): The all features DataFrame.
    X_train (pd.DataFrame): The training features DataFrame.
    y_train (pd.DataFrame): The training labels DataFrame.
    X_test (pd.DataFrame): The testing features DataFrame.
    y_test (pd.DataFrame): The testing labels DataFrame.

    Returns:
    None
    """
    # Train and test models
    print("Training Neural Network...")
    neural_network = train_neural_network_model(X_train, y_train)
    print("Done!")
    loss, mae, mse, r2 = neural_network.evaluate(X_test, y_test)
    print(f"Mean Absolute Error on Test Data: {mae} | Mean Squared Error: {mse} | R-Squared: {r2}")

    print("Feature importance by order:")
    result, sorted_idx = get_feature_importances(neural_network, X_test, y_test)
    for idx in sorted_idx:
        print(f"{X.columns.to_list()[idx]}: {result.importances_mean[idx]:.4f}")


def get_feature_importances(model, X_test, y_test):
    """
    Trains a sequential neural network.

    Parameters:
    model: Any trained model
    X_test (pd.DataFrame): The test features
    y_test (pd.DataFrame): The test labels

    Returns:
    result (Dictionary-like object): Result of the permutation_importance function
    sorted_idx (List): The indexes of the features in order of importance
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42,
                                    scoring='neg_mean_squared_error')

    sorted_idx = result.importances_mean.argsort()

    return result, sorted_idx

def random_forest(X, X_train, y_train, X_test, y_test):

    # Since there are numerous categorical features in our dataset we need to encode them numerically
    # Which is done in the prune_data function
    print("Begin Random Forest Grid Search")

    # Tests different parameters for RandomForestRegressor to see which is best
    param_grid = {
        "n_estimators" : [80, 100, 150],
        "criterion" : ["squared_error", "absolute_error", "poisson"],
        "max_depth" : [3, 10, None],
        "min_samples_split" : [2, 5, 10],
        "random_state" : [42]
    }

    gs = GridSearchCV(estimator=RandomForestRegressor(),
                  param_grid=param_grid,
                  scoring="r2",
                  refit=True,
                  cv=3,
                  n_jobs=-1)
    
    #rf = RandomForestRegressor(n_estimators=100, criterion= "squared_error",  n_jobs=-1, random_state=42)

    gs.fit(X_train, y_train)

    print("Random Forest Grid Search Finish")

    predictions = gs.predict(X_test) 
    print("Finished with parameters:", gs.best_params_)
    print("R-Squared:", gs.best_score_, "MSE:", sqrt(mean_squared_error(y_test, predictions)))
    print("\n")

    # Takes the first tree [0]
    rf = gs.best_estimator_
    first_rf = rf.estimators_[0]

    # Plot but only limit to depth of 4 including root
    tree.plot_tree(first_rf, max_depth=3, feature_names = X_train.columns, filled = True, fontsize=8)
    
    #plt.savefig("first_rf.png", dpi=900)
    plt.figure(figsize=(20, 12))
    plt.tight_layout()
    plt.show()
    


def svr(X, X_train, y_train, X_test, y_test):

    print("Begin SVR Grid Search")

    # Using param_grid we were able to test and narrow down to specific ranges of values for the given parameters 
    param_grid = {
        "kernel" : ["rbf", "poly"],
        "gamma" : ["scale", "auto", 1, 5, .5],
        "C" : [.8, .9, 1, 5],
        "epsilon" : [.07, .08, .1, .3],
    }

    gs = GridSearchCV(estimator=SVR(),
                  param_grid=param_grid,
                  scoring="r2",
                  refit=True,
                  cv=5,
                  n_jobs=-1)


    gs.fit(X_train, y_train)

    print("SVR Grid Search Finish")

    predictions = gs.predict(X_test) 
    print("Finished with parameters:", gs.best_params_)
    print("R-Squared:", gs.best_score_, "MSE:", sqrt(mean_squared_error(y_test, predictions)))
    print("\n")


def main(visualize_data: bool):
    # Get dataset as dataframe and their corresponding category values
    df = fetch_data()

    # Prune dataset and store the category value mappings
    df, category_values = prune_data(df)

    # Visualize raw data
    if visualize_data:
        visualize_data_distribution(df, category_values)

    # Normalize the datasets
    df_norm = normalize_data(df)

    # Define features and labels
    X = df_norm.drop(columns=['Age (years)'])
    y = df_norm['Age (years)']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Run Neural Network related tasks
    neural_network_tasks(X, X_train, y_train, X_test, y_test)

    # Random Forest with Tree Output
    random_forest(X, X_train, y_train, X_test, y_test)

    # Suppot Vector Regression
    svr(X, X_train, y_train, X_test, y_test)


    #Run two KNN models
    print("Using all features:")
    model = KNN_model(X_train, X_test, y_train, y_test)

    print("Using only 5 most correlated features:")
    XCor5 = df_norm[['Systolic_BP', 'Diastolic_BP', 'Cholesterol Level (mg/dL)', 'Blood Glucose Level (mg/dL)', 'Hearing Ability (dB)']]
    XCor5_train, XCor5_test, y_train, y_test = train_test_split(XCor5, y)
    model = KNN_model(XCor5_train, XCor5_test, y_train, y_test)

    #Run two linear regression modles
    model = LinearRegression_model(df_norm)



if __name__ == '__main__':
    main(visualize_data=False)
