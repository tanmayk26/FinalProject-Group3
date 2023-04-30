# Import required libraries

# Data Manipulation & Visualization libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def impute_scale(X_train, X_test):
    # Separate the numerical and categorical features
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=object).columns.tolist()

    # Perform preprocessing for numerical features
    imputer_num = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train[numerical_features] = imputer_num.fit_transform(X_train[numerical_features])
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = imputer_num.transform(X_test[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    return X_train, X_test

def impute_one_hot_encode(X_train, X_test):
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=object).columns.tolist()
    # One-hot encoding for categorical features

    # Identify categorical features

    # Impute missing values in categorical features
    imputer_cat = SimpleImputer(strategy='most_frequent')

    X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])

    # One-hot encoding
    onehot = OneHotEncoder(handle_unknown='ignore')

    X_train_onehot = onehot.fit_transform(X_train[categorical_features]).toarray()
    X_test_onehot = onehot.transform(X_test[categorical_features]).toarray()

    # Get the names of the encoded columns
    encoded_feature_names = onehot.get_feature_names_out(categorical_features)

    # Put transformed data back into DataFrames
    X_train_onehot = pd.DataFrame(X_train_onehot, columns=encoded_feature_names)
    X_test_onehot = pd.DataFrame(X_test_onehot, columns=encoded_feature_names)

    print(X_train_onehot.head())

    X_train.reset_index(inplace=True)
    X_test.reset_index(inplace=True)

    # Concatenate the encoded features with the continuous features
    X_train = pd.concat([X_train[numerical_features], X_train_onehot], axis=1)
    X_test = pd.concat([X_test[numerical_features], X_test_onehot], axis=1)
    return X_train, X_test


