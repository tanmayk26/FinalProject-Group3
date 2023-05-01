# Import required libraries

# Data Manipulation & Visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#def imputer(X_train, X_test):
    # # Separate the numerical and categorical features
    # numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # categorical_features = X_train.select_dtypes(include=object).columns.tolist()
    # # Perform preprocessing for numerical features
    # imputer_num = SimpleImputer(strategy='mean')
    # imputer_cat = SimpleImputer(strategy='most_frequent')
    # X_train[numerical_features] = imputer_num.fit_transform(X_train[numerical_features])
    # X_test[numerical_features] = imputer_num.transform(X_test[numerical_features])
    # X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
    # X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])
    # return X_train, X_test

def imputer(X):
    # Separate the numerical and categorical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=object).columns.tolist()
    # Perform preprocessing for numerical features
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[numerical_features] = imputer_num.fit_transform(X[numerical_features])
    X[categorical_features] = imputer_cat.fit_transform(X[categorical_features])
    return X


def impute_scale(X_train, X_test):
    # Separate the numerical and categorical features
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=object).columns.tolist()
    #X_train, X_test = imputer(X_train, X_test)
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    return X_train, X_test


def impute_one_hot_encode(X_train, X_test):
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=object).columns.tolist()
    # Impute missing values in categorical features
    #X_train, X_test = imputer(X_train, X_test)
    # One-hot encoding
    onehot = OneHotEncoder(handle_unknown='ignore')
    X_train_onehot = onehot.fit_transform(X_train[categorical_features]).toarray()
    X_test_onehot = onehot.transform(X_test[categorical_features]).toarray()
    # Get the names of the encoded columns
    encoded_feature_names = onehot.get_feature_names_out(categorical_features)
    # Put transformed data back into DataFrames
    X_train_onehot = pd.DataFrame(X_train_onehot, columns=encoded_feature_names)
    X_test_onehot = pd.DataFrame(X_test_onehot, columns=encoded_feature_names)
    X_train.reset_index(inplace=True)
    X_test.reset_index(inplace=True)
    # Concatenate the encoded features with the continuous features
    X_train = pd.concat([X_train[numerical_features], X_train_onehot], axis=1)
    X_test = pd.concat([X_test[numerical_features], X_test_onehot], axis=1)
    return X_train, X_test


def plot_confusion_matrix(cm, labels, method_name):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=f'Confusion Matrix-{method_name}',
           ylabel='True label',
           xlabel='Predicted label')

    # Add text annotations to each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()


def yes_no_to_binary(response):
    if response.lower() == "yes":
        return 1
    elif response.lower() == "no":
        return 0
    else:
        return None

