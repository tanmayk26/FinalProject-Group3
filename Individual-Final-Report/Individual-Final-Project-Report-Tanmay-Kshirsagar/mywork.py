# Put code here

# Import required libraries

from tools import imputer, impute_scale, impute_one_hot_encode, yes_no_to_binary
from models import models

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


from tools import plot_confusion_matrix

import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score



# One hot encoder
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



def models(X_train, X_test, y_train, y_test, labels):
    results = []
    print('Building Models...')

    # 1. Logistic Regression
    method_name = 'Logistic Regression'
    print(f'Building {method_name} model...')
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Logistic Regression Accuracy: ", lg_accuracy)
    # print("Logistic Regression ROC AUC: ", lg_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 9. MLP Classifier
    method_name = 'MLP Classifier'
    print(f'Building {method_name} model...')
    classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                               solver='sgd', verbose=10, random_state=42,
                               learning_rate_init=.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("MLP Classifier Accuracy: ", mlp_accuracy)
    # print("MLP Classifier ROC AUC: ", mlp_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 10. XGB Classifier
    method_name = 'XGB Classifier'
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("XGB Classifier Accuracy: ", mlp_accuracy)
    # print("XGB Classifier ROC AUC: ", mlp_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-score'])
    print(results_df)




df = pd.read_csv('weatherAUS.csv')
df = df.dropna(subset=['RainTomorrow'])
labels = ['No', 'Yes']
print(df.head(10))
print(df.info())
print(df.isnull().sum())

df['RainTomorrow'] = df['RainTomorrow'].apply(yes_no_to_binary)

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop('Date', axis=1)

#df = df.dropna(axis=0, how='any', subset=["RainTomorrow"])

print(df.shape)


X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# Perform preprocessing for numerical features
X = imputer(X)

# 6 columns are of type 'object' and remaining of 'float'
cat, cont = [], []


# # MODELLING CODE STARTS HERE

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.info())
print(X_train.info())

#X_train, X_test = impute_scale(X_train, X_test)
X_train, X_test = impute_one_hot_encode(X_train, X_test)

# models(X_train, X_test, y_train, y_test, labels)
sns.countplot(x = y_train, hue = df['RainTomorrow'])
plt.show()

sm = SMOTE()
#(sampling_strategy='minority', random_state=42)

# Fit and resample the data
X_train, y_train = sm.fit_resample(X_train, y_train)
print(y_train)

sns.countplot(x = y_train)
plt.show()
models(X_train, X_test, y_train, y_test, labels)

# from lazypredict.Supervised import LazyClassifier
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)
#
# print(models)

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)*100
d = [n for n in range(len(cumsum))]
plt.figure(figsize=(10, 10))
plt.plot(d, cumsum, color='red', label='cumulative explained variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axhline(y=95, color='k', linestyle='--', label='95% Explained Variance')
plt.legend(loc='best')

pca_train = PCA(.95)
pca_train.fit(X_train)
X_train_pca = pca_train.transform(X_train)
# pca_test = PCA(.95)
# pca_test.fit(X_test)
X_test_pca = pca_train.transform(X_test)

print('After PCA:')
models(X_train_pca, X_test_pca, y_train, y_test, labels)