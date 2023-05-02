# Put code here

# Import required libraries

from tools import imputer, impute_scale, impute_one_hot_encode, yes_no_to_binary
from models import models
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


from tools import plot_confusion_matrix

import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score



def impute_scale(X_train, X_test):
    # Separate the numerical and categorical features
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=object).columns.tolist()
    #X_train, X_test = imputer(X_train, X_test)
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    return X_train, X_test



def models(X_train, X_test, y_train, y_test, labels):
    results = []
    print('Building Models...')

    # 5. Decision Tree Classifier
    method_name = 'Decision Tree Classifier'
    print(f'Building {method_name} model...')
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Decision Tree Accuracy: ", dtt_accuracy)
    # print("Decision Tree ROC AUC: ", dtt_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 6. Random Forest Classifier
    method_name = 'Random Forest Classifier'
    print(f'Building {method_name} model...')
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Random Forest Accuracy: ", rf_accuracy)
    # print("Random Forest ROC AUC: ", rf_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 7. Gaussian
    method_name = 'Naive Bayes'
    print(f'Building {method_name} model...')
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Naive Bayes Accuracy: ", nb_accuracy)
    # print("Naive Bayes ROC AUC: ", nb_roc)
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


print('k-Fold Cross Validation:')
# k = n_neighbors
knn = KNeighborsClassifier(n_neighbors=3)
# K Fold CV K=10
accuracies = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=10)
print(accuracies)

print("average accuracy: ", np.mean(accuracies))
print("average std: ", np.std(accuracies))

knn.fit(X_train, y_train)
# print("test accuracy: ", knn.score(X_test,y_test))


