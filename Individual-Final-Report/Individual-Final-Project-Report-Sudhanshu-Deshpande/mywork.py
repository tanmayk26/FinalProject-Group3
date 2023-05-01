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
import missingno as msno
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import researchpy as rp
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.impute import SimpleImputer


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score

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


def yes_no_to_binary(response):
    if response.lower() == "yes":
        return 1
    elif response.lower() == "no":
        return 0
    else:
        return None


def models(X_train, X_test, y_train, y_test, labels):
    results = []
    print('Building Models...')

    # 3. KNN
    method_name = 'KNN'
    print(f'Building {method_name} model...')
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("KNN Accuracy: ", knn_accuracy)
    # print("KNN ROC AUC: ", knn_roc)
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

X_viz_list = X.drop(['Year', 'Month', 'Day'], axis=1)

for i in X_viz_list.columns:
    if X[i].dtypes == 'object':
        cat.append(i)
    else:
        cont.append(i)

print(cat)

print("Categorical Variables are", cat)
print("Numerical Variables are", cont)

num_cols = len(cont)
num_rows = (num_cols + 2) // 3  # Round up to nearest multiple of 3
fig, axes = plt.subplots(num_rows, 3, figsize=(16, 4*num_rows))

for i, column in enumerate(X[cont]):
    row_idx = i // 3
    col_idx = i % 3
    axes[row_idx, col_idx].hist(X[column])
    axes[row_idx, col_idx].set_title(column)

plt.tight_layout()
plt.show()


sns.countplot(x=X["WindGustDir"], hue=y)
plt.show()
crosstab, test_results, expected = rp.crosstab(X["WindDir9am"], y,
                                               test="chi-square",
                                               expected_freqs= True,
                                               prop="cell")

print("\n", test_results)

sns.countplot(x=X["WindDir3pm"], hue=y)
plt.show()
crosstab, test_results, expected = rp.crosstab(X["WindDir3pm"], y,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print("\n", test_results)

sns.countplot(x=X["RainToday"], hue=y)
plt.show()

crosstab, test_results, expected = rp.crosstab(X["RainToday"], y,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print("\n", test_results)

sns.countplot(x=X["Location"], hue=y)
plt.show()

crosstab, test_results, expected = rp.crosstab(X["Location"], y,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print("\n", test_results)

feature_selector = SelectKBest(score_func=f_regression, k="all")
fit = feature_selector.fit(X[cont], y)

p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_names = pd.DataFrame(X[cont].columns)
print(input_variable_names)
summary_stats = pd.concat([input_variable_names, p_values, scores], axis=1)
summary_stats.columns = ["input_variable", "p_value", "F-Score"]
summary_stats.sort_values(by="p_value", inplace=True)

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["F-Score"] >= score_threshold) &
                                       (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
print(selected_variables)

# # MODELLING CODE STARTS HERE

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.info())
print(X_train.info())

X_train, X_test = impute_scale(X_train, X_test)
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
