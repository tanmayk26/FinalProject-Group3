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


df = pd.read_csv('weatherAUS.csv')
df = df.dropna(subset=['RainTomorrow'])
labels = ['No', 'Yes']
print(df.head(10))

print(f'The number of rows are {df.shape[0] } and the number of columns are {df.shape[1]}')

print(df.info())

print(df.nunique())

print(df.isnull().sum())

msno.bar(df, sort='ascending')
plt.show()

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

