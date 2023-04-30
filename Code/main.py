# Import required libraries

# Calling different modules
from models import models
from tools import impute_scale, impute_one_hot_encode, yes_no_to_binary

# Data Manipulation & Visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectKBest, chi2



# Feature Selection
from sklearn.ensemble import RandomForestClassifier

# import warnings
# warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('weatherAUS.csv')

# Drop columns that are not required
df = df.drop(['Date',  'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'], axis=1)
df.info()
# Remove rows with missing values in the target variable
df = df.dropna(subset=['RainTomorrow'])

# Split the dataset into features and target variable
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

sns.histplot(df['RainTomorrow'])
plt.show()


# Separate categorical and discrete data
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
disc_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Plot categorical data
n_rows = (len(cat_cols) // 3) + (len(cat_cols) % 3 > 0) # Calculate number of subplot rows
plt.figure(figsize=(12, n_rows*4))
for i, col in enumerate(cat_cols):
    plt.subplot(n_rows, 3, i+1)
    sns.countplot(x=col, data=df)
    plt.title(col)
plt.tight_layout()
plt.show()

# # Plot discrete data
# n_rows = (len(disc_cols) // 3) + (len(disc_cols) % 3 > 0) # Calculate number of subplot rows
# plt.figure(figsize=(12, n_rows*4))
# for i, col in enumerate(disc_cols):
#     plt.subplot(n_rows, 3, i+1)
#     sns.histplot(x=col, data=df, kde=True)
#     plt.title(col)
# plt.tight_layout()
# plt.show()

# print(X.head())
# print(y.head())

# Perform Hypothesis Testing
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Perform preprocessing for numerical features
imputer_num = SimpleImputer(strategy='mean')
X[numerical_features] = imputer_num.fit_transform(X[numerical_features])

X_star=X[numerical_features]
X_star=pd.DataFrame(X_star)
print(X_star.head())
# Apply the function to the 'response' column and create a new 'binary_response' column
y_test= y.apply(yes_no_to_binary)


feature_selector = SelectKBest(f_regression, k="all")
fit = feature_selector.fit(X_star, y_test)

p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_names = pd.DataFrame(X_star.columns)
print(input_variable_names)
summary_stats = pd.concat([input_variable_names, p_values, scores], axis=1)
summary_stats.columns = ["input_variable", "p_value", "chi2_score"]
summary_stats.sort_values(by="p_value", inplace=True)

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["chi2_score"] >= score_threshold) &
                                       (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
X_new = X_star[selected_variables]
X_new=pd.DataFrame(X_new)
print(X_new.head())

categorical_features = X.select_dtypes(include=object).columns.tolist()
X_car=X[categorical_features]
import researchpy as rp
crosstab, test_results, expected = rp.crosstab(X_car["WindGustDir"], y_test,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print(test_results)

import researchpy as rp
crosstab, test_results, expected = rp.crosstab(X_car["WindDir9am"], y_test,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print(test_results)
import researchpy as rp
crosstab, test_results, expected = rp.crosstab(X_car["WindDir3pm"], y_test,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print(test_results)
import researchpy as rp
crosstab, test_results, expected = rp.crosstab(X_car["RainToday"], y_test,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print(test_results)
import researchpy as rp
crosstab, test_results, expected = rp.crosstab(X_car["Location"], y_test,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

print(test_results)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)




# # Impute & scale
X_train, X_test = impute_scale(X_train, X_test)
# One-hot encoding for categorical features
X_train, X_test = impute_one_hot_encode(X_train, X_test)

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

# 1 Call model here
models(X_train, X_test, y_train, y_test)

# Feature Selection

sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_train, y_train)

features = X_train.columns[(sel.get_support())]
print(len(features))
print(features)

X_train_up = X_train.filter(items=features)
X_test_up = X_test.filter(items=features)

# 2 Call model here
models(X_train_up, X_test_up, y_train, y_test)

sns.histplot(df['RainTomorrow'])
plt.show()

# # UPSAMPLING

local_df = df.copy(deep=True)

print(local_df)

# create two different dataframe of majority and minority class
df_majority = local_df[(local_df['RainTomorrow'] == "No")]
df_minority = local_df[(local_df['RainTomorrow'] == "Yes")]
# upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=110316,  # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

sns.histplot(df_upsampled['RainTomorrow'])
plt.show()

X = df_upsampled.drop('RainTomorrow', axis=1)
y = df_upsampled['RainTomorrow']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Impute & scale
X_train, X_test = impute_scale(X_train, X_test)
# One-hot encoding for categorical features
X_train, X_test = impute_one_hot_encode(X_train, X_test)

# 3 Call model here
models(X_train, X_test, y_train, y_test)

# # SMOTE

sm_df = df.copy(deep=True)

X = sm_df.drop('RainTomorrow', axis=1)
y = sm_df['RainTomorrow']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Impute & scale
X_train, X_test = impute_scale(X_train, X_test)
# One-hot encoding for categorical features
X_train, X_test = impute_one_hot_encode(X_train, X_test)

sm = SMOTE(sampling_strategy='minority', random_state=42)

# Fit and resample the data
X_train, y_train = sm.fit_resample(X_train, y_train)
X_test, y_test = sm.fit_resample(X_test, y_test)

# 4 Call model here
models(X_train, X_test, y_train, y_test)

