# Import required libraries

# Calling different modules
from models import models
from tools import impute_scale, impute_one_hot_encode

# Data Manipulation & Visualization libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Feature Selection
from sklearn.ensemble import RandomForestClassifier

# import warnings
# warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('weatherAUS.csv')

# Drop columns that are not required
df = df.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'], axis=1)
df.info()
# Remove rows with missing values in the target variable
df = df.dropna(subset=['RainTomorrow'])

# Split the dataset into features and target variable
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

sns.histplot(df['RainTomorrow'])
plt.show()


# Plot discrete data
n_rows = (len(disc_cols) // 3) + (len(disc_cols) % 3 > 0) # Calculate number of subplot rows
plt.figure(figsize=(12, n_rows*4))
for i, col in enumerate(disc_cols):
    plt.subplot(n_rows, 3, i+1)
    sns.histplot(x=col, data=df, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

print(X.head())
print(y.head())

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

