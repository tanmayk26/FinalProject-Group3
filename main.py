import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to display all the columns
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# Importing the data
url = 'weatherAUS.csv'
df = pd.read_csv(url)

# snapshot of the dataset
print(df)

# Datatype information
print(df.info())

# Data description
print(df.describe())

# To check null values in our dataset
print(df.isnull().sum())


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

# Plot discrete data
n_rows = (len(disc_cols) // 3) + (len(disc_cols) % 3 > 0) # Calculate number of subplot rows
plt.figure(figsize=(12, n_rows*4))
for i, col in enumerate(disc_cols):
    plt.subplot(n_rows, 3, i+1)
    sns.histplot(x=col, data=df, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()