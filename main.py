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