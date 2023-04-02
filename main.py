import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the data
df = pd.read_csv('weatherAUS.csv')

# snapshot of the dataset
print(df)

# Datatype information
print(df.info())

# Data description
print(df.describe())
