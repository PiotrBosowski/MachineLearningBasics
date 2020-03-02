# DATA PREPARATION

# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# LOADING THE DATASET
dataset = pd.read_csv(R'C:\Users\Peter\Desktop\MachineLearningInPyCharm\dataPreparingTemplate\Data.csv')
x = dataset.iloc[:, :-1].values  # allRows, <0, lastColumn) independent variables
y = dataset.iloc[:, -1].values   # allRows, lastColumn        dependent variable

# IMPUTING MISSING DATA
from sklearn.impute import SimpleImputer
x[:, 1:3] = SimpleImputer().fit_transform(x[:, 1:3])  # imputing missing ('nan' by default) values by the mean (default) of the column

# ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
x[:, 0] = LabelEncoder().fit_transform(x[:, 0])  # encoding categorical values into numbers, eg. France = 0, Germany = 1 etc.
x = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])], remainder='passthrough').fit_transform(x)  # splitting encoded categorical variable column into N columns with boolean values
x = x[:, 1:5]  # reducing the number of boolean columns, because we only need N-1 columns to describe N states
y = LabelEncoder().fit_transform(y)  # encoding categorical variable with numeric value

# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)  # random_state = randomness seed

# Standarization/normalization - transforming values from different columns to fit in the same range (f.e. -1,1)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()              # creating StandardScaler
xTrain = scaler.fit_transform(xTrain)  # fitting it with xTrain
xTest = scaler.transform(xTest)        # applying same scaler on xTest
