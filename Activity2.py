# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('Titanic.csv')

# Initial data inspection
print("First 5 rows:")
print(data.head(5))
print("\nNull values:")
print(data.isnull().sum())
print("\nDescriptive statistics:")
print(data.describe())

# Define relevant numerical features for Titanic analysis
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']

# Boxplot analysis for numerical features
print("\nBoxplot Analysis for Numerical Features:")
for feature in numerical_features:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

# Countplot analysis for categorical features
print("\nCountplot Analysis for Categorical Features:")
for feature in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(x=data[feature])
    plt.title(f'Count of {feature}')
    plt.xticks(rotation=45)
    plt.show()

# Correlation analysis (numerical features only)
print("\nCorrelation Heatmap:")
plt.figure(figsize=(8,6))
sns.heatmap(data[numerical_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Numerical Features Correlation Matrix')
plt.show()

# Distribution and skewness analysis
print("\nDistribution and Skewness:")
for feature in numerical_features:
    plt.figure(figsize=(8,4))
    sns.histplot(data[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
    plt.show()
    print(f'Skewness of {feature}: {data[feature].skew():.4f}\n')

# Survival analysis by different features
print("\nSurvival Rate Analysis:")
for feature in ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']:
    plt.figure(figsize=(8,4))
    sns.barplot(x=feature, y='Survived', data=data)
    plt.title(f'Survival Rate by {feature}')
    plt.ylabel('Survival Rate')
    plt.show()