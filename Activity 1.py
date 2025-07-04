import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Titanic Dataset.csv")

print(data.head(5))
print(data.info)

sns.countplot(data["Gender"], hue=data["Survived"])
sns.countplot(data["PClass"], hue=data["Survived"])

sns.displot(data["Age"], kde=False, bins=40)
sns.countplot(data["Gender"])

sns.countplot(x="Survived", hue="SibSp", data=data, palette="mako")
sns.countplot(x="Survived", hue="Parch", data=data, palette="mako")

sns.displot(data["Fare"])
plt.show()

sns.boxplot(data=data, x="Pclass", y="Age", palette="winter")
sns.heatmap(data.corr())