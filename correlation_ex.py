import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2

data = load_iris()

df = pd.DataFrame(data.data)

# print first 5 rows
print(df.head(5))

# add column target
print(data.target)
df["target"] = data.target
print(df.head(5))

# checking the correlation
corr = df.corr(method='pearson')["target"]
print(corr)
"""
correlation between single feature and target
0 = sepal length (cm) =         0.782561
1 = sepal width (cm) =          -0.426658
2 = petal length (cm) =         0.949035
3 = petal width (cm) =          0.956547
target    1.000000
"""

sns.heatmap(df.corr(method='pearson'), annot=True)
# plt.show()

"""
feature selection with Chi-Square Test

il test di indipendenza determina se esiste una relazione significativa
tra due variabili.

test = Summation (Observed Value - Expected Value) / Expected Value
somma(valore osservato - valore atteso) / valore atteso
se il p-value è inferire a 0.05, allora rifiutiamo l'ipotesi nulla
e andiamo con l'ipotesi alternativa
"""
X = df.drop("target", axis=1)
print(X)
y = df.target
print(y)

model = SelectKBest(chi2, k=2)
new = model.fit(X, y)
X_new = new.transform(X)
print(X_new)

cols = new.get_support(indices=True)
features_df_new = X.iloc[:, cols]
print(features_df_new)