import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
df = sns.load_dataset("iris")

print(df.columns)

# 1. Histogram
sns.histplot(df["petal_length"])
sns.histplot(df["sepal_length"])
plt.title("Petal Length and Sepal Length Histogram")
plt.show()

# 2. Box Plot
sns.boxplot(data=df[["sepal_length", "sepal_width",
                     "petal_length", "petal_width"]])
plt.title("Box Plot of All Iris Features")
plt.show()

# 3. Scatter Plot
sns.scatterplot(data=df, x="sepal_length",
                y="petal_length", hue="species")
plt.title("Sepal Length and Petal Length by Species")
plt.show()

# 4. Pair Plot
sns.pairplot(df, hue="species")
plt.show()

# 5. Correlation
correlation = df[["sepal_length", "sepal_width",
                   "petal_length", "petal_width"]].corr()

sns.heatmap(correlation, annot=True)
plt.title("Correlation of Iris Features")
plt.show()

# 6. Subplots - Histograms
plt.subplot(1, 2, 1)
plt.hist(df["petal_length"], bins=10)
plt.title("Petal Length")

plt.subplot(1, 2, 2)
plt.hist(df["sepal_length"], bins=10)
plt.title("Sepal Length")

plt.show()

# 7. Pandas Scatter Matrix
pd.plotting.scatter_matrix(
    df[["sepal_length", "sepal_width",
        "petal_length", "petal_width"]]
)
plt.show()

# 8. Correlation Matrix using matshow
plt.matshow(correlation)
plt.xticks(range(4), correlation.columns)
plt.yticks(range(4), correlation.columns)
plt.title("Correlation of Iris Features")
plt.show()