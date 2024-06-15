import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("customer_churn_prediction.csv")

def plot_heatmap(df, drop_columns=[], figsize=(20, 10), image_name='heatmap.png'):
    plt.subplots(figsize=figsize)
    sns.heatmap(df.drop(columns=drop_columns).corr(), annot=True, fmt=".3f", cmap="viridis")
    plt.savefig(image_name)
    plt.show()

def plot_feature_correlation(df, target_column):
    corr = df.corr()[target_column]
    plt.figure(figsize=(18, 10))
    sns.barplot(x=corr[:-1].index, y=corr[:-1])
    plt.title("Correlation of Features to " + target_column.capitalize())
    plt.xticks(rotation=90)
    plt.savefig('corr.png')
    plt.show()

def plot_boxplots(df, columns):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)

    for i, col in enumerate(columns):
        sns.boxplot(x="churn", y=col, data=df, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(col.capitalize() + " by Churn")

    plt.tight_layout()
    plt.savefig('boxes.png')
    plt.show()

def plot_distribution(df, columns):
    for col in columns:
        g = sns.distplot(df[col], color="b", label="Skewness: %.2f" % (df[col].skew()))
        g.legend(loc="best")
        plt.title("Distribution of " + col.capitalize())
        plt.savefig('distribuations.png')
        plt.show()


columns_to_plot = ["credit_score", "age", "balance", "estimated_salary"]
plot_distribution(df, columns_to_plot)

columns_to_plot = ["credit_score", "balance", "age", "estimated_salary"]
plot_boxplots(df, columns_to_plot)

plot_feature_correlation(df, 'churn')

plot_heatmap(df, drop_columns=['churn'], save_image=True)