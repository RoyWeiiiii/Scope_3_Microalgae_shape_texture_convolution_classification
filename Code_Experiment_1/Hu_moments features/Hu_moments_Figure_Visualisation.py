##################### Hu_moments Figure Visualisation #########################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("Input file .csv")

########################################### For Density curve visualisation ######################################

# Define the column names for the features
feature_cols = ['Hu_1', 'Hu_2', 'Hu_3', 'Hu_4', 'Hu_5', 'Hu_6', 'Hu_7']

# Define the color palette for the class labels
colors = {'Chlamydomonas_Reinhardtii': 'blue', 'Chlorella_FSP': 'red', 'Spirulina_Platensis': 'green'}

# Create a 3x3 matrix of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 10), dpi=100, constrained_layout=True)

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Loop through each feature column and plot the histogram and density curve
for i, col in enumerate(feature_cols):
    sns.histplot(data=df, x=col, hue='Class', kde=True, ax=axes[i], palette=colors, alpha=0.5)
    axes[i].set_title(col, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='both', labelsize=10)

    if i % 3 == 0:
        axes[i].set_ylabel('Count', fontsize=12)
    else:
        axes[i].set_ylabel('')

# Remove any unused subplots
for i in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[i])

# Adjust the layout of the subplots to prevent overlapping
plt.tight_layout()

# Show the plot
plt.savefig("Save image plot as .png")
plt.show()

# Loop through each feature and plot the distribution for each class label
for feature in feature_cols:
    plt.figure(figsize=(8, 5))
    for label, color in colors.items():
        sns.distplot(df.loc[df['Class'] == label, feature], hist=True, kde=True, kde_kws={'bw': 0.2}, color=color, label=label)
    plt.xlabel(feature)
    plt.legend()
plt.show()

################################# Generate scatter plots for each pair of morphological features (Individual scatter plots) #####################

for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        fig, ax = plt.subplots()
        for class_name, group in df.groupby('Class'):
            ax.scatter(group[feature_cols[i]], group[feature_cols[j]], label=class_name, color=colors[class_name], alpha=0.5, s=10)
        ax.legend()
        ax.set_xlabel(feature_cols[i])
        ax.set_ylabel(feature_cols[j])
        plt.tight_layout()
        plt.savefig(f"Save scatter plots as {feature_cols[i]}_{feature_cols[j]}.png")
        plt.show()

################################### Create a heatmap of the correlation matrix #############################

corr = df.iloc[:, :-1].corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, xticklabels=corr.columns, yticklabels=corr.columns)

# Rotate x-axis tick labels by 45 degrees
plt.xticks(rotation=45)
plt.tight_layout()

# plt.title('Correlation Matrix Heatmap')
plt.savefig("Save Correlation Matrix Heatmapp file as .png")
plt.show()