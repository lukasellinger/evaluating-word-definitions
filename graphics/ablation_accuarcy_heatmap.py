"""Create svg for ablation accuracy heatmap."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Example accuracy data for different combinations of documents and sentences
data = {
    '1 Document': [71.22, 82.73, 82.01, 82.73, 83.45],
    '2 Documents': [74.82, 86.33, 84.89, 85.61, 86.33],
    '3 Documents': [74.82, 87.05, 85.61, 85.61, 87.05],
    '4 Documents': [74.10, 86.33, 85.61, 85.61, 87.05],
    '5 Documents': [74.10, 86.33, 85.61, 85.61, 87.05],
}

# Convert to DataFrame
df = pd.DataFrame(data, index=['1 Sentence', '2 Sentences', '3 Sentences',
                               '4 Sentences', '5 Sentences'])

# Set the minimum and maximum values for consistent color coding
vmin = df.min().min()
vmax = df.max().max()

# Set font scale and plot the heatmap
FONT_SCALE = 2.5
sns.set(font_scale=FONT_SCALE)

plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5,
            cbar_kws={"shrink": .8}, annot_kws={"size": 36}, vmin=vmin, vmax=vmax)

plt.xticks(rotation=45, ha='right', fontsize=36)
plt.yticks(fontsize=36)
plt.ylabel('', fontsize=24)
plt.xlabel('', fontsize=24)

# Save as SVG
plt.tight_layout()
plt.savefig("accuracy_heatmap.svg", format='svg')
plt.show()
