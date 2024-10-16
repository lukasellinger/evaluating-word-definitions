"""Create svg for normal vs split claims."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Example classification metrics data with added empty columns
data = {
    'Metric': ['Precision (Positive)', 'Recall (Positive)',
               'Precision (Negative)', 'Recall (Negative)',
               'Accuracy'],
    'Pipeline': [87.88, 82.86, 83.56, 88.41, 85.61],
    'GPT-3.5 Turbo': [85.71, 94.29, 93.55, 84.06, 89.21],
    'GPT-4o mini':   [90.91, 85.71, 86.30, 91.30, 88.49],
    'GPT-4o':        [100, 78.57, 82.14, 100, 89.21],
    'Pipeline Splitted': [95.74, 64.29, 72.83, 97.10, 80.58],
    'GPT-3.5 Turbo Splitted': [93.10, 77.14, 80.25, 94.20, 85.61],
    'GPT-4o mini Splitted':   [95.65, 62.86, 72.04, 97.10, 79.86],
    'GPT-4o Splitted':        [97.22, 50.00, 66.02, 98.55, 74.10],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the 'Metric' column as the index
df.set_index('Metric', inplace=True)

# Split the DataFrame into two: one for the original models and one for the 'Splitted' models
df_original = df[['Pipeline', 'GPT-3.5 Turbo', 'GPT-4o mini', 'GPT-4o']]
df_splitted = df[['Pipeline Splitted', 'GPT-3.5 Turbo Splitted',
                  'GPT-4o mini Splitted', 'GPT-4o Splitted']]

# Rename columns of df_splitted to remove "Splitted"
df_splitted.columns = ['Pipeline', 'GPT-3.5 Turbo', 'GPT-4o mini', 'GPT-4o']

# Get the minimum and maximum values across both DataFrames to ensure consistent color coding
vmin = min(df_original.min().min(), df_splitted.min().min())
vmax = max(df_original.max().max(), df_splitted.max().max())

# Set even larger font sizes
FONT_SCALE = 2.5  # Further increase the font scale
sns.set(font_scale=FONT_SCALE)

# Plot the original models heatmap
plt.figure(figsize=(16, 10))  # Increased figure size for better readability
sns.heatmap(df_original, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5,
            cbar=False,  # Remove the color bar
            annot_kws={"size": 36},  # Increase annotation font size
            xticklabels=df_original.columns, yticklabels=df_original.index,
            vmin=vmin, vmax=vmax)

# Rotate the x-axis labels
plt.title('No Fact-splitting', fontsize=40)  # Further increase title font size
plt.xticks(rotation=45, ha='right', fontsize=36)  # Increase x-tick font size
plt.yticks(fontsize=36)  # Increase y-tick font size

# Remove y-axis label and x-axis label
plt.ylabel('', fontsize=24)
plt.xlabel('', fontsize=24)

# Save as SVG
plt.tight_layout()
plt.savefig("original_models_heatmap.svg", format='svg')
plt.show()

# Plot the 'Splitted' models heatmap without y-axis labels and renamed columns
plt.figure(figsize=(16, 10))  # Increased figure size for better readability
sns.heatmap(df_splitted, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5,
            cbar_kws={"shrink": .8}, annot_kws={"size": 36},  # Increase annotation font size
            xticklabels=df_splitted.columns, yticklabels=False,  # Remove y-axis labels
            vmin=vmin, vmax=vmax)

# Rotate the x-axis labels
plt.title('Fact-splitting', fontsize=40)  # Further increase title font size
plt.xticks(rotation=45, ha='right', fontsize=36)  # Increase x-tick font size

# Remove y-axis label and x-axis label
plt.ylabel('', fontsize=24)
plt.xlabel('', fontsize=24)

# Save as SVG
plt.tight_layout()
plt.savefig("splitted_models_heatmap.svg", format='svg')
plt.show()
