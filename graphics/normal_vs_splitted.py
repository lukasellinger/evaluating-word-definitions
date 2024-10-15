"""Create svg for normal vs split claims."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Example classification metrics data with added empty columns
data = {
    'Metric': ['Precision (Positive)', 'Recall (Positive)',
               'Precision (Negative)', 'Recall (Negative)',
               'Accuracy'],
    'Pipeline': [0.9008, 0.7002, 0.9474, 0.4286, 0.7024],
    'GPT-3.5 Turbo': [0.6308, 0.9762, 0.9474, 0.4286, 0.7024],
    'GPT-4o mini':   [0.7547, 0.9524, 0.9355, 0.6905, 0.8214],
    'GPT-4o':        [0.9080, 0.9405, 0.9383, 0.9048, 0.9226],
    'Pipeline Splitted': [0.9500, 0.6500, 0.9474, 0.4286, 0.7024],
    'GPT-3.5 Turbo Splitted': [0.8539, 0.9048, 0.8947, 0.8452, 0.8750],
    'GPT-4o mini Splitted':   [0.9189, 0.8095, 0.8298, 0.9286, 0.8608],
    'GPT-4o Splitted':        [0.9315, 0.8095, 0.8316, 0.9405, 0.8750],
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
sns.heatmap(df_original, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5,
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
sns.heatmap(df_splitted, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5,
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
