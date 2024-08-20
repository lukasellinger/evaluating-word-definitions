import matplotlib.pyplot as plt
import numpy as np

from fetchers.wiktionary_parser import WiktionaryParser

# Data
models = ['Model A', 'Model B', 'Model C']
german_words_dataset1 = [95.0, 93.8, 94.6]
translated_english_dataset1 = [92.5, 91.4, 92.1]
german_words_dataset2 = [94.0, 92.9, 93.2]
translated_english_dataset2 = [91.7, 90.5, 91.0]

# Bar width
bar_width = 0.2
index = np.arange(len(models))

# Create the bar chart
fig, ax = plt.subplots()

# Dataset 1 bars
bars1 = ax.bar(index - bar_width, german_words_dataset1, bar_width, label='German Words (Dataset 1)')
bars2 = ax.bar(index, translated_english_dataset1, bar_width, label='Translated English (Dataset 1)')

# Dataset 2 bars
bars3 = ax.bar(index + bar_width, german_words_dataset2, bar_width, label='German Words (Dataset 2)')
bars4 = ax.bar(index + 2*bar_width, translated_english_dataset2, bar_width, label='Translated English (Dataset 2)')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Ablation Study: Accuracy Using Translated English vs. Original German Words')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

# Display the bar chart
plt.tight_layout()
plt.show()

WiktionaryParser().get_wiktionary_glosses('asdf', 'asdf')