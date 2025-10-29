import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix data
confusion_matrix = np.array([
    [177, 0, 23, 0, 0],
    [3, 195, 2, 0, 0],
    [5, 0, 190, 0, 5],
    [6, 0, 0, 194, 0],
    [12, 1, 11, 11, 165]
])

# Custom labels
labels = ['冰雹侵害', '健康叶片', '棉铃虫害', '根系损伤', '强光灼伤']

# Set up plot
plt.figure(figsize=(10, 7))
ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Scale'},
                 xticklabels=labels, yticklabels=labels, annot_kws={"size": 18})  # Set font size for numbers

# Add title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Optimize label display
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Show plot
plt.show()
