import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Assuming 'df_cleaned' is your cleaned DataFrame
# Step 10: Visualize the correlation matrix
corr_matrix = df_cleaned.corr().round(2)

# Define the list of colors for your custom colormap
colors = ["red", "orange", "yellow", "lightgreen", "darkgreen"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot the heatmap
plt.figure(figsize=(12, 10))  # Adjust the figure size
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    vmin=-1,
    vmax=1,
    center=0,
    cmap=custom_cmap,
    linewidths=0.5,
    mask=mask,
    annot_kws={"size": 10},
)

# Highlight the first column
for i in range(len(corr_matrix)):
    ax.add_patch(plt.Rectangle((0, i), 1, 1, fill=False, edgecolor='black', lw=3))  # Adds a thick border around the first column

# Optional: Change the font size for the first column to make it bolder
for text in ax.texts:
    if text.get_position()[0] < 1:  # If the text is in the first column
        text.set_size(10)  # Make the text size bigger
        text.set_weight('bold')  # Make the font bold

# Display the heatmap
plt.show()
