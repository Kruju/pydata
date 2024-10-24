import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

    """
Carefull with this, if vif drop too low you remove the intrinsic engagement
threshold of this value affectgs the percentage of missing values that we will drop per collum
    """
*/ 
threshold = 33
vif_drop = 5

# Step 1: Load the CSV file
csv_file_path = (
    r"C:\Users\boulb\OneDrive\Documents\A002 - CS50X\pydata\data\testdata5.csv"
)
df = pd.read_csv(csv_file_path)

# Step 2: Get a summary of the data
print(df.info())  # Check the structure of the data

# Step 3: Calculate the percentage of missing values for each column
missing_percentage = df.isnull().mean() * 100
print("Missing values percentage per column:\n", missing_percentage)

# Step 4: Drop columns with more than a threshold (20%) of missing values

columns_to_drop = missing_percentage[missing_percentage > threshold].index
df_cleaned = df.drop(columns=columns_to_drop)

# Step 5: Drop rows with missing values
initial_row_count = df.shape[0]
df_cleaned = df_cleaned.dropna()
final_row_count = df_cleaned.shape[0]
rows_dropped = initial_row_count - final_row_count
print(
    f"Initial rows: {initial_row_count}, Final rows: {final_row_count}, Rows dropped: {rows_dropped}"
)

# Step 6: Ensure all columns are numeric
print("Data types after dropping columns and rows:\n", df_cleaned.dtypes)
df_cleaned = df_cleaned.select_dtypes(include=[np.number])  # Keep only numeric columns

# Step 7: Check multicollinearity with Variance Inflation Factor (VIF)
X = df_cleaned.drop(
    "Intrinsic Engagement", axis=1
)  # Assuming 'Engagement' is the dependent variable
y = df_cleaned["Intrinsic Engagement"]

# Replace infinite values with NaN and fill missing values with the mean
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

# Add constant (intercept) to the model for statsmodels
X = sm.add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]

print("VIF values:\n", vif_data)

# Step 8: Drop columns with high VIF (indicating multicollinearity)
# You might need to manually decide based on VIF values; here, let's assume you drop those above 10
high_vif_cols = vif_data[vif_data["VIF"] > vif_drop]["feature"]
X = X.drop(columns=high_vif_cols)

print(f"Columns dropped due to high multicollinearity: {high_vif_cols.tolist()}")

# Step 9: Perform Multiple Linear Regression
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

# Step 10: Visualize the correlation matrix
corr_matrix = df_cleaned.corr().round(2)

# Define the list of colors for your custom colormap
colors = ["red", "orange", "yellow", "lightgreen", "darkgreen"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Create a mask for the upper triangle (optional)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Create a heatmap with the custom colormap
plt.figure(figsize=(12, 10))  # Adjust the figure size
sns.heatmap(
    corr_matrix,
    annot=True,
    vmin=-1,
    vmax=1,
    center=0,
    cmap=custom_cmap,
    linewidths=0.5,
    mask=mask,
    annot_kws={"size": 10},
)  # You can adjust the font size

# Display the heatmap
plt.show()

# Print the regression summary
print(model.summary())
