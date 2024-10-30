import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

# Configuration Parameters
#------------------------
threshold = 33  # Threshold for dropping columns with missing values
vif_drop = 30    # Threshold for VIF analysis
csv_file_path = r"C:\Users\boulb\OneDrive\Documents\A002 - CS50X\pydata\data\testdata5.csv"

# Data Loading and Initial Analysis
#--------------------------------
def load_and_examine_data(file_path):
    """Load data and perform initial examination"""
    df = pd.read_csv(file_path)
    print("=== Initial Data Information ===")
    print(df.info())
    return df

# Data Cleaning Functions
#----------------------
def handle_missing_values(df, threshold):
    """Handle missing values in the dataset"""
    # Calculate missing value percentages
    missing_percentage = df.isnull().mean() * 100
    print("\n=== Missing Values Analysis ===")
    print("Missing values percentage per column:\n", missing_percentage)
    
    # Drop columns with high missing values
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Drop rows with any missing values
    initial_row_count = df_cleaned.shape[0]
    df_cleaned = df_cleaned.dropna()
    final_row_count = df_cleaned.shape[0]
    
    print(f"\nRows analysis:")
    print(f"Initial rows: {initial_row_count}")
    print(f"Final rows: {final_row_count}")
    print(f"Rows dropped: {initial_row_count - final_row_count}")
    
    return df_cleaned

def prepare_numeric_data(df):
    """Ensure all columns are numeric"""
    print("\n=== Data Types Analysis ===")
    print("Data types:\n", df.dtypes)
    return df.select_dtypes(include=[np.number])

# Multicollinearity Analysis
#-------------------------
def perform_vif_analysis(df, target_col, vif_threshold):
    """Perform VIF analysis and remove highly correlated features"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    X = sm.add_constant(X)
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\n=== VIF Analysis ===")
    print(vif_data.sort_values('VIF', ascending=False))
    
    # Remove high VIF features
    high_vif_cols = vif_data[vif_data["VIF"] > vif_threshold]["feature"]
    X = X.drop(columns=high_vif_cols)
    print(f"\nColumns dropped due to high multicollinearity: {high_vif_cols.tolist()}")
    
    return X, y

# Statistical Analysis
#-------------------
def perform_regression(X, y):
    """Perform multiple linear regression"""
    model = sm.OLS(y, X).fit()
    print("\n=== Regression Analysis ===")
    print(model.summary())
    return model

# Visualization Functions
#----------------------
def plot_correlation_heatmap(df):
    """Create correlation matrix heatmap with improved colors"""
    corr_matrix = df.corr().round(2)
    colors = ["#FF9999", "#FFCC99", "#FFFF99", "#99FF99", "#99CCFF"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, center=0,
                cmap=custom_cmap, linewidths=0.5, mask=mask,
                annot_kws={"size": 10})
    plt.title("Correlation Heatmap", pad=20)
    plt.show()

def plot_residuals(model, y):
    """Plot residuals analysis with enhanced colors"""
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Residuals vs Fitted
    axes[0,0].scatter(fitted_values, residuals, c=colors[0], alpha=0.6)
    axes[0,0].axhline(y=0, color='#FF9F1C', linestyle='--')
    axes[0,0].set_xlabel('Fitted values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].get_lines()[0].set_markerfacecolor(colors[1])
    axes[0,1].get_lines()[0].set_alpha(0.6)
    axes[0,1].set_title('Normal Q-Q Plot')
    
    # Histogram of residuals
    axes[1,0].hist(residuals, bins=30, color=colors[2], alpha=0.7)
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_title('Histogram of Residuals')
    
    # Scale-Location plot
    axes[1,1].scatter(fitted_values, np.sqrt(np.abs(residuals)), 
                     c=colors[3], alpha=0.6)
    axes[1,1].set_xlabel('Fitted values')
    axes[1,1].set_ylabel('√|Residuals|')
    axes[1,1].set_title('Scale-Location Plot')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X):
    """Plot feature importance with enhanced visualization"""
    # Create a copy of X without 'const' if it exists
    X_importance = X.copy()
    if 'const' in X_importance.columns:
        X_importance = X_importance.drop('const', axis=1)
        coefficients = pd.DataFrame({
            'Feature': X_importance.columns,
            'Coefficient': abs(model.params[1:])  # Skip the constant term
        })
    else:
        coefficients = pd.DataFrame({
            'Feature': X_importance.columns,
            'Coefficient': abs(model.params)
        })
    
    coefficients = coefficients.sort_values('Coefficient', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(coefficients)))
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(coefficients['Feature'], coefficients['Coefficient'], 
                   color=colors)
    plt.title('Feature Importance (Absolute Regression Coefficients)')
    plt.xlabel('|Coefficient Value|')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def perform_random_forest_analysis(X, y):
    """Perform Random Forest analysis for feature importance"""
    # Remove 'const' column if it exists
    X_rf = X.copy()
    if 'const' in X_rf.columns:
        X_rf = X_rf.drop('const', axis=1)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_rf, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X_rf.columns,
        'Importance': rf_model.feature_importances_
    })
    importance = importance.sort_values('Importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(importance)))
    bars = plt.barh(importance['Feature'], importance['Importance'], 
                   color=colors)
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance Score')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return importance

def perform_cross_validation_analysis(X, y):
    """Perform cross-validation analysis for each feature"""
    # Remove 'const' column if it exists
    X_cv = X.copy()
    if 'const' in X_cv.columns:
        X_cv = X_cv.drop('const', axis=1)
    
    results = []
    scaler = StandardScaler()
    
    for column in X_cv.columns:
        X_scaled = scaler.fit_transform(X_cv[[column]])
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X_scaled, y, cv=5, scoring='r2'
        )
        results.append({
            'Feature': column,
            'Mean R2': scores.mean(),
            'Std R2': scores.std()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Mean R2', ascending=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(results_df)))
    bars = plt.barh(results_df['Feature'], results_df['Mean R2'], 
                   xerr=results_df['Std R2'], color=colors)
    plt.title('Feature Individual Predictive Power (Cross-Validation R² Score)')
    plt.xlabel('R² Score')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def calculate_feature_rankings(X, y, model):
    """Calculate and combine different feature importance metrics"""
    # Remove 'const' column if it exists
    X_rank = X.copy()
    if 'const' in X_rank.columns:
        X_rank = X_rank.drop('const', axis=1)
    
    # Get linear regression coefficients (excluding constant)
    coef_importance = pd.DataFrame({
        'Feature': X_rank.columns,
        'Linear_Coefficient': abs(model.params[1:])  # Skip the constant term
    })
    
    # Get Random Forest importance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_rank, y)
    rf_importance = pd.DataFrame({
        'Feature': X_rank.columns,
        'RF_Importance': rf_model.feature_importances_
    })
    
    # Get correlation with target
    correlations = abs(X_rank.corrwith(y))
    corr_importance = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    
    # Combine all metrics
    rankings = pd.merge(coef_importance, rf_importance, on='Feature')
    rankings = pd.merge(rankings, corr_importance, on='Feature')
    
    # Calculate average ranking
    for column in ['Linear_Coefficient', 'RF_Importance', 'Correlation']:
        rankings[f'{column}_Rank'] = rankings[column].rank()
    
    rankings['Average_Rank'] = rankings[[
        'Linear_Coefficient_Rank', 
        'RF_Importance_Rank', 
        'Correlation_Rank'
    ]].mean(axis=1)
    
    rankings = rankings.sort_values('Average_Rank')
    
    # Plot combined rankings
    plt.figure(figsize=(12, 8))
    
    # Create a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(rankings)))
    
    x = np.arange(len(rankings))
    width = 0.25
    
    plt.barh(x - width, rankings['Linear_Coefficient_Rank'], 
            width, label='Linear Regression', color=colors, alpha=0.6)
    plt.barh(x, rankings['RF_Importance_Rank'], 
            width, label='Random Forest', color=colors, alpha=0.8)
    plt.barh(x + width, rankings['Correlation_Rank'], 
            width, label='Correlation', color=colors)
    
    plt.yticks(x, rankings['Feature'])
    plt.xlabel('Rank (lower is more important)')
    plt.title('Feature Importance Rankings by Different Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return rankings

# Main Execution
#-------------
def main():
    # Load and clean data and print information
    df = load_and_examine_data(csv_file_path)
    df_cleaned = handle_missing_values(df, threshold)
    df_cleaned = prepare_numeric_data(df_cleaned)
    
    print("\n=== Missing Values Info ===")
    miss_V = perform_vif_analysis(df, target_col, vif_threshold)
    print(miss_V)
    
    
    # Perform analysis
    X, y = perform_vif_analysis(df_cleaned, "Intrinsic Engagement", vif_drop)
    model = perform_regression(X, y)
    
    # Create visualizations
    plot_correlation_heatmap(df_cleaned)
    plot_residuals(model, y)
    plot_feature_importance(model, X)
    
    # Perform additional analyses
    print("\n=== Random Forest Feature Importance ===")
    rf_importance = perform_random_forest_analysis(X, y)
    print(rf_importance)
    
    print("\n=== Cross-Validation Analysis ===")
    cv_results = perform_cross_validation_analysis(X, y)
    print(cv_results)
    
    print("\n=== Combined Feature Rankings ===")
    rankings = calculate_feature_rankings(X, y, model)
    print("\nFinal Feature Rankings (lower rank = more important):")
    print(rankings[['Feature', 'Average_Rank']])
    
if __name__ == "__main__":
    main()
