# %%
#%pip install matplotlib

#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression

# Directory paths
csv_directory = 'C:/Users/Nicolas/Downloads/TasaLingualyzedResults-2024-0502'
text_directory = 'C:/Users/Nicolas/Downloads/preprocessingTasa/TasaPostProcess'

# Headers for the CSV files as they don't contain any headers
headers = ['Linguistic Feature', 'Amount Present', 'Doctype', 'Statistical Category']

# Initialize an empty DataFrame to collect all data
full_data = pd.DataFrame()
print('Process Initialized')

# Function to safely read CSV with specified headers and different encodings
def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path, names=headers)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='latin1', names=headers)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='ISO-8859-1', names=headers)
        
# Get all CSV files
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')][:5000]

processed_count = 0

# Process each file
for filename in csv_files:
    file_path = os.path.join(csv_directory, filename)
    
    # Load the CSV file as a DataFrame using the safe_read_csv function
    df = safe_read_csv(file_path)
    
    # Extract the base identifier to match the text file
    identifier = filename.split('_DRP')[0]  # Splitting by '_DRP' to get the base identifier
    
    # Find the corresponding text file
    text_filename = [f for f in os.listdir(text_directory) if f.startswith(identifier)][0]
    
    # Extract DRP score from the text filename
    drp_score = float(text_filename.split('=')[-1].replace('.txt', ''))
    
    # Filter to only 'average' statistical category and reset index to remove gaps
    df = df[df['Statistical Category'] == 'average'].reset_index(drop=True)

    # Set the index to identifier for better organization
    df['Identifier'] = identifier

    # Add a column for the DRP score
    df['DRP_Score'] = drp_score
    
    # Pivot the data so each linguistic feature becomes a column
    pivot_df = df.pivot(index='Identifier', columns='Linguistic Feature', values='Amount Present')
    pivot_df.reset_index(inplace=True)  # Reset index to turn 'Identifier' back to a column
    # Add DRP score to the pivoted dataframe
    pivot_df['DRP_Score'] = drp_score
    
    # Append to the full DataFrame
    full_data = pd.concat([full_data, pivot_df], ignore_index=True)

    # Increment and check the counter (purely for diagnostics)
    processed_count += 1
    if processed_count % 500 == 0:
        print(f"Processed {processed_count} files")

# Now full_data contains all the information
print(full_data.head())
print('Process Terminated')

# Specify the path where you want to save the CSV file
output_csv_path = 'C:/Users/Nicolas/Downloads/full_data.csv'

# Save the DataFrame to a CSV file
full_data.to_csv(output_csv_path, index=False)

print("DataFrame saved successfully to", output_csv_path)

# %%
# Beginning with Exploratory Data Analysis

# Display the summary statistics for the DataFrame
#print(full_data.describe())

# %%
# Heatmap of the correlation matrix
# Assuming 'full_data' is your DataFrame and you need to plot histograms
numeric_data = full_data.select_dtypes(include=['float64', 'int64'])  # Only select numeric columns
numeric_data.hist(figsize=(20, 15))
plt.show()

# %%
# Drop missing values for simplicity
full_data.dropna(inplace=True)

# Or fill missing values
# full_data.fillna(full_data.mean(), inplace=True)

# Splitting data for training and testing
X = full_data.drop(['DRP_Score', 'Identifier'], axis=1)
y = full_data['DRP_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Function to evaluate and plot model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse}, R² Score: {r2}")
    return mse, r2

# %%
# Linear Regression Model
lr = LinearRegression()
lr_mse, lr_r2 = evaluate_model(lr, X_train, X_test, y_train, y_test, "Linear Regression")

# %%
# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_mse, rf_r2 = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest Regressor")

# %%
# Feature importance for Random Forest
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances in Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# %%
# Gradient Boosting Regressor
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_mse, gb_r2 = evaluate_model(gb, X_train, X_test, y_train, y_test, "Gradient Boosting Regressor")

# %%
# Lasso Regression
lasso = Lasso(alpha=0.01)
lasso_mse, lasso_r2 = evaluate_model(lasso, X_train, X_test, y_train, y_test, "Lasso Regression")

# %%
# PCA + Linear Regression
pca = PCA()
lm = LinearRegression()
pipeline = make_pipeline(pca, lm)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pca_mse = mean_squared_error(y_test, y_pred)
pca_r2 = r2_score(y_test, y_pred)
print(f"PCA + Linear Regression - Mean Squared Error: {pca_mse}, R² Score: {pca_r2}")

# %%
# PLS Regression
pls = PLSRegression(n_components=5)
pls_mse, pls_r2 = evaluate_model(pls, X_train, X_test, y_train, y_test, "PLS Regression")

# %%
# SVM with linear kernel
svm_linear = SVR(kernel='linear')
svm_linear_mse, svm_linear_r2 = evaluate_model(svm_linear, X_train_scaled, X_test_scaled, y_train, y_test, "SVM Linear Kernel")

# %%
# SVM with RBF kernel
svm_rbf = SVR(kernel='rbf')
svm_rbf_mse, svm_rbf_r2 = evaluate_model(svm_rbf, X_train_scaled, X_test_scaled, y_train, y_test, "SVM RBF Kernel")

# %%
# Summarizing model performance
performance_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'Lasso Regression', 'PCA + Linear Regression', 'PLS Regression', 'SVM Linear Kernel', 'SVM RBF Kernel'],
    'MSE': [lr_mse, rf_mse, gb_mse, lasso_mse, pca_mse, pls_mse, svm_linear_mse, svm_rbf_mse],
    'R² Score': [lr_r2, rf_r2, gb_r2, lasso_r2, pca_r2, pls_r2, svm_linear_r2, svm_rbf_r2]
})

print(performance_df)

# Plotting the performance of models
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='MSE', data=performance_df)
plt.title('Model Performance (Mean Squared Error)')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='R² Score', data=performance_df)
plt.title('Model Performance (R² Score)')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.show()

