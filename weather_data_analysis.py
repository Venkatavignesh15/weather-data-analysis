import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv('weather.csv')

# Step 2: Handle Missing Values
print("Missing values before cleaning:\n", df.isnull().sum())
df.dropna(inplace=True)  # Remove missing values
print("Missing values after cleaning:\n", df.isnull().sum())

# Step 3: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Data Visualization - Pair Plot
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 5: Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Step 6: Histogram of Rainfall
plt.hist(df['Rainfall'], bins=30, color='blue', alpha=0.7)
plt.xlabel('Rainfall')
plt.ylabel('Frequency')
plt.title('Rainfall Distribution')
plt.show()

# Step 7: Feature Engineering (Adding Month column)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Step 8: Monthly Average Max Temperature
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show()

# Step 9: Advanced Analysis - Rainfall Prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')

# Step 10: Export Insights to CSV
monthly_avg_max_temp.to_csv('monthly_avg_max_temp.csv', index=True)
print("Monthly average max temperature saved to 'monthly_avg_max_temp.csv'")
