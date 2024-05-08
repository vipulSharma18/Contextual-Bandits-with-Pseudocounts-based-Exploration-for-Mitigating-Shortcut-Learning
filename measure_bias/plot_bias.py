import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data from a CSV file
data = pd.read_csv('bias_data.csv')

# Filter data where p_s is 0.9
filtered_data = data[data['p_s'] == 0.9]

# Prepare the data for linear regression model
X = filtered_data[['a_s']]
y = filtered_data['bias']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict values for the existing a_s values
filtered_data['predicted_bias'] = model.predict(X)

# Calculate residuals and the standard error for the 95% confidence interval
residuals = y - filtered_data['predicted_bias']
stderr = residuals.std()
confidence_interval = 1.96 * stderr

# Sort data for plotting
sorted_data = filtered_data.sort_values(by='a_s')

# Explicitly convert columns to numpy arrays for plotting
x_values = sorted_data['a_s'].to_numpy()
y_values = sorted_data['predicted_bias'].to_numpy()
confidence_upper = y_values + confidence_interval
confidence_lower = y_values - confidence_interval

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sorted_data, x='a_s', y='bias', hue='seed', style='seed', palette='viridis', s=50, alpha=0.6)
plt.plot(x_values, y_values, color='red', label='Linear Fit')
plt.fill_between(x_values, confidence_lower, confidence_upper, color='red', alpha=0.3, label='95% Confidence Interval')
plt.title('Linear Fit with Confidence Interval for Bias (p_s = 0.9)')
plt.xlabel('a_s Value')
plt.ylabel('Bias')
plt.legend()
plt.grid(True)
plt.show()
