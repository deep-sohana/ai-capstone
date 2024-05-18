import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Data: Renewable Energy Generation (Gigawatt hours) and Investment ($ Billions)
data = {
    "Year": range(1997, 2022),
    "Generation_GWh": [
        16803.5, 17761.7, 17955.5, 17838.0, 17426.3, 18835.0, 18903.7, 20405.0, 
        21743.6, 21185.8, 19868.8, 18644.0, 21802.7, 26523.7, 26655.4, 33199.0, 
        36588.2, 34035.0, 38145.9, 40454.1, 44642.8, 52023.7, 59930.3, 70798.3, 
        83995.6
    ],
    "Investment_Billion": [
        0, 0, 0, 0, 0.10, 0.21, 0.35, 0.40, 0.61, 0.88, 1.00, 1.20, 1.50, 1.80, 
        2.14, 2.67, 3.12, 3.32, 3.55, 4.10, 4.29, 3.90, 4.11, 4.07, 5.15
    ]
}

df = pd.DataFrame(data)

# Calculate annual growth rate for generation
df['Growth_Rate'] = df['Generation_GWh'].pct_change() * 100
print(df)

# Plot the data
plt.figure(figsize=(12, 6))

# Plot renewable energy generation
plt.subplot(1, 2, 1)
plt.plot(df['Year'], df['Generation_GWh'], marker='o', color='b', label='Generation (GWh)')
plt.xlabel('Year')
plt.ylabel('Generation (GWh)')
plt.title('Renewable Energy Generation Over Time')
plt.legend()

# Plot investment
plt.subplot(1, 2, 2)
plt.plot(df['Year'], df['Investment_Billion'], marker='o', color='g', label='Investment ($ Billions)')
plt.xlabel('Year')
plt.ylabel('Investment ($ Billions)')
plt.title('Renewable Energy Investment Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Prepare the data for GBR
X = df[['Year', 'Investment_Billion']].values
y = df['Generation_GWh'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# Predict and plot the results
y_pred = gbr.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Generation_GWh'], color='blue', label='Actual Generation')
plt.plot(X_test[:, 0], y_pred, color='red', linewidth=2, label='GBR Predictions')
plt.xlabel('Year')
plt.ylabel('Renewable Energy Generation (GWh)')
plt.title('Renewable Energy Generation Over Time with Investment')
plt.legend()
plt.show()
