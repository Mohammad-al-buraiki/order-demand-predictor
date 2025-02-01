import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Fake Order Data (Simulating Past Orders)
data = {
    "day_of_week": np.random.randint(0, 7, 100),  # 0 = Monday, 6 = Sunday
    "hour_of_day": np.random.randint(0, 24, 100),  # 0 to 23 hours
    "total_orders": np.random.randint(5, 50, 100)  # Orders in that time slot
}

df = pd.DataFrame(data)

# Display first 5 rows
print(df.head())

# Splitting Data (Features & Target)
X = df[["day_of_week", "hour_of_day"]]
y = df["total_orders"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Show Prediction vs Actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Orders")
plt.ylabel("Predicted Orders")
plt.title("Actual vs Predicted Order Volume")
plt.show()

# Make a Prediction (Example: Friday, 6 PM)
sample_input = [[4, 18]]  # Friday, 6 PM
predicted_orders = model.predict(sample_input)
print(f"Predicted Orders for Friday 6 PM: {predicted_orders[0]:.0f}")
