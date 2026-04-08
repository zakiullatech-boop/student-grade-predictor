import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("student-mat.csv", sep=';')

# Select features (simple ones for now)
X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

# Predict
predictions = model.predict(X_test)

# Show results
print("Predictions:", predictions[:5])
print("Actual:", y_test.values[:5])

# Accuracy metric
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)