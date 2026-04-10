import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Load data
df = pd.read_csv("student-mat.csv", sep=';')

# Select features and target
X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Show sample results
print("Predictions:", predictions[:5])
print("Actual:", y_test.values[:5])

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Plot actual vs predicted
plt.scatter(y_test, predictions)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.plot([0, 20], [0, 20], color='red')
plt.show()

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

# Load model
loaded_model = joblib.load("model.pkl")


# Interactive prediction
print("\n--- Student Grade Prediction ---")
studytime = float(input("Enter study time (1-4): "))
failures = float(input("Enter number of past class failures: "))
absences = float(input("Enter number of absences: "))
g1 = float(input("Enter first period grade (G1): "))
g2 = float(input("Enter second period grade (G2): "))

sample = pd.DataFrame(
    [[studytime, failures, absences, g1, g2]],
    columns=['studytime', 'failures', 'absences', 'G1', 'G2']
)

prediction = loaded_model.predict(sample)

print("\nPredicted Final Grade (G3):", round(prediction[0], 2))