import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load data
df = pd.read_csv("student-mat.csv", sep=';')

# Features and target
X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# Model 1: Linear Regression
# ========================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)


# ========================
# Model 2: Decision Tree
# ========================
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_mae = mean_absolute_error(y_test, tree_predictions)
tree_r2 = r2_score(y_test, tree_predictions)


# ========================
# Results Comparison
# ========================
print("\n--- Model Comparison ---")

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("R2:", lr_r2)

print("\nDecision Tree:")
print("MAE:", tree_mae)
print("R2:", tree_r2)


# ========================
# Choose best model
# ========================
if lr_mae < tree_mae:
    best_model = lr_model
    print("\nBest Model: Linear Regression")
else:
    best_model = tree_model
    print("\nBest Model: Decision Tree")


# ========================
# Visualization (for best model)
# ========================
best_predictions = best_model.predict(X_test)

plt.scatter(y_test, best_predictions)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Best Model: Actual vs Predicted")
plt.plot([0, 20], [0, 20], color='red')
plt.show()


# ========================
# Save best model
# ========================
joblib.dump(best_model, "model.pkl")
print("Best model saved as model.pkl")


# ========================
# Interactive prediction
# ========================
loaded_model = joblib.load("model.pkl")

print("\n--- Student Grade Prediction ---")

studytime = float(input("Enter study time (1-4): "))
failures = float(input("Enter number of failures: "))
absences = float(input("Enter number of absences: "))
g1 = float(input("Enter G1: "))
g2 = float(input("Enter G2: "))

sample = pd.DataFrame(
    [[studytime, failures, absences, g1, g2]],
    columns=['studytime', 'failures', 'absences', 'G1', 'G2']
)

prediction = loaded_model.predict(sample)

print("\nPredicted Final Grade (G3):", round(prediction[0], 2))