\# Student Grade Predictor



This project is a beginner machine learning project in Python.



\## Tools Used

\- Python

\- pandas

\- scikit-learn

\- matplotlib



\## Goal

Predict student grades using basic machine learning techniques.

This means the model predicts final grades within approximately ±1.25 points on average.
## Results

The model was trained to predict student grades based on study time, failures, and absences.

Further improvements can include:
- adding more features
- trying different models
- improving accuracy


## Model Performance

The model was trained using Linear Regression.

### Features used:
- Study time
- Failures
- Absences
- Previous grades (G1, G2)

### Results:
- Mean Absolute Error: ~1.25

This means the model predicts final grades within approximately ±1.25 points on average.

## Visualization

The scatter plot shows predicted vs actual grades.

Points close to the red line indicate accurate predictions.

## Model Saving

The trained model is saved as `model.pkl` and can be reused without retraining.

Example input:
- studytime
- failures
- absences
- G1, G2