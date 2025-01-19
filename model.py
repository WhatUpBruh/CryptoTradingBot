import pandas as pd
import sklearn
import indicator as ind
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestRegressor for regression
from sklearn.metrics import accuracy_score, classification_report


def train_model(file):
    # Load the data
    data = pd.read_csv(file)

    # Separate features (X) and target (y)
    X = data.iloc[:, 1:-3]  # All columns except the last
    y = data.iloc[:, -3:]   # The last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    rf_model = RandomForestClassifier(random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_model.predict(X_test)

    # Evaluate the model

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return rf_model

def make_predictions(model, new_data):
    # Ensure new_data has the same features as the training data
    X_new = new_data[trainedModel.feature_names_in_]

    # Make predictions
    predictions = model.predict(X_new)

    return predictions



# Train the model
trainedModel = train_model('training_Data.csv')

# Get the testing data for the model
prediction_data = pd.read_csv('prediction_Data.csv')

# Use the model to make predictions
predictions = make_predictions(trainedModel, prediction_data)

# Finding the actual trend of the data
ind.trend_analysis(prediction_data)

# Updating the prediction to the csv file
prediction_data['Sideways'] = predictions[:, 0]
prediction_data['Uptrend'] = predictions[:, 1]
prediction_data['Downtrend'] = predictions[:, 2]

# Converting it to csv file
prediction_data.to_csv('prediction_Data.csv')


    

# print(type(predictions))


