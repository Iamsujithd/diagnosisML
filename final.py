import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training and testing data
train_data = pd.read_csv('Training.csv')
test_data = pd.read_csv('Testing.csv')

# Assuming the last column is the target (prognosis) and the rest are features (symptoms)
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict prognosis based on user input symptoms
def predict_prognosis(symptoms):
    # Convert the input symptoms to a dataframe
    input_data = pd.DataFrame([symptoms], columns=X_train.columns)
    
    # Predict prognosis
    prediction = rf_model.predict(input_data)[0]
    prediction_proba = rf_model.predict_proba(input_data)[0]
    
    # Find the confidence of the prediction
    confidence = max(prediction_proba) * 100
    
    return prediction, confidence

# Interactive symptom input
def get_user_input():
    symptoms = {}
    print("Please enter the presence of the following symptoms (1 for Yes, 0 for No):")
    for symptom in X_train.columns:
        while True:
            try:
                value = input(f"{symptom}: ").strip().lower()
                if value in ['1', '0', 'yes', 'no']:
                    symptoms[symptom] = 1 if value in ['1', 'yes'] else 0
                    break
                else:
                    print("Invalid input. Please enter 1 for Yes, 0 for No.")
            except ValueError:
                print("Invalid input. Please enter 1 for Yes, 0 for No.")
    return symptoms

# Main function to run the interactive prediction
def main():
    symptoms = get_user_input()
    if all(value == 0 for value in symptoms.values()):
        print("\nNo symptoms selected. Unable to provide a prognosis based on no symptoms.")
    else:
        prognosis, confidence = predict_prognosis(symptoms)
        print(f"\nPredicted Prognosis: {prognosis}")
        print(f"Confidence Level: {confidence:.2f}%")

if __name__ == "__main__":
    main()
