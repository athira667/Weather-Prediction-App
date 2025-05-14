import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample weather dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert categorical columns to numerical
df = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])

# Target variable
y = df['Play'].map({'No': 0, 'Yes': 1})
X = df.drop('Play', axis=1)

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and columns
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
# Save the dataset for EDA
df['Play'] = y  # Add target column back for EDA
df.to_csv('weather.csv', index=False)
