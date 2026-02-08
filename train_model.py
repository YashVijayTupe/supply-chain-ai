import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = {
    'Supplier_Rating': [5, 2, 4, 1, 5, 3, 2, 4, 1, 5, 4, 2, 3, 1, 5, 2, 4, 3, 1, 5],
    'Distance_km':     [100, 800, 150, 1200, 50, 400, 900, 200, 1500, 80, 300, 600, 450, 1100, 120, 750, 250, 500, 1300, 90],
    'Weather_Factor':  [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    'Traffic_Density': [2, 8, 3, 9, 1, 6, 9, 2, 10, 2, 4, 7, 5, 9, 1, 8, 3, 4, 10, 2],
    'Vehicle_Type':    [0, 2, 1, 2, 0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 1, 0, 2, 0], 
    'Disruption':      [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Supplier_Rating', 'Distance_km', 'Weather_Factor', 'Traffic_Density', 'Vehicle_Type']]
y = df['Disruption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully trained")