#pip install pandas numpy scikit-learn

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("car_evaluation.csv", header=None)

data.columns = ["A", "B", "C", "D","E", "F", "class"]

le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col])

x = data.drop("class", axis=1)
y = data["class"]

model = RandomForestClassifier()

model.fit(x, y)

result = model.predict([[1, 1, 2, 1, 1, 0]])
print("Car Safety:", result[0])
