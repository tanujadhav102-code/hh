#pip install pandas numpy scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("emails.csv")

X = data.drop(["Email No.", "Prediction"], axis=1)
y = data["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = SVC()
model.fit(X_train, y_train)

result = model.predict(X_test)

if result[0] == 1:
    print("Email is Spam")
else:
    print("Email is Normal")
