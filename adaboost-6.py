#pip install pandas numpy scikit-learn xgboost


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create models
m1 = AdaBoostClassifier()
m2 = GradientBoostingClassifier()
m3 = XGBClassifier(eval_metric='mlogloss')

# Train models
m1.fit(x_train, y_train)
m2.fit(x_train, y_train)
m3.fit(x_train, y_train)

# Predict
p1 = m1.predict(x_test)
p2 = m2.predict(x_test)
p3 = m3.predict(x_test)

# Display accuracy
print("AdaBoost Accuracy:", accuracy_score(y_test, p1))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, p2))
print("XGBoost Accuracy:", accuracy_score(y_test, p3))

# Voting Classifier
voting = VotingClassifier(
    estimators=[('ada', m1), ('gb', m2), ('xgb', m3)],
    voting='hard'
)

voting.fit(x_train, y_train)

result = voting.predict(x_test)

print("Voting Accuracy:", accuracy_score(y_test, result))
