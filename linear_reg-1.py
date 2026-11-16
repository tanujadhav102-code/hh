#pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,6,8,10])

model=LinearRegression()
model.fit(x,y)
prediction=model.predict([[10]])

print("slope:",model.coef_)
print("Intercept:",model.intercept_)
print("prediction",prediction)

plt.scatter(x,y,color='blue', label='actual_data')
plt.plot(x,model.predict(x),color='red',label='regression_line')
plt.xlabel("Input")
plt.ylabel("output")
plt.title("Regression line")
plt.show()
