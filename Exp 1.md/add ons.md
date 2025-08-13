code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))

for i in range(len(X)):
 if y[i] == 0:
 plt.scatter(X[i][0], X[i][1], color='red')
 else:
 plt.scatter(X[i][0], X[i][1], color='blue')

x_values = [0, 1]
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()


Output:
<img width="653" height="490" alt="Screenshot 2025-08-06 112152" src="https://github.com/user-attachments/assets/6e9dd005-bbac-4dd3-9df9-c2cb6150204c" />
