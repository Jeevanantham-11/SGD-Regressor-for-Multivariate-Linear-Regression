# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and prepare the multivariate input data and target values. 

2.Initialize the SGD Regressor with appropriate learning rate and iterations.

3.Train the model using the given dataset and predict the output values.

4.Compare actual and predicted values using graphical visualization.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Jeevanantham C
RegisterNumber: 212225240059
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

# Select first 3 features (for demonstration)
X = data.data[:, :3]   # shape (n_samples, 3)

# Create a multi-output target: [median_house_value, some_other_numeric_column]
# Here we use column index 6 (for demonstration) as the second output
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])
```
### Output:
<img width="498" height="73" alt="image" src="https://github.com/user-attachments/assets/239fa4ba-f21f-4433-b92c-0d93f33d4eb7" />


```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)
```
### Output:
<img width="352" height="42" alt="image" src="https://github.com/user-attachments/assets/13257f00-dd77-4f61-8e5e-6ae53ae55897" />


```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))
```
### Output:
<img width="742" height="40" alt="image" src="https://github.com/user-attachments/assets/c7bf7555-292a-400b-a6b5-d29c5f4fe471" />


```
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.
multi_output_sgd = MultiOutputRegressor(sgd)

# Fit on scaled training data
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)
```
### Output:
<img width="617" height="160" alt="image" src="https://github.com/user-attachments/assets/6b9c6a7d-8458-4f81-a429-c5a0db7bccf7" />


```
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   # predicted in scaled space
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)         # back to original units
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)    # ground-truth back to original

print("First 5 predictions (original units):")
print(Y_pred[:5])
```
### Output:
<img width="326" height="111" alt="image" src="https://github.com/user-attachments/assets/e7436a60-0a71-48dc-b0f8-67fd9f5cc814" />


```
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)

# Per-output MSE (optional, helpful for debugging)
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)
```
### Output:
<img width="460" height="38" alt="image" src="https://github.com/user-attachments/assets/ffff1517-b7c2-4e98-b3f5-650942b05bff" />


```
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted   :", Y_pred[i])
    print("-" * 40)
```
### Output:
<img width="492" height="440" alt="image" src="https://github.com/user-attachments/assets/bbd97c15-a579-4ea1-9eec-8b7d7c0a4024" />


```
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
X, y = data.data[:, :3], data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

# Compare
print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))
print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))
```
### Output:
<img width="388" height="38" alt="image" src="https://github.com/user-attachments/assets/f250c7ca-5607-4fe7-8ba5-82da5ceceae8" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
