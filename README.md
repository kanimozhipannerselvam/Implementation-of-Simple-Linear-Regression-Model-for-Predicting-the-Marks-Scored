# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe.

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kanimozhi
RegisterNumber: 212222230060
*/
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('dataset/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/

## Output:

![233586995-3162a75b-2ee4-4a5a-bc39-1491cd869629](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/792ebea9-4147-4d48-8ac5-5a3374be7ea6)


![233587105-5a7e3059-e0f1-451c-ab17-67527596d7b3](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/451c2cfb-4bf2-4602-b87c-d0cb6c1a4b07)

![233587147-c705553c-1cc6-4969-8f5b-20bbdf0f20d5](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/3d99c84a-7edb-4b4a-9304-4039b5145ac3)

![233587191-42c69afc-324d-4d8e-8e20-e87236a45ca8](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/08bf5e0d-b452-4c89-82d7-87adcdea7fdc)

![233587444-68a43d67-011c-43a8-96c9-81a96f178b2f](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/4a6e7f62-3496-4761-96d7-f1e5862d9672)

-![image](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/a68d2055-d438-45c2-84ea-82009924c79c)


![image](https://github.com/kanimozhipannerselvam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476060/58046949-36e5-4805-8715-1febeb23251d)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
