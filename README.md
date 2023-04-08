# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.JAYACHANDRAN
RegisterNumber:  212222240038
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
df.haed()

![image](https://user-images.githubusercontent.com/118447015/230724530-8fbd62bb-f668-48c8-8203-a8658e5cd1d3.png)

df.tail()

![image](https://user-images.githubusercontent.com/118447015/230724622-ed2ab144-1d28-427d-8c5e-7b762a277bbc.png)

Array value of X

![image](https://user-images.githubusercontent.com/118447015/230724647-c5749848-f067-44a8-9cd1-93556087c66b.png)

Array value of Y

![image](https://user-images.githubusercontent.com/118447015/230724666-57cef51d-518a-4170-9968-ac9435a6af50.png)

Values of Y prediction

![image](https://user-images.githubusercontent.com/118447015/230724696-53f60e0d-cbfa-497f-8637-d85372069bcc.png)

Array values of Y test

![image](https://user-images.githubusercontent.com/118447015/230724715-80198178-41f9-461e-95fa-85d2ab3a55ca.png)

Training Set Graph

![image](https://user-images.githubusercontent.com/118447015/230724735-58d79b0e-29e3-4a91-b589-dd3ab0483eb1.png)

Test Set Graph

![image](https://user-images.githubusercontent.com/118447015/230724764-4bd8cb79-4bd9-4fe2-a685-922e7f4cf896.png)

Values of MSE, MAE and RMSE

![image](https://user-images.githubusercontent.com/118447015/230724788-7ab640c1-538d-40a6-9cfe-0d1010f34182.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
