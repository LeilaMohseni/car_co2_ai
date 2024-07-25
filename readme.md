# Linear Regression 

## 1.Simple Linear Regression
```python
### Python lib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
### Open Dataset & print
df=pd.read_csv('co2.csv')
df.describe()
df
```
![dataset](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/1.jpg)
### Show by Plt
```python
sns.countplot(x='out1', data=df)
plt.subplots(figsize=(9 , 9))
sns.heatmap(df.corr() , annot= True)
```
![showplt](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/2.png)

```python
plt.show()
```
![showplt2](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/3.jpg)
```python
x=df.drop("out1", axis=1)
y=df.out1
x
```
![showoutput](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/4.jpg)
```python
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2)
model= linear_model.LinearRegression()
model.fit(X_train , y_train)
out_robot = model.predict(X_test)
out_robot
```
![out_robot](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/5.jpg)
```python
plt.scatter(X_test.engine , y_test, color='red')
plt.scatter(X_test.fuelcomb , y_test, color='blue')
plt.scatter(X_test.cylandr , y_test, color='green')
plt.plot(X_test, out_robot, color='black', linewidth=2)
plt.show()
```
![showpllt3](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/6.jpg)

### 2.Multi Linear Regression
![printx](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/11.jpg)
```python
sns.countplot(x='out1', data=df)
plt.subplots(figsize=(9 , 9))
sns.heatmap(df.corr() , annot= True)
plt.show()
```
![pltshow](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/22.jpg)

![showplt4](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/33.jpg)
```python
x=df.drop("out1", axis=1)
x=x.drop("fuelcomb", axis=1)
x=x.drop("cylandr", axis=1)
y=df.out1
y
```
![printy](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/44.jpg)
```python
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2)
model= linear_model.LinearRegression()
model.fit(X_train , y_train)
out_robot = model.predict(X_test)
out_robot
```
![printout_robot](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/55.jpg)
```python
plt.scatter(X_test , y_test, color='red')
plt.plot(X_test, out_robot, color='black', linewidth=2)
plt.show()
```
![pltshow](https://raw.githubusercontent.com/LeilaMohseni/car_co2_ai/master/img/66.jpg)
