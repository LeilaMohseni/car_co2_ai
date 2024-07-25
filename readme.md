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
![dataset](https://drive.google.com/file/d/1BX3mVXv0yhXDletgrXzEXRbnxt9gu2DW/view?usp=drive_link)
### Show by Plt
```python
sns.countplot(x='out1', data=df)
plt.subplots(figsize=(9 , 9))
sns.heatmap(df.corr() , annot= True)
```
![showplt](https://drive.google.com/file/d/1NvRikjHCLLLRoQ6Qd0a2EUnKLMVfDcqp/view?usp=drive_link)

```python
plt.show()
```
![showplt2](https://drive.google.com/file/d/15yz7fpQVU0j5gP7bQzg_x16t9ysG6sRE/view?usp=drive_link)
```python
x=df.drop("out1", axis=1)
y=df.out1
x
```
![showoutput](https://drive.google.com/file/d/1PMr21pyrulKmgFnXrCN3CDmEgunAOuQI/view?usp=drive_link)
```python
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2)
model= linear_model.LinearRegression()
model.fit(X_train , y_train)
out_robot = model.predict(X_test)
out_robot
```
![out_robot](https://drive.google.com/file/d/1dEvA-gwhdK3zlw26qvzx6uU1YkPgH89s/view?usp=drive_link)
```python
plt.scatter(X_test.engine , y_test, color='red')
plt.scatter(X_test.fuelcomb , y_test, color='blue')
plt.scatter(X_test.cylandr , y_test, color='green')
plt.plot(X_test, out_robot, color='black', linewidth=2)
plt.show()
```
![showpllt3](https://drive.google.com/file/d/1jkH6hxWtwGnFJXZ8w329b_VAcshTWZcu/view?usp=drive_link)

### 2.Multi Linear Regression
![printx](https://drive.google.com/file/d/1K3bMa4cc3bL7cdoqDqs2VB9pn5nMN4Mi/view?usp=drive_link)
```python
sns.countplot(x='out1', data=df)
plt.subplots(figsize=(9 , 9))
sns.heatmap(df.corr() , annot= True)
plt.show()
```
![pltshow](https://drive.google.com/file/d/187X8wI-dOZc2HV0Djfq3PEf-MpoSiotN/view?usp=drive_link)

![showplt4](https://drive.google.com/file/d/1VtQs078vxaC3MUR4iSKe38XptVB3xJKr/view?usp=drive_link)
```python
x=df.drop("out1", axis=1)
x=x.drop("fuelcomb", axis=1)
x=x.drop("cylandr", axis=1)
y=df.out1
y
```
![printy](https://drive.google.com/file/d/1VzSOBMZXI19SpRA1ye9hkdLv8XubjydL/view?usp=drive_link)
```python
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2)
model= linear_model.LinearRegression()
model.fit(X_train , y_train)
out_robot = model.predict(X_test)
out_robot
```
![printout_robot](https://drive.google.com/file/d/1dMlpcVrBz3X8Io9ueotoeTGTHRF__o1O/view?usp=drive_link)
```python
plt.scatter(X_test , y_test, color='red')
plt.plot(X_test, out_robot, color='black', linewidth=2)
plt.show()
```
![pltshow](https://drive.google.com/file/d/1BkgU5-G-frGOgip2LZFFR_cnLQ9p1fLb/view?usp=drive_link)
