import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_csv('co2.csv')
df.describe()
print(df)

sns.countplot(x='out1', data=df)
plt.subplots(figsize=(9 , 9))
sns.heatmap(df.corr() , annot= True)
plt.show()

x=df.drop("out1", axis=1)
y=df.out1
print(x)

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2)
model= linear_model.LinearRegression()
model.fit(X_train , y_train)
out_robot = model.predict(X_test)
print(out_robot)

plt.scatter(X_test.engine , y_test, color='red')
plt.scatter(X_test.fuelcomb , y_test, color='blue')
plt.scatter(X_test.cylandr , y_test, color='green')
plt.plot(X_test, out_robot, color='black', linewidth=2)

plt.show()
