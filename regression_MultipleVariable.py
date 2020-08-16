import pandas as pd
import numpy as np
import math
from sklearn import linear_model

######### Preprocessing the dataset ###########

df = pd.read_csv("dataset.csv")
m= math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(m) # Filling the NaN value in the dataset
######### Creating a Linear Regression model #####

print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['areas','bedrooms','age']],df.price)

print(reg)

print("The set of coeficients is:")
print(reg.coef_)
print("The intercept is :")
print(reg.intercept_)

pred= reg.predict([[3000,3,40]])

print("The predicted price is :",+pred)






