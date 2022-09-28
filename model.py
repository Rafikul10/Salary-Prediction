# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:46:21 2022

@author: RAFIKUL
"""
#Importing libaries
import numpy as np
import pandas as pd
import pickle

#Importing train dataset
df=pd.read_excel("salary.xlsx")
df

#EDA process-->
df.info()

#Changing Experience(Y) column type into int
df["Experience(Y)"]=df["Experience(Y)"].map({"one":1,"two":2,"three":3,"four":4,"five":5,"saven":7,"eight":8,"nine":9,"ten":10})

#Now all let's devide the data into features and labels
x=df.drop(columns={"Salary"})
y=df.iloc[:,-1]

#Now data is ready for train 

#I will use Linearregression in this problem statement 
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

#Let's pickle the model 
pickle.dump(model,open("model.pkl","wb"))

model1=pickle.load(open("model.pkl","rb"))
#print(model1.predict[[7,8,9]])

print(model1.predict([[7,8,9]]))

