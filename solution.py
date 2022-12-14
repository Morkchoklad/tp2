from sklearn import tree
import numpy as np
import csv
from sklearn.model_selection import train_test_split

file = open('output.csv')
csvreader = csv.reader(file)

header = []
header = next(csvreader)

X = []
for row in csvreader:
        X.append(row)




y = []

for a in X:
    
    
    y.append(a.pop())    

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.33, random_state=42)

print(y_test)