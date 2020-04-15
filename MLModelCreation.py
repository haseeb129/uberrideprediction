import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("taxi.csv")
y = data["Numberofweeklyriders"]
X = data.drop("Numberofweeklyriders", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)
pickle.dump(reg, open('taxi.pk1', 'wb'))

model = pickle.load(open('taxi.pk1', 'rb'))
# conda list to view install pakages
