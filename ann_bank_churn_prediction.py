# -*- coding: utf-8 -*-
"""ANN_Bank_Churn_Prediction.py

# Artificial Neural Network

### Importing the libraries
"""

import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

tensorflow.__version__

"""## Part 1 - Data Preprocessing

### Importing the dataset
"""

df = pd.read_csv('Churn_Modelling.csv')
df.head(10)

df.shape

df.isna().sum()

"""Dropping the irrelevant columns"""

df.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)
df.head(3)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X)

print(y)

"""### Encoding categorical data

Label Encoding the "Gender" column
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

print(X)

"""Encoding the "Geography" column"""

X = pd.get_dummies(X, columns=['Geography'])

print(X)

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""### Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Part 2 - Building the ANN

### Initializing the ANN
"""

ann_model = Sequential()

"""### Adding the input layer and the first hidden layer"""

ann_model.add(Dense(units=6, activation='relu'))

"""### Adding the second hidden layer"""

ann_model.add(Dense(units=6, activation='relu'))

"""### Adding the output layer"""

ann_model.add(Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the ANN on the Training set"""

ann_model.fit(X_train, y_train, batch_size = 32, epochs = 100)

ann_model.summary()

"""## Part 4 - Making the predictions and evaluating the model

### Predicting the result of a single observation

Predicting if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: \$ 60000

Number of Products: 2

Does this customer have a credit card ? Yes

Is this customer an Active Member: Yes

Estimated Salary: \$ 50000

So, should we say goodbye to that customer ?
"""

print(ann_model.predict(sc.transform([[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 1, 0, 0]])) > 0.5)

"""### Predicting the Test set results"""

y_pred = ann_model.predict(X_test)

y_predicted = []

for i in y_pred:
    if i <0.5:
        y_predicted.append(0)
    else:
        y_predicted.append(1)

"""### Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_predicted)
print(cm)
accuracy_score(y_test, y_predicted)
