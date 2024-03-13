# -*- coding: utf-8 -*-


# Importing the Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

Dataset=sklearn.datasets.load_breast_cancer()
print(Dataset)

df=pd.DataFrame(Dataset.data,columns=Dataset.feature_names)

# adding the 'target' column to df

df['label']=Dataset.target
#df.tail()

# checking for missing values
print(df.isnull().sum())

# Inspecting distribution of Target Varibale
print(df['label'].value_counts())

df.groupby('label').mean()
X = df.drop(columns='label', axis=1)
Y = df['label']

# Splitting into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

from sklearn.preprocessing import StandardScaler

# Standardizing  the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Importing tensorflow and Keras
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

# Setting The Layers and makig two hidden layers having 30 nodes each

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(30, activation='relu'),
                          keras.layers.Dense(30, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])

# Compiling The Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# We choose sparse_categorical_crossentropy because number of labels are less

# Training The Neural Network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# Visualizing accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'lower right')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')

loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)

Y_pred = model.predict(X_test_std)


# Converting prediction probability to  labels
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)



input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# Changing input_data to a numpy array
data = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = data.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
#print(prediction)

# Choosing The Required Label
prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('Please Immediately Start The Treatment as The tumor Is Malignant  ')

else:
  print('The tumor is Benign')