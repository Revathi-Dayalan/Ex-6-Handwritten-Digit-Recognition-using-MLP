# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
Introduction: The project titled "Digit Recognition using Artificial Neural Networks (ANN)" endeavors to develop a sophisticated system capable of accurately identifying and categorizing handwritten digits. Utilizing the capabilities of machine learning, specifically Artificial Neural Networks, the project aims to achieve precise recognition of digits ranging from 0 to 9.

Dataset: The project relies on the widely acknowledged MNIST dataset, a fundamental resource in the machine learning community. This dataset comprises 28x28 grayscale images of handwritten digits, accompanied by corresponding labels, making it an ideal tool for both training and testing the neural network.

Artificial Neural Network (ANN): The architecture of the Artificial Neural Network consists of multiple layers, including the input layer, hidden layers, and the output layer. Through a combination of feedforward and backpropagation techniques, the network is designed to comprehend intricate patterns and nuances within the dataset.




## Algorithm :
Data Preprocessing: The initial step involves normalizing pixel values and appropriately formatting labels to ensure the data is suitable for training the neural network.

Model Architecture: The architecture of the ANN is carefully designed, taking into consideration the specific number of layers, neurons, and activation functions. The selection of these components is crucial for the overall performance and accuracy of the network.

Model Training: The model undergoes training using mini-batch gradient descent and backpropagation. These optimization techniques play a vital role in refining the network's parameters, enhancing its ability to accurately recognize and classify handwritten digits.

Model Evaluation: The project employs various metrics, including accuracy, precision, recall, and the F1 score, to comprehensively assess the model's performance and its capability to make accurate predictions.

Model Deployment: The final model is deployed with a user-friendly interface, enabling users to input their own handwritten digits for real-time recognition and visualization of the model's predictions.




## Program:
DEPENDENCIES:
```
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
LOADING AND DATA-PREPROCESSING
```
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train[0].shape
x_train[0]
plt.matshow(x_train[7])
y_train[7]
x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)
```
NETWORK ARCHITECTURE
```
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))
```
TRAINING - VALIDATION
```
model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
f=model.fit(x_train,y_train,epochs=5, validation_split=0.3)
f.history
```
VISUALIZATION
```
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f.history['loss'], color = 'green', label='loss')
plt.plot(f.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('LOSS', fontsize=20)
plt.legend(loc='upper left')
plt.show()

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f.history['accuracy'], color = 'green', label='accuracy')
plt.plot(f.history['val_accuracy'], color = 'orange', label = 'val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()
```
TESTING
```
prediction = model.predict(x_test)
print(prediction)
print(np.argmax(prediction[0]))
plt.imshow(x_test[0])
```
SAVING THE MODEL
```
model.save(os.path.join('model','digit_recognizer.keras'),save_format = 'keras')
```
PREDICTION
```
img = cv2.imread('test.png')
plt.imshow(img)
rimg=cv2.resize(img,(28,28))
plt.imshow(rimg)
rimg.shape
new_model = load_model(os.path.join('model','digit_recognizer.keras'))
new_img = tf.keras.utils.normalize(rimg, axis = 1)
new_img = np.array(rimg).reshape(-1,28,28,1)
prediction = model.predict(new_img)
print(np.argmax(prediction))
new_img.shape
```

## Output :
# MODEL SUMMARY
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/e04c5813-d0d0-409d-893e-4b5ec3544214)
# TRAINING
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/f6ab460c-9699-498b-b668-0152225f7873)
# ACCURACY AND LOSS PERCENTAIL
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/b3202fbb-ba87-47ec-b9ee-cf2cad3f1c78)
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/ca0a1a30-926d-4100-ac41-58f52de0ac62)
# PREDICTION
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/fdb5bed6-0e36-4781-94ea-2c13444c91fc)
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/928499ea-3638-4c3f-927f-1c3649b89c1a)
![image](https://github.com/Revathi-Dayalan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/96000574/fa7be2cf-63ba-4921-9c8b-21c2f30e55bb)



## Result:
To sum up, the "Digit Recognition using Artificial Neural Networks (ANN)" project demonstrates the effectiveness of deep learning in precisely classifying handwritten digits. By showcasing the practical application of ANN in image recognition tasks, the project establishes a foundation for further exploration and advancement in the fields of computer vision and deep learning


