CODE:

from sklearn.datasets import fetch_olivetti_faces 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.utils import to_categorical 
import numpy as np 

faces = fetch_olivetti_faces() 
X, y = faces.images, faces.target 
X = X.reshape(-1, 64, 64, 1) 
X = X.astype('float32') 
y = to_categorical(y) 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

model = Sequential([ 
Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)), 
MaxPooling2D((2,2)), 
Flatten(), 
Dense(256, activation='relu'), 
Dense(40, activation='softmax') 
]) 
model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

OUPTUT:
<img width="1513" height="432" alt="Screenshot 2025-08-13 120648" src="https://github.com/user-attachments/assets/f02f8015-61e0-4086-a526-93f47e284d77" />
