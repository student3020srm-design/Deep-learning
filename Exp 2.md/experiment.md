CODE:

from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.utils import to_categorical 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255 
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
model = Sequential([ 
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
MaxPooling2D(pool_size=(2, 2)), 
Flatten(), 
Dense(128, activation='relu'), 
Dense(10, activation='softmax') 
]) 
model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, 
y_test))
import matplotlib.pyplot as plt
import numpy as np

# Show the second image in the test set
index = 1  # Python is zero-based, so 1 means "second image"
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

# Predict the label for the second image
prediction = model.predict(np.expand_dims(X_test[index], axis=0))
predicted_label = np.argmax(prediction)

print(f"Predicted label: {predicted_label}")

OUTPUT:
<img width="663" height="447" alt="Screenshot 2025-08-13 101519" src="https://github.com/user-attachments/assets/a4744f21-4fc6-4ae3-a4ec-28f802171936" />


