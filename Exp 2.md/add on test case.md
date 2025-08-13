CODE:

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# Class names for Fashion MNIST
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load and preprocess Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))

# Select test case indices for your experiment
test_indices = [0, 1, 2, 7]  # Example: 4 sample images

# Collect results
rows = []
for idx in test_indices:
    pred = np.argmax(model.predict(np.expand_dims(X_test[idx], axis=0)))
    rows.append({
        "Input Image": class_names[y_test[idx]],
        "True Label": class_names[y_test[idx]],
        "Predicted Label": class_names[pred],
        "Correct (Y/N)": "Y" if pred == y_test[idx] else "N"
    })

# Create and display DataFrame
df = pd.DataFrame(rows)
df


OUTPUT:
<img width="1119" height="557" alt="Screenshot 2025-08-13 111625" src="https://github.com/user-attachments/assets/d19e6563-1081-4e31-9d89-3d529921ed9a" />
