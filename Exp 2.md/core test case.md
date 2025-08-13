<img width="1046" height="386" alt="Screenshot 2025-08-13 101754" src="https://github.com/user-attachments/assets/cc8daa1a-9ab0-4028-9472-f648b437dc97" />
CODE:

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# Load and preprocess MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
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

# Train model (few epochs to be quick)
model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))

# Test case indices and expected labels
test_indices = [0, 1, 2, 3]  # Sample images
expected_labels = [7, 3, 8, 1]

# Predictions
predictions = []
correct_list = []

for idx, expected in zip(test_indices, expected_labels):
    pred = np.argmax(model.predict(np.expand_dims(X_test[idx], axis=0)))
    predictions.append(pred)
    correct_list.append("Y" if pred == expected else "N")

# Create DataFrame
df = pd.DataFrame({
    "Input Digit Image": [f"Image of {expected}" for expected in expected_labels],
    "Expected Label": expected_labels,
    "Model Output": predictions,
    "Correct (Y/N)": correct_list
})

# Display the results table
df

OUTPUT:

https://github.com/student3020srm-design/Deep-learning/issues/5#issue-3316901129
