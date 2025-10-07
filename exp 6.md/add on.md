from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
import numpy as np
input_layer = Input(shape=(max_len,))
x = Embedding(input_dim=len(word_to_index), output_dim=50, input_length=max_len)(input_layer)
x = Bidirectional(LSTM(units=50, return_sequences=True))(x)
output_layer = TimeDistributed(Dense(len(tag_to_index), activation='softmax'))(x)
model = Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
predictions = model.predict(padded_sentences)
index_to_tag = {v: k for k, v in tag_to_index.items()}
predicted_tag_indices = np.argmax(predictions, axis=-1)
predicted_tags = [[index_to_tag[index] for index in sentence_tags if index != tag_to_index["<PAD>"]] for sentence_tags in predicted_tag_indices]
for i, sentence in enumerate(corpus):
    original_sentence = " ".join([word for word, tag in sentence])
    print(f"Original Sentence: {original_sentence}")
    print(f"Predicted POS Tags: {predicted_tags[i]}")
    print("-" * 50)
    output
  <img width="1160" height="466" alt="Screenshot 2025-09-24 112501" src="https://github.com/user-attachments/assets/6070cfc6-38a1-43db-8825-9650e7fdd309" />
