
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_df = pd.read_csv('train-data.tsv', sep='\t', header=None, names=['label','message'])
valid_df = pd.read_csv('valid-data.tsv', sep='\t', header=None, names=['label','message'])

train_labels = train_df['label'].factorize()[0]
label_index = train_df['label'].factorize()[1]
label_dict = {label: idx for idx, label in enumerate(label_index)}
valid_labels = valid_df['label'].map(label_dict).values

train_messages = train_df['message'].values
valid_messages = valid_df['message'].values

# ⚙️ Vectorization & Embedding
max_tokens = 10000
max_len = 100

vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    standardize='lower_and_strip_punctuation',
    output_mode='int',
    output_sequence_length=max_len
)
vectorizer.adapt(train_messages)

# Model building
model = keras.Sequential([
    layers.Input(shape=(1,), dtype=tf.string, name="SMS_input"),
    vectorizer,
    layers.Embedding(input_dim=max_tokens, output_dim=64),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ✅ Train
history = model.fit(
    train_messages, train_labels,
    epochs=10,
    validation_data=(valid_messages, valid_labels)
)

# 🧪 Prediction function
def predict_message(msg: str):
    prob = float(model.predict(np.array([msg]))[0][0])
    label = 'spam' if prob >= 0.5 else 'ham'
    return [prob, label]

# 🔍 Test your model
for test_msg in [
    "how are you doing today",
    "sale today! to stop texts call 98912460324",
    "i don't want to go. can we try a different day?",
    "our new mobile video service is live. install on your phone to start watching.",
    "you have won £1000 cash! call to claim your prize.",
]:
    print(test_msg, "→", predict_message(test_msg))
