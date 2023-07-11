import datetime
import time

import numpy as np
import pandas as pd
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

t_0 = time.time_ns()
# Load the UrbanSound8K metadata CSV file
metadata = pd.read_csv('metadata/UrbanSound8K.csv')


# Select the first N entries
# metadata = metadata.head(500)

# Extract the audio features (MFCC) for each audio file
def extract_features(file_path):
    audio, _ = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


# Apply feature extraction to all audio files
features = []
labels = []

for index, row in metadata.iterrows():
    file_name = os.path.join('audio', 'fold' + str(row["fold"]), str(row["slice_file_name"]))
    class_label = row["class"]
    features.append(extract_features(file_name))
    labels.append(class_label)

# Convert the labels to numerical encoding
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), batch_size=128, epochs=1000,
          validation_data=(np.array(X_test), np.array(y_test)))

# Evaluate the model
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# After training the model
model.save('urban_sound_model.h5')

print(f'Total time = {time.time_ns() - t_0}')
