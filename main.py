import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        spectrogram = librosa.stft(audio)
        feature = librosa.amplitude_to_db(abs(spectrogram))
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return feature


# Load metadata
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

features = []
labels = []
file_paths = []

# Iterate over each sound file and extract features
for class_label in metadata['class'].unique():
    class_data = metadata[metadata['class'] == class_label].head(2)
    for index, row in class_data.iterrows():
        relative_path = 'fold' + str(row["fold"]) + '/' + str(row["slice_file_name"])
        file_name = os.path.join(os.path.abspath('UrbanSound8K/audio/'), relative_path)
        data = extract_features(file_name)

        features.append(data)
        labels.append(class_label)
        file_paths.append(relative_path)

# Convert into a Panda dataframe
features_df = pd.DataFrame(list(zip(features, labels, file_paths)), columns=['feature', 'class_label', 'file_path'])

print(features_df)

# Organize data/label lists
X = features_df.feature.tolist()
y = features_df.class_label.tolist()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'x_train: {x_train}\n')
print(f'x_test: {x_test}\n')
print(f'y_train: {y_train}\n')
print(f'y_test: {y_test}\n')

features_df.to_csv('pandas_df.csv', index=False)

# Choose all audio files' features, labels, and file paths
features = features_df['feature']
labels = features_df['class_label']
file_paths = features_df['file_path']

fig, axes = plt.subplots(4, 5, constrained_layout=True)
for i, ax in enumerate(axes.flatten()):
    if i < len(features):
        img = ax.imshow(features[i], aspect='auto', origin='lower')
        ax.set_title(f'\n{labels[i]}'
                     f'\n{file_paths[i]}')
plt.show()

# Convert features and labels to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels to integers
le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

print(le)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Testing accuracy: ", score[1])
