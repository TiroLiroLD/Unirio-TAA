import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

''' Load the trained model '''
model = load_model('urban_sound_model.h5')

''' Load the UrbanSound8K metadata CSV file '''
metadata = pd.read_csv('metadata/UrbanSound8K.csv')

''' Folder number to process '''
folder_number = 1

''' Select the audio files from the specified folder'''
folder_files = metadata.loc[metadata['fold'] == folder_number]['slice_file_name']

# Initialize counters for hits and errors
hits = 0
errors = 0
label_counts = {}  # Initialize dictionaries to store the counts for each label
error_info = []  # Initialize a list to store error information

# Iterate over the audio files
for file_name in folder_files:
    # Construct the file path
    file_path = os.path.join('audio', 'fold' + str(folder_number), file_name)

    # Extract the expected label from the CSV
    expected_label = metadata.loc[metadata['slice_file_name'] == file_name]['class'].values[0]

    # Extract the audio features
    def extract_features(file_path):
        audio, _ = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40, n_fft=1024)
        return np.mean(mfccs.T, axis=0)


    features = extract_features(file_path)
    features = features.reshape(1, -1)

    # Make a prediction
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions)

    # Convert the numerical labels to class names
    le = LabelEncoder()
    le.fit(metadata['class'])
    class_labels = le.inverse_transform([predicted_label])

    # Check if the prediction matches the expected label
    if class_labels[0] == expected_label:
        hits += 1
    else:
        errors += 1

        # Append error information to the list
        error_info.append({'Expected Label': expected_label, 'Mistaken Label': class_labels[0]})

    # Update the label counts dictionary
    if expected_label not in label_counts:
        label_counts[expected_label] = {'hits': 0, 'errors': 0, 'mistaken_with': {}}
    if class_labels[0] == expected_label:
        label_counts[expected_label]['hits'] += 1
    else:
        label_counts[expected_label]['errors'] += 1
        mistaken_with_label = class_labels[0]
        if mistaken_with_label not in label_counts[expected_label]['mistaken_with']:
            label_counts[expected_label]['mistaken_with'][mistaken_with_label] = 0
        label_counts[expected_label]['mistaken_with'][mistaken_with_label] += 1

# Create bar chart for hits and errors count by label
labels = list(label_counts.keys())
hits_count = [label_counts[label]['hits'] for label in labels]
errors_count = [label_counts[label]['errors'] for label in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, hits_count, width, label='Hits')
rects2 = ax.bar(x + width / 2, errors_count, width, label='Errors')

ax.set_xlabel('Label')
ax.set_ylabel('Count')
ax.set_title('Hits and Errors Count by Label')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()

# Create bar chart for mistaken classes by label
fig, ax = plt.subplots(figsize=(12, 6))

mistaken_labels = []
mistaken_counts = []

for label in labels:
    mistaken_with = label_counts[label]['mistaken_with']
    for mistaken_label, count in mistaken_with.items():
        mistaken_labels.append(label + ' mistaken with ' + mistaken_label)
        mistaken_counts.append(count)

sns.barplot(x=mistaken_counts, y=mistaken_labels, ax=ax)
ax.set_xlabel('Count')
ax.set_ylabel('Label')
ax.set_title('Mistaken Classes by Label')

plt.tight_layout()
plt.show()

# Create a table for error information
error_df = pd.DataFrame(error_info)
print('Error Information:')
print(error_df)
