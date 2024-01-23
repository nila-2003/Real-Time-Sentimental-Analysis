# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

batch_size = 64
epochs = 30
learning_rate = 3e-4

df = pd.read_csv("/content/drive/MyDrive/llm/preprocessed_features.csv")

"""Data Preprocessing"""

df.describe()

df.columns

# df = pd.get_dummies(df, columns=['emotion'], drop_first=True)

df.columns

df['new_feature'] = df['mfcc1'] + df['mfcc2']

np.random.seed(42)
n_components = 20
pca = PCA(n_components=n_components)
df_pca = pd.DataFrame(pca.fit_transform(df.drop('emotion', axis=1)))
df = pd.concat([df[['emotion']], df_pca], axis=1)

df.columns

df.head()

np.random.seed(42)
df.iloc[:, 1:] = StandardScaler().fit_transform(df.iloc[:, 1:])

df.head()

X = df.drop('emotion', axis=1)
y = df['emotion']
rf_classifier = RandomForestClassifier()

num_features_to_select = 17
rfe = RFE(rf_classifier, n_features_to_select=num_features_to_select)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
selected_features = selected_features.tolist() + ['emotion']
df_final = df[selected_features]

# df_final = df[selected_features.union(['emotion'])]

df_final.head()

df_final.columns

"""## Data Augmentation"""

X_selected = X_rfe

shift_range = 0.1
noise_range = 0.01

augmented_features = []
for x in X_selected:
    augmented_features.append(x)

    for _ in range(3):
        shift = np.random.uniform(-shift_range, shift_range)
        x_shifted = x + shift
        augmented_features.append(x_shifted)

    x_noise = x + np.random.uniform(-noise_range, noise_range, len(x))
    augmented_features.append(x_noise)

X_augmented = np.array(augmented_features)

X_augmented[:100]

feature_names = X.columns[rfe.support_]

df_augmented = pd.DataFrame(X_augmented, columns=feature_names)
df_augmented['emotion'] = y

print(df_augmented.head())

label_encoder = LabelEncoder()
df_augmented['emotion_encoded'] = label_encoder.fit_transform(df_augmented['emotion'].fillna('unknown'))
df_augmented = shuffle(df_augmented)
df_augmented['emotion'] = df_augmented['emotion'].fillna('unknown')
df_augmented['emotion_encoded'] = label_encoder.fit_transform(df_augmented['emotion'])

X = df_augmented.drop(['emotion', 'emotion_encoded'], axis=1)
y = df_augmented['emotion_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(3, padding='same'))
model.add(LSTM(64, return_sequences=True))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
              metrics=['accuracy'])

model.summary()

cost = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

