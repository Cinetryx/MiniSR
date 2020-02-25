import os
import csv
import pickle
import librosa
import librosa.display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import randint

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras import models
from keras import layers

from keras.models import load_model


###################################################################################################

labels =  ['bye', 'hello', 'merci', 'yo']

print(f"Name of labels: {labels}")

try:
    data = pd.read_csv('data.csv')
except:
    nbRecordings = []

    header = 'filename label chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    file = open('data/output/data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for label in labels:
        print(f"Creating data ({label}) plz wait !")
        waves = [f for f in os.listdir(trainPath + label) if f.endswith('.wav')]
        nbRecordings.append(len(waves))
        for i, wav in enumerate(waves):
            filepath = trainPath + label + '/' + wav
            samples, sample_rate = librosa.load(filepath)
            
            plt.figure(figsize=(12, 5))
            librosa.display.waveplot(samples, sample_rate)
            plt.savefig(f"data/graph/{label}/{label}_{i}_waveform.png")
            plt.close()

            data = librosa.stft(samples)
            Xdb = librosa.amplitude_to_db(abs(data))
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
            plt.colorbar()
            plt.savefig(f"data/graph/{label}/{label}_{i}_spectral.png")
            plt.close()

            mfccs = librosa.feature.mfcc(samples, sr=sample_rate)
            plt.figure(figsize=(10, 5))
            librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
            plt.savefig(f"data/graph/{label}/{label}_{i}_mfccs.png")
            plt.close()

            hop_length = 512
            chromagram = librosa.feature.chroma_stft(samples, sr=sample_rate, hop_length=hop_length)
            plt.figure(figsize=(15, 5))
            librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
            plt.savefig(f"data/graph/{label}/{label}_{i}_chromagram.png")
            plt.close()

            chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
            spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
            spec_bw = librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sample_rate)
            zcr = librosa.feature.zero_crossing_rate(samples)
            mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
            rms = librosa.feature.rms(y=samples)
            
            to_append = f'{wav} {label} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            
            file = open('data/output/data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
            
            data = pd.read_csv('data/output/data.csv')

    print(f"Number of recording by each class: {nbRecordings}")

print(data)

data = data.drop(['filename'], axis=1)

genre_list = data.iloc[:, 0]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

"""
y = y.reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
y = enc.fit(y).transform(y).toarray()
"""

data = data.drop(['label'], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

print(f"X shape: {X.shape}")
print(f"y shape {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.25)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

test_loss, test_acc = model.evaluate(X_test, y_test)

print("\n")
print(f"Test Shape: {X_test.shape}")
print(f"Test accuracy: {test_acc}")
print("\n")

model.save("model.h5")

"""
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("data/output/accuracy.graph.png")
plt.show()
plt.close()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("data/output/loss.graph.png")
plt.show()
plt.close()
"""

model = load_model("model.h5")

for i in range(X_test.shape[0]):
    tst = model.predict(np.array([X_test[i]]))
    print(labels[np.argmax(tst)], labels[y_test[i]], tst)
    print(X_test[i])

    print("\n")

def preprocessAndPred(filepath, model="model"):
    model = load_model(model+".h5")

    samples, sample_rate = librosa.load(filepath)
    chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
    spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(samples)
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
    rms = librosa.feature.rms(y=samples)

    data = [np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
        data.append(np.mean(e))

    data.pop(-1)

    print(f"Lenght data (25): {len(data)}")
    data = np.array([data], dtype = float)

    print(f"data shape before normalize: {data.shape}")

    scaler = StandardScaler()
    predX = scaler.fit_transform(data)

    #from sklearn.preprocessing import scale
    #predX = scale(data)

    #from sklearn.preprocessing import normalize
    #predX = normalize(data)

    print(predX[0])
    print(f"data shape after normalize: {data.shape}")

    awaitLabel = filepath.split("/")
    
    predict = model.predict(predX)
    label = labels[np.argmax(model.predict(predX))]
    awaitLabel = awaitLabel[2].replace(".wav", "")

    print(f"{awaitLabel} => {label} {predict}")
    print("\n")
    #return label, predict


filepath = "bye3.wav"
preprocessAndPred(filepath)

"""
filepath = "data/test/bye.wav"
preprocessAndPred(filepath)

filepath = "data/test/merci.wav"
preprocessAndPred(filepath)

filepath = "data/test/bye2.wav"
preprocessAndPred(filepath)

filepath = "data/test/hello.wav"
preprocessAndPred(filepath)

filepath = "data/test/merci2.wav"
preprocessAndPred(filepath)

filepath = "data/test/yo.wav"
preprocessAndPred(filepath)
"""
