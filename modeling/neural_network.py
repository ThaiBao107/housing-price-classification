import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from data.data_loader import *



def get_data():
    data = load_data()
    feature = data.iloc[:,1:12]
    label = data.iloc[:, -1]

    return feature, label

def normalization():
    features, labels = get_data()
    data_encoded = pd.get_dummies(features, columns=['house_type', 'legal_status', 'street', 'ward',
                                              'district'])
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(data_encoded)

    encoder_label = LabelEncoder()
    label_encoded = encoder_label.fit_transform(labels)

    return feature_scaled, label_encoded, encoder_label

def training():
    feature_scaled, label_encoded, encoder_label = normalization()
    feature_train, feature_test, label_train, label_test = train_test_split(feature_scaled, label_encoded,
                                                                            test_size=0.25, random_state=42)
    with tf.device('/GPU:0'):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(feature_train.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(feature_train, label_train, epochs=100, batch_size=512, validation_split=0.2, verbose=1)

    return model, feature_train, feature_test, label_train, label_test, encoder_label

def predict():
    model, f_train, f_test, l_train, l_test, en_label = training()
    prediction = model.predict(f_test)
    pred = np.argmax(prediction, axis=1)
    cm = confusion_matrix(l_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=en_label.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    print("\n📊 Báo cáo phân loại:")
    print(classification_report(l_test, pred, target_names=en_label.classes_))



