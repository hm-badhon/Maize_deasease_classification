from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import Flatten, Dense, Dropout, Conv2D, MaxPooling2D

def transfer_learning_model(input_shape=(224,224,3)):

    base_model = ResNet50(weights='imagenet',include_top = False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs= base_model.input, outputs= output)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def custom_cnn(input_shape=(224,224,3)):
    model = Sequential([
        Conv2D(32,(3,3),activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128,activation = 'relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model