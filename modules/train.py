from tensorflow.keras.optimizers import Adam

def train_model(model, train_generator, validation_generator, epochs=10 , learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history= model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = epochs
    )
    return history
