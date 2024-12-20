from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, validation_dir, target_size = (224,224),batch_size = 16):
    train_datagen = ImageDataGenerator(
        rescale = 1./255, 
        rotation_range =30,
        width_shift_range =0.2,
        height_shit_range = 0.2,
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
        )
    validation_datagen = ImageDataGenerator(
        rescale = 1./255
        )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size = batch_size
        class_mode='binary' # binary classification
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary' # binary classification
    )
    return train_generator, validation_generator