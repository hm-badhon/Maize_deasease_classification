import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_data(train_dir, validation_dir, target_size=(224, 224), batch_size=16):
    """
    Loads and preprocesses image data for training and validation using ImageDataGenerator.

    Args:
        train_dir (str): Path to the training data directory.
        validation_dir (str): Path to the validation data directory.
        target_size (tuple): Target size for resizing images (height, width). Default is (224, 224).
        batch_size (int): Batch size for data generators. Default is 16.

    Returns:
        tuple: (train_generator, validation_generator) for training and validation data.

    Raises:
        ValueError: If train_dir or validation_dir is invalid or empty.
    """
    if not train_dir or not validation_dir:
        raise ValueError("train_dir and validation_dir must be valid directory paths.")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'  # Binary classification
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'  # Binary classification
    )
    
    return train_generator, validation_generator

def transfer_learning_model(input_shape=(224, 224, 3)):
    """
    Creates a transfer learning model using ResNet50 as the base.

    Args:
        input_shape (tuple): Input shape for the model (height, width, channels). Default is (224, 224, 3).

    Returns:
        Model: Compiled Keras model with ResNet50 base and custom top layers.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def custom_cnn(input_shape=(224, 224, 3)):
    """
    Creates a custom CNN model for binary classification.

    Args:
        input_shape (tuple): Input shape for the model (height, width, channels). Default is (224, 224, 3).

    Returns:
        Sequential: Keras Sequential model with convolutional layers.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(model, train_generator, validation_generator, epochs=10, learning_rate=0.001, loss_fn='binary_crossentropy', metrics_list=None, verbose=1):
    """
    Compiles and trains a Keras model using the Adam optimizer.

    Args:
        model: Keras model to train.
        train_generator: Generator for training data.
        validation_generator: Generator for validation data.
        epochs (int): Number of epochs to train. Default is 10.
        learning_rate (float): Learning rate for Adam optimizer. Default is 0.001.
        loss_fn (str): Loss function to use. Default is 'binary_crossentropy'.
        metrics_list (list): List of metrics to track. Default is ['accuracy'].
        verbose (int): Verbosity mode for training (0, 1, or 2). Default is 1.

    Returns:
        History: Keras History object from model.fit().

    Raises:
        ValueError: If train_generator or validation_generator is None.
    """
    if train_generator is None or validation_generator is None:
        raise ValueError("train_generator and validation_generator must not be None.")

    if metrics_list is None:
        metrics_list = ['accuracy']

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_fn,
                  metrics=metrics_list)
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=verbose
    )
    return history

def main():
    """
    Main function to orchestrate image classification training using command-line arguments.

    Parses arguments, loads data, selects and builds the model, trains it, and saves the trained model.
    """
    parser = argparse.ArgumentParser(description="Train an image classification model using Keras.")
    parser.add_argument('--train-dir', type=str, required=True, help="Path to the training data directory.")
    parser.add_argument('--validation-dir', type=str, required=True, help="Path to the validation data directory.")
    parser.add_argument('--model-type', type=str, choices=['resnet50', 'custom_cnn'], default='custom_cnn',
                        help="Model type to use: 'resnet50' or 'custom_cnn'. Default is 'custom_cnn'.")
    parser.add_argument('--model-save-path', type=str, required=True, help="Path to save the trained model file (.h5 or .keras).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train. Default is 10.")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="Learning rate for the optimizer. Default is 0.001.")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for data generators. Default is 16.")
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                        help="Target image size (height width). Default is 224 224.")
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1,
                        help="Verbosity mode: 0 (silent), 1 (progress bar), 2 (one line per epoch). Default is 1.")

    args = parser.parse_args()

    # Validate directories and save path
    if not os.path.isdir(args.train_dir) or not os.path.isdir(args.validation_dir):
        raise ValueError("Provided train_dir or validation_dir does not exist or is not a directory.")
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    train_generator, validation_generator = load_data(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size
    )

    # Select and build model
    input_shape = (*args.target_size, 3)
    if args.model_type == 'resnet50':
        model = transfer_learning_model(input_shape=input_shape)
    else:
        model = custom_cnn(input_shape=input_shape)

    # Train model
    history = train_model(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        verbose=args.verbose
    )

    # Save the trained model
    try:
        model.save(args.model_save_path)
        print(f"Model saved successfully to {args.model_save_path}")
    except Exception as e:
        raise ValueError(f"Failed to save model to {args.model_save_path}: {str(e)}")

    return history

if __name__ == "__main__":
    main()