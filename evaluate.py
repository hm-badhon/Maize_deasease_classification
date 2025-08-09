import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def load_validation_data(validation_dir, target_size=(224, 224), batch_size=16):
    """
    Loads and preprocesses validation image data using ImageDataGenerator.

    Args:
        validation_dir (str): Path to the validation data directory.
        target_size (tuple): Target size for resizing images (height, width). Default is (224, 224).
        batch_size (int): Batch size for the data generator. Default is 16.

    Returns:
        validation_generator: Generator for validation data.

    Raises:
        ValueError: If validation_dir is invalid or empty.
    """
    if not validation_dir:
        raise ValueError("validation_dir must be a valid directory path.")
    if not os.path.isdir(validation_dir):
        raise ValueError(f"Directory {validation_dir} does not exist or is not a directory.")

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'  # Binary classification
    )
    
    return validation_generator

def evaluate_model(model, validation_generator):
    """
    Evaluates a Keras model on a validation dataset.

    Args:
        model: Keras model to evaluate.
        validation_generator: Generator for validation data.

    Returns:
        tuple: (val_loss, val_acc) containing validation loss and accuracy.

    Raises:
        ValueError: If validation_generator is None.
    """
    if validation_generator is None:
        raise ValueError("validation_generator must not be None.")
    
    val_loss, val_acc = model.evaluate(validation_generator, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    return val_loss, val_acc

def main():
    """
    Main function to evaluate a trained Keras model on a validation dataset using command-line arguments.

    Parses arguments, loads the validation data and model, and evaluates the model.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained Keras model on a validation dataset.")
    parser.add_argument('--validation-dir', type=str, required=True, help="Path to the validation data directory.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model file (.h5 or .keras).")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for the validation data generator. Default is 16.")
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                        help="Target image size (height width). Default is 224 224.")
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1,
                        help="Verbosity mode: 0 (silent), 1 (progress bar), 2 (one line per epoch). Default is 1.")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.model_path):
        raise ValueError(f"Model file {args.model_path} does not exist or is not a file.")

    # Load validation data
    validation_generator = load_validation_data(
        validation_dir=args.validation_dir,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size
    )

    # Load model
    try:
        model = load_model(args.model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {args.model_path}: {str(e)}")

    # Evaluate model
    val_loss, val_acc = evaluate_model(
        model=model,
        validation_generator=validation_generator
    )

    return val_loss, val_acc

if __name__ == "__main__":
    main()