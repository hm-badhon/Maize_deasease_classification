def evaluate_model(model, validation_generator):
    val_loss , val_acc = model.evaluate(validation_generator)
    print(f"Validation Accuracy:{val_acc*100:0.2f}%")
    return val_loss , val_acc

