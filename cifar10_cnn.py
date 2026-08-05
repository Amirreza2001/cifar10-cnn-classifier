"""
CIFAR-10 Image Classification: Comparing Two CNN Architectures

Trains and compares two convolutional neural networks on CIFAR-10:
  - Model 1: deeper 3-block CNN with data augmentation
  - Model 2: wider 2-block CNN without augmentation

Both models use batch normalization, dropout, and are trained with
early stopping, learning-rate reduction on plateau, and model
checkpointing. Confusion matrices and classification reports are
generated for both models on the test set.

Author: Amirreza Mohammadi
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Configuration ---
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 100

# --- 1. Load and preprocess data ---
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f"Shapes: x_train={x_train.shape}, x_test={x_test.shape}")

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_one_hot = to_categorical(y_train, NUM_CLASSES)
y_test_one_hot = to_categorical(y_test, NUM_CLASSES)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- Data augmentation layer (used only by Model 1) ---
data_augmentation = Sequential(
    [RandomFlip("horizontal"), RandomRotation(0.1), RandomZoom(0.1)],
    name="data_augmentation",
)


# --- 2. Model definitions ---
def build_cnn_model_1(input_shape, num_classes, use_augmentation=True):
    """Deeper 3-block CNN, trained with data augmentation."""
    model = Sequential(name="CIFAR10_CNN_Model_1")
    model.add(Input(shape=input_shape))
    if use_augmentation:
        model.add(data_augmentation)

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_cnn_model_2(input_shape, num_classes):
    """Wider, shallower 2-block CNN, trained without augmentation."""
    model = Sequential(name="CIFAR10_CNN_Model_2")
    model.add(Input(shape=input_shape))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# --- 3. Train / evaluate helper ---
def train_evaluate_model(model, model_name, x_train, y_train_oh, x_test, y_test_oh, epochs, batch_size):
    print(f"\n--- Training {model_name} ---")
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    log_dir = os.path.join("logs", "fit", model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_filepath = f'best_{model_name}.keras'

    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    history = model.fit(
        x_train, y_train_oh,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test_oh),
        callbacks=callbacks_list,
        verbose=1,
    )

    if os.path.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)

    loss, accuracy = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"\n{model_name} — Test Loss: {loss:.4f} | Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Total parameters: {model.count_params()}")

    # Accuracy / loss curves
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'{model_name} Training History', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{model_name}_history.png')
    plt.show()

    # Confusion matrix + classification report
    y_pred_classes = np.argmax(model.predict(x_test), axis=1)
    y_true_classes = np.argmax(y_test_oh, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, digits=4))
    return model, history, loss, accuracy


if __name__ == "__main__":
    model1 = build_cnn_model_1(INPUT_SHAPE, NUM_CLASSES, use_augmentation=True)
    model1, history1, loss1, accuracy1 = train_evaluate_model(
        model1, "Model_1_Augmented", x_train, y_train_one_hot, x_test, y_test_one_hot, EPOCHS, BATCH_SIZE
    )

    model2 = build_cnn_model_2(INPUT_SHAPE, NUM_CLASSES)
    model2, history2, loss2, accuracy2 = train_evaluate_model(
        model2, "Model_2_Simpler_NoAug", x_train, y_train_one_hot, x_test, y_test_one_hot, EPOCHS, BATCH_SIZE
    )

    print("\n--- Final Comparison ---")
    print(f"Model 1 (Augmented) Test Accuracy: {accuracy1 * 100:.2f}%")
    print(f"Model 2 (Simpler, No Augmentation) Test Accuracy: {accuracy2 * 100:.2f}%")

    # Optional: view training curves with TensorBoard
    # tensorboard --logdir logs/fit
