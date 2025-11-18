import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2
)

DATASET = "dataset"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def load_data():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    ).flow_from_directory(
        f"{DATASET}/train",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        f"{DATASET}/val",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE
    )

    return train_gen, val_gen


def get_cnn_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def get_transfer_model(model_name, num_classes):
    if model_name == "resnet":
        base = ResNet50(include_top=False, input_shape=(224,224,3), weights="imagenet")
    elif model_name == "vgg":
        base = VGG16(include_top=False, input_shape=(224,224,3), weights="imagenet")
    else:
        base = MobileNetV2(include_top=False, input_shape=(224,224,3), weights="imagenet")

    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def main(model_name):
    train_gen, val_gen = load_data()
    num_classes = train_gen.num_classes

    if model_name == "cnn":
        model = get_cnn_model(num_classes)
        save_path = f"{SAVE_DIR}/cnn_model.h5"
    else:
        model = get_transfer_model(model_name, num_classes)
        save_path = f"{SAVE_DIR}/{model_name}_model.h5"

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn","resnet","vgg","mobilenet"])
    args = parser.parse_args()

    main(args.model)
