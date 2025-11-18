import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main(model_path):
    model = tf.keras.models.load_model(model_path)

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        "dataset/val",
        target_size=(224,224),
        class_mode="categorical"
    )

    loss, acc = model.evaluate(val_gen)
    print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    main(args.model)
