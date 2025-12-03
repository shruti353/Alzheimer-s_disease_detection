#!/usr/bin/env python3
"""
evaluate.py
Evaluate a saved model (.h5) on a test dataset and save classification report + confusion matrix.

Example:
python src/evaluate.py --test_dir ../dataset/test --model_path ../model/model.h5 --output_dir ../model/outputs
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet

def get_test_generator(test_dir, target_size=(224,224), batch_size=32):
    preprocess_fn = efficientnet.preprocess_input
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    gen = datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
    return gen

def plot_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_gen = get_test_generator(args.test_dir, target_size=(args.img_size, args.img_size), batch_size=args.batch_size)
    classes = [k for k,v in sorted(test_gen.class_indices.items(), key=lambda x: x[1])]

    # Load model
    model = load_model(args.model_path)
    print("Model loaded:", args.model_path)

    steps = int(np.ceil(test_gen.samples / args.batch_size))
    preds_proba = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(preds_proba, axis=1)
    y_true = test_gen.classes

    # Metrics
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n", report)
    cm = confusion_matrix(y_true, y_pred)

    # Save outputs
    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    plot_confusion_matrix(cm, classes, out_dir / 'confusion_matrix.png')

    # Save numpy arrays
    np.save(out_dir / 'y_true.npy', y_true)
    np.save(out_dir / 'y_pred.npy', y_pred)

    print("Evaluation artifacts saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--model_path', required=True, help='Path to .h5 Keras model')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()
    main(args)
