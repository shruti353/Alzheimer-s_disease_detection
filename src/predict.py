#!/usr/bin/env python3
"""
predict.py
Predict a single image or directory of images using either ONNX (.onnx) or Keras (.h5).

Examples:
# ONNX
python src/predict.py --image_path samples/img1.png --model_path model/model.onnx

# Keras
python src/predict.py --image_path samples/img1.png --model_path model/model.h5

# Folder (predict all images)
python src/predict.py --image_path samples_folder/ --model_path model/model.h5 --batch
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications import efficientnet

IMG_SIZE = (224,224)
VALID_EXT = ('.png', '.jpg', '.jpeg')

def preprocess_image(p: Path, img_size=IMG_SIZE):
    im = Image.open(p).convert('RGB')
    im = im.resize(img_size)
    arr = np.array(im).astype('float32')
    arr = efficientnet.preprocess_input(arr)   # EfficientNet preprocessing
    return arr

def load_keras_and_predict(model_path, images):
    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))
    X = np.stack(images, axis=0)
    preds = model.predict(X)
    return preds

def load_onnx_and_predict(model_path, images):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    X = np.stack(images, axis=0).astype('float32')
    preds = sess.run(None, {input_name: X})[0]
    return preds

def main(args):
    mp = Path(args.model_path)
    ip = Path(args.image_path)
    if ip.is_dir():
        image_files = sorted([p for p in ip.iterdir() if p.suffix.lower() in VALID_EXT])
    else:
        image_files = [ip]

    if len(image_files) == 0:
        print("No images found at:", ip)
        return

    images = [preprocess_image(p, img_size=(args.img_size, args.img_size)) for p in image_files]

    preds = None
    if mp.suffix.lower() == '.onnx':
        try:
            preds = load_onnx_and_predict(mp, images)
        except Exception as e:
            print("ONNX runtime error:", e)
            return
    else:
        preds = load_keras_and_predict(mp, images)

    # labels: optionally pass a labels file containing one label per line in class order
    labels = None
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.exists():
            labels = [line.strip() for line in open(labels_path, 'r', encoding='utf8') if line.strip()]

    for p, pr in zip(image_files, preds):
        top_idx = int(np.argmax(pr))
        prob = float(np.max(pr))
        label = labels[top_idx] if labels and top_idx < len(labels) else str(top_idx)
        print(f"{p.name}: predicted -> {label} (confidence: {prob:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Path to single image or directory of images')
    parser.add_argument('--model_path', required=True, help='Path to model (.onnx or .h5)')
    parser.add_argument('--labels', required=False, help='Optional: path to labels txt (one label per line, ordered)')
    parser.add_argument('--batch', action='store_true', help='If set and image_path is a folder, predict all images')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()
    main(args)
