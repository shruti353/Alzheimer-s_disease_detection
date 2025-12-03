#!/usr/bin/env python3
"""
train.py
Train an EfficientNetB0-based classifier for Alzheimer's stages.

Example:
python src/train.py \
  --train_dir ../dataset/train \
  --val_dir ../dataset/val \
  --output_dir ../model \
  --epochs 15 \
  --batch_size 32
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def build_model(input_shape=(224,224,3), n_classes=4, base_trainable=False, dropout_rate=0.5):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model

def get_generators(train_dir, val_dir, target_size=(224,224), batch_size=32):
    # Use EfficientNet preprocessing
    preprocess_fn = efficientnet.preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen

def plot_history(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # accuracy
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(out_dir / 'accuracy.png')
    plt.close()

    # loss
    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(out_dir / 'loss.png')
    plt.close()

def try_export_onnx(keras_model_path, onnx_path):
    try:
        import tf2onnx
        import onnx
        import tensorflow as tf
        model = tf.keras.models.load_model(str(keras_model_path))
        spec = (tf.TensorSpec((None,224,224,3), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(onnx_path))
        print(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        print("ONNX export failed (tf2onnx required). Error:", e)

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_folder = out_dir / 'outputs'
    outputs_folder.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = get_generators(args.train_dir, args.val_dir,
                                        target_size=(args.img_size, args.img_size),
                                        batch_size=args.batch_size)

    n_classes = train_gen.num_classes
    model = build_model(input_shape=(args.img_size, args.img_size, 3),
                        n_classes=n_classes,
                        base_trainable=args.unfreeze_base,
                        dropout_rate=args.dropout)
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # Callbacks
    checkpoint = ModelCheckpoint(out_dir / 'model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    steps_per_epoch = args.steps_per_epoch or (train_gen.samples // args.batch_size)
    validation_steps = args.validation_steps or (val_gen.samples // args.batch_size)

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[checkpoint, reduce_lr, early]
    )

    # Save final model and plots
    model.save(out_dir / 'final_model.h5')
    plot_history(history, outputs_folder)

    if args.export_onnx:
        try_export_onnx(out_dir / 'model.h5', out_dir / 'model.onnx')

    print("Training completed. Artifacts saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, help='Path to training directory')
    parser.add_argument('--val_dir', required=True, help='Path to validation directory')
    parser.add_argument('--output_dir', required=True, help='Where to save models and outputs')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--validation_steps', type=int, default=None)
    parser.add_argument('--unfreeze_base', action='store_true', help='If set, unfreeze EfficientNet base for fine-tuning')
    parser.add_argument('--img_size', type=int, default=224, help='Image width/height (EfficientNetB0 uses 224)')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--export_onnx', action='store_true', help='Attempt TF -> ONNX export (requires tf2onnx)')
    args = parser.parse_args()
    main(args)
