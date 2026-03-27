"""
score.py  —  Asma Riyaz (60305750)
------------------------------------
Azure ML batch endpoint scoring script for the CNN model.
Deployed by Maymona; scoring logic written by Asma.

This script is loaded by Azure ML's batch deployment runtime.
It must implement init() and run(mini_batch).

Input:  Mini-batch of PNG image file paths (from Azure Blob Storage)
Output: DataFrame with columns [filename, prediction, confidence]
"""

import os, logging
import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger("azureml")

MODEL       = None
IMG_SIZE    = (224, 224)
THRESHOLD   = 0.5
FLARE_LABEL = {0: "no-flare", 1: "flare"}


def init():
    """Called once when the batch deployment starts. Load the model."""
    global MODEL

    # Azure ML injects AZUREML_MODEL_DIR pointing to the registered model folder
    model_dir = os.environ.get("AZUREML_MODEL_DIR", "outputs/cnn/")
    model_path = os.path.join(model_dir, "cnn_best.keras")

    logger.info(f"Loading model from {model_path}")
    MODEL = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")


def run(mini_batch):
    """
    Called for each mini-batch of input files.

    Parameters
    ----------
    mini_batch : list[str]
        List of local file paths to PNG images in the current mini-batch.

    Returns
    -------
    pd.DataFrame with columns: filename, prediction, confidence
    """
    results = []

    for img_path in mini_batch:
        try:
            # Load and preprocess
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, axis=0)  # add batch dim

            # Predict
            proba = float(MODEL.predict(img, verbose=0)[0][0])
            pred  = 1 if proba >= THRESHOLD else 0
            label = FLARE_LABEL[pred]

            results.append({
                "filename":   os.path.basename(img_path),
                "prediction": label,
                "confidence": round(proba, 4),
            })

        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            results.append({
                "filename":   os.path.basename(img_path),
                "prediction": "error",
                "confidence": -1.0,
            })

    return pd.DataFrame(results)
