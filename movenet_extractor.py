import tensorflow as tf
import numpy as np
import cv2
import os

# ================= MoveNet Local Model Path =================
MODEL_PATH = os.path.join("models", "movenet_lightning")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "MoveNet model not found! Ensure 'models/movenet_lightning/' "
        "contains saved_model.pb and variables/"
    )

print("Loading local MoveNet model...")
model = tf.saved_model.load(MODEL_PATH)
movenet = model.signatures["serving_default"]

# ================= Preprocessing =================
def _preprocess(img):
    """
    img: RGB image (H,W,3) uint8
    """
    img = tf.image.resize_with_pad(img, 192, 192)
    img = tf.cast(img, tf.int32)
    img = tf.expand_dims(img, axis=0)  # (1,192,192,3)
    return img

# ================= Keypoint Extraction =================
def extract_keypoints_from_frame(frame):
    """
    frame: BGR image from OpenCV
    returns: (17,3) keypoints or None
    """
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _preprocess(img_rgb)

        outputs = movenet(inp)
        keypoints_with_scores = outputs["output_0"].numpy()  # (1,1,17,3)

        return keypoints_with_scores[0, 0, :, :]  # (17,3)

    except Exception as e:
        print("MoveNet error:", e)
        return None
