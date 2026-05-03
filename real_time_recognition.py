import cv2, numpy as np, time, pickle, os
import tensorflow as tf
from movenet_extractor import extract_keypoints_from_frame

MODEL_PATH = 'models/lstm_gestures.h5'
ENC_PATH = 'data/sequences/label_encoder.pkl'  # note: preprocess saves encoder in sequences folder by default

def load_model_and_encoder():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model not found, train first.')
    model = tf.keras.models.load_model(MODEL_PATH)
    if not os.path.exists(ENC_PATH):
        # try models folder
        alt = 'models/label_encoder.pkl'
        if os.path.exists(alt):
            ENC = alt
        else:
            raise FileNotFoundError('Label encoder not found.')
    else:
        ENC = ENC_PATH
    with open(ENC, 'rb') as f:
        le = pickle.load(f)
    return model, le

def run_realtime():
    model, le = load_model_and_encoder()
    cap = cv2.VideoCapture(0)
    seq = []
    SEQ_LEN = 30
    CONFIDENCE_THRESHOLD = 0.85  # Only accept predictions above 85% confidence
    PREDICTION_BUFFER_SIZE = 5   # Require 5 consecutive same predictions
    prediction_buffer = []
    last_stable_label = "Waiting..."

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            kp = extract_keypoints_from_frame(frame)
            if kp is None:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            coords = []
            h, w, _ = frame.shape
            for (y, x, s) in kp:
                coords.extend([float(x), float(y)])
            seq.append(coords)
            if len(seq) > SEQ_LEN: seq.pop(0)
            if len(seq) == SEQ_LEN:
                pred = model.predict(np.expand_dims(np.array(seq),0), verbose=0)
                confidence = float(pred.max())
                idx = int(pred.argmax(axis=1)[0])
                label = le.inverse_transform([idx])[0]

                # Apply confidence threshold and smoothing
                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(label)
                    if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
                        prediction_buffer.pop(0)

                    # Only change label if we have consistent predictions
                    if len(prediction_buffer) == PREDICTION_BUFFER_SIZE and \
                       all(p == prediction_buffer[0] for p in prediction_buffer):
                        if last_stable_label != label:
                            last_stable_label = label
                            print(f"Recognized: {label} (confidence: {confidence:.2%})")
                else:
                    prediction_buffer = []  # Reset buffer on low confidence

                display_text = f"{last_stable_label} ({confidence:.0%})" if confidence >= CONFIDENCE_THRESHOLD else f"Low confidence ({confidence:.0%})"
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # draw keypoints
            for i, (y,x,s) in enumerate(kp):
                px=int(x*w); py=int(y*h)
                if s>0.3: cv2.circle(frame,(px,py),5,(0,255,0),-1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
    run_realtime()
