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
                idx = int(pred.argmax(axis=1)[0])
                label = le.inverse_transform([idx])[0]

                # 🔥🔥 Added: Print recognized gesture in the console
                print("Recognized:", label)

                cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

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
