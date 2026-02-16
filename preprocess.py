import os, numpy as np, pandas as pd, pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

SEQ_LEN = 30

def build_sequences_from_keypoints(keypoints_root='data/keypoints', out_npz='data/sequences/dataset.npz'):
    keypoints_root = Path(keypoints_root)
    classes = sorted([p.name for p in keypoints_root.iterdir() if p.is_dir()])
    X = []
    y = []
    for c in classes:
        files = sorted(keypoints_root.joinpath(c).glob('*.npy'))
        # load and convert to sequence list
        kps = [np.load(str(f)) for f in files]
        # For each contiguous window of length SEQ_LEN create sequence (non-overlapping stride SEQ_LEN)
        for i in range(0, max(1, len(kps) - SEQ_LEN + 1), SEQ_LEN):
            seq = kps[i:i+SEQ_LEN]
            if len(seq) != SEQ_LEN:
                continue
            # flatten each frame to (34,) coords (x,y for 17 keypoints)
            seq_flat = []
            for frame_kp in seq:
                coords = []
                for (y_coord, x_coord, score) in frame_kp:
                    coords.extend([float(x_coord), float(y_coord)])
                seq_flat.append(coords)
            X.append(seq_flat)
            y.append(c)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    if len(X)==0:
        raise ValueError('No sequences found. Collect more data or reduce SEQ_LEN.')
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # save label encoder
    with open(os.path.join(os.path.dirname(out_npz),'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    np.savez_compressed(out_npz, sequences=X, labels=y_enc)
    print('Saved dataset to', out_npz, 'with', X.shape[0], 'sequences and classes:', list(le.classes_))

if __name__ == '__main__':
    build_sequences_from_keypoints()
