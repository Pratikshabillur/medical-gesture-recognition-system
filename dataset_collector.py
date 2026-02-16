import cv2, os, time, numpy as np
from movenet_extractor import extract_keypoints_from_frame
from pathlib import Path

def collect_for_class(class_name, samples=300, out_dir='data'):
    out_dir = Path(out_dir)
    img_dir = out_dir / 'raw_images' / class_name
    kp_dir = out_dir / 'keypoints' / class_name
    img_dir.mkdir(parents=True, exist_ok=True)
    kp_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    collected = 0
    print(f'Starting collection for class: {class_name} samples={samples}. Press q to quit early.')
    while collected < samples:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extract_keypoints_from_frame(frame)
        if kp is None:
            # show frame but don't save
            cv2.imshow('Collect', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        timestamp = int(time.time()*1000)
        img_path = img_dir / f'{class_name}_{timestamp}.jpg'
        kp_path = kp_dir / f'{class_name}_{timestamp}.npy'
        cv2.imwrite(str(img_path), frame)
        np.save(str(kp_path), kp)
        collected += 1
        cv2.imshow('Collect', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f'Collected {collected} samples for {class_name}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', required=True)
    parser.add_argument('--samples', type=int, default=300)
    args = parser.parse_args()
    collect_for_class(args.class_name, samples=args.samples)
