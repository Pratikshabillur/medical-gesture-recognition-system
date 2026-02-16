
import os, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import argparse
import pickle
import matplotlib.pyplot as plt  # for graphs

SAVED_DIR = 'models'
MODEL_PATH = os.path.join(SAVED_DIR, 'lstm_gestures.h5')
ENCODER_PATH = os.path.join(SAVED_DIR, 'label_encoder.pkl')

def build_model(seq_len, feat_dim, num_classes):
    model = Sequential([
        Masking(mask_value=0., input_shape=(seq_len, feat_dim)),
        LSTM(256, return_sequences=True),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train(npz_path='data/sequences/dataset.npz', epochs=50, batch_size=32):
    data = np.load(npz_path)
    X = data['sequences']
    y = data['labels']
    if X.ndim != 3:
        raise ValueError('Unexpected X shape: ' + str(X.shape))
    
    seq_len = X.shape[1]; feat = X.shape[2]
    num_classes = len(set(y))
    os.makedirs(SAVED_DIR, exist_ok=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model(seq_len, feat, num_classes)
    print(model.summary())

    # Train and capture history
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=epochs, batch_size=batch_size, verbose=1)

    # ==================== FINAL ACCURACY CALCULATION ====================
    val_predictions = np.argmax(model.predict(X_val, verbose=0), axis=1)
    overall_accuracy = accuracy_score(y_val, val_predictions)
    
    print("\n" + "="*60)
    print(f"OVERALL VALIDATION ACCURACY: {overall_accuracy*100:.2f}%")
    print("="*60 + "\n")

    # Save accuracy to a text file (great for report)
    with open('models/results.txt', 'w') as f:
        f.write(f"Overall Validation Accuracy: {overall_accuracy*100:.2f}%\n")
        f.write(f"Total Classes: {num_classes}\n")
        f.write(f"Validation Samples: {len(y_val)}\n")
        f.write(f"Epochs: {epochs} | Batch Size: {batch_size}\n")

    # Save model and encoder
    model.save(MODEL_PATH)
    le_path = os.path.join(os.path.dirname(npz_path), 'label_encoder.pkl')
    if os.path.exists(le_path):
        import shutil
        shutil.copy(le_path, ENCODER_PATH)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
    print('Model & Results Saved!')

    # ==================== GRAPHS WITH ACCURACY ANNOTATION ====================
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2.5)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5)
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=overall_accuracy, color='green', linestyle='--', linewidth=2, label=f'Final Acc: {overall_accuracy*100:.2f}%')
    plt.legend()
    plt.savefig('models/accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2.5)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5)
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_val, val_predictions)
    plt.figure(figsize=(12,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', xticks_rotation=90, values_format='d')
    plt.title(f'Confusion Matrix\n(Overall Accuracy: {overall_accuracy*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All graphs updated with Overall Accuracy!")
    print("Check: models/accuracy_plot.png | loss_plot.png | confusion_matrix.png | results.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/sequences/dataset.npz')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    train(npz_path=args.npz, epochs=args.epochs, batch_size=args.batch)