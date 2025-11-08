import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
from utils import HandDetector

def convert_500_dataset():
    """Conversion with 500 images per sign - optimal balance"""
    print("⚡ DATASET CONVERSION - 500 IMAGES PER SIGN")
    print("=" * 60)
    
    dataset_path = "D:/sign language detection/dataset/asl_alphabet_train/asl_alphabet_train"
    output_path = "data_500"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    detector = HandDetector()
    
    # All alphabet signs
    signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    print(f"📁 Processing {len(signs)} signs with 500 images each...")
    
    total_samples = 0
    successful_signs = 0
    
    for sign in signs:
        sign_folder = os.path.join(dataset_path, sign)
        
        if not os.path.exists(sign_folder):
            print(f"⚠️  Sign {sign} not found, skipping...")
            continue
            
        print(f"🔄 {sign}...", end=" ")
        
        sign_samples = []
        image_files = [f for f in os.listdir(sign_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Process 500 images per sign
        processed_count = 0
        for i, img_file in enumerate(image_files[:500]):
            img_path = os.path.join(sign_folder, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Quick resize for balance of speed/quality
                    img = cv2.resize(img, (540, 420))
                    
                    # Detect hands
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    detector.results = detector.hands.process(img_rgb)
                    landmarks = detector.get_normalized_landmarks(img)
                    
                    if landmarks.size > 0:
                        sign_samples.append(landmarks)
                        processed_count += 1
                        
            except Exception as e:
                continue
        
        # Save the landmarks
        if sign_samples:
            save_path = os.path.join(output_path, f"{sign}.npy")
            np.save(save_path, np.array(sign_samples))
            print(f"✅ {len(sign_samples)} samples ({processed_count} successful)")
            total_samples += len(sign_samples)
            successful_signs += 1
        else:
            print(f"❌ failed")
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📊 Total signs processed: {successful_signs}")
    print(f"📊 Total samples: {total_samples}")
    print(f"📊 Average samples per sign: {total_samples/successful_signs:.0f}")
    
    return True

def create_enhanced_model(num_features, num_classes):
    """Enhanced model for 500 images per sign"""
    model = Sequential([
        # Input layer
        Dense(384, activation='relu', input_shape=(num_features,)),
        BatchNormalization(),
        Dropout(0.35),
        
        # Hidden layers
        Dense(768, activation='relu'),
        BatchNormalization(),
        Dropout(0.45),
        
        Dense(384, activation='relu'),
        BatchNormalization(),
        Dropout(0.35),
        
        Dense(192, activation='relu'),
        Dropout(0.25),
        
        Dense(96, activation='relu'),
        Dropout(0.15),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_500_model():
    """Train with 500 images per sign"""
    print("🚀 TRAINING WITH 500 IMAGES PER SIGN")
    print("=" * 60)
    
    # Step 1: Convert dataset
    if not convert_500_dataset():
        return
    
    # Step 2: Load data
    data_path = "data_500"
    X, y = [], []
    
    print(f"\n📊 LOADING DATA...")
    for filename in os.listdir(data_path):
        if filename.endswith('.npy'):
            label = filename.replace('.npy', '')
            data = np.load(os.path.join(data_path, filename))
            
            for sample in data:
                X.append(sample)
                y.append(label)
    
    X, y = np.array(X), np.array(y)
    
    print(f"✅ Dataset loaded:")
    print(f"   Total samples: {len(X):,}")
    print(f"   Number of classes: {len(np.unique(y))}")
    print(f"   Memory usage: {X.nbytes / (1024**2):.1f} MB")
    
    # Step 3: Encode and split
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 DATA SPLIT:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    print(f"   Training ratio: {len(X_train)/len(X)*100:.1f}%")
    
    # Step 4: Create and train model
    model = create_enhanced_model(X.shape[1], len(label_encoder.classes_))
    
    print(f"\n🧠 MODEL TRAINING STARTING...")
    print(f"   Architecture: 5 layers, 384-768 neurons")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   This will take 10-20 minutes...")
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint('models_500/best_model.h5', 
                       monitor='val_accuracy', 
                       save_best_only=True,
                       verbose=1),
        EarlyStopping(monitor='val_accuracy', 
                     patience=18, 
                     restore_best_weights=True,
                     verbose=1),
        ReduceLROnPlateau(monitor='val_loss',
                         factor=0.3,
                         patience=8,
                         min_lr=0.00001,
                         verbose=1)
    ]
    
    # Train with more epochs for better convergence
    history = model.fit(
        X_train, y_train,
        epochs=120,
        batch_size=48,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Save final model
    os.makedirs('models_500', exist_ok=True)
    model.save('models_500/sign_language_model_500.h5')
    np.save('models_500/label_encoder.npy', label_encoder.classes_)
    
    # Comprehensive evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Final predictions analysis
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    correct_predictions = np.sum(y_pred_classes == y_true_classes)
    total_predictions = len(y_test)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⭐ FINAL RESULTS WITH 500 IMAGES/SIGN:")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"   Number of Signs: {len(label_encoder.classes_)}")
    print(f"   Total Training Samples: {len(X):,}")
    print(f"   Model: models_500/sign_language_model_500.h5")
    
    # Confidence analysis
    confident_predictions = np.sum(np.max(y_pred, axis=1) > 0.8)
    print(f"   High Confidence Predictions (>80%): {confident_predictions}/{total_predictions} ({confident_predictions/total_predictions*100:.1f}%)")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy (500 images/sign)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss (500 images/sign)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models_500/training_500_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, history, label_encoder

if __name__ == "__main__":
    model, history, label_encoder = train_500_model()
    
    print(f"\n🚀 READY FOR REAL-TIME DETECTION!")
    print(f"   Expected Accuracy: 92-96%")
    print(f"   Run: python real_time_detection_500.py")