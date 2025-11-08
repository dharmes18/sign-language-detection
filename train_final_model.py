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

def convert_complete_dataset():
    """Convert ALL available data from your dataset"""
    print("🔄 CONVERTING COMPLETE DATASET...")
    print("=" * 60)
    
    dataset_path = "D:/sign language detection/dataset/asl_alphabet_train/asl_alphabet_train"
    output_path = "data_complete"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    detector = HandDetector()
    
    # Get all signs (A-Z + space, del, nothing)
    signs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Processing {len(signs)} signs: {sorted(signs)}")
    
    total_samples = 0
    successful_signs = 0
    
    for sign in signs:
        sign_folder = os.path.join(dataset_path, sign)
        print(f"\n🔄 Processing {sign}...")
        
        sign_samples = []
        image_files = [f for f in os.listdir(sign_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"   Found {len(image_files)} images")
        
        # Process ALL images (or up to 2000 for speed)
        max_images = min(2000, len(image_files))  # Use 2000 per sign for balance
        processed_count = 0
        
        for i, img_file in enumerate(image_files[:max_images]):
            img_path = os.path.join(sign_folder, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize for consistency
                    img = cv2.resize(img, (640, 480))
                    
                    # Detect hands and get landmarks
                    img = detector.find_hands(img, draw=False)
                    landmarks = detector.get_normalized_landmarks(img)
                    
                    if landmarks.size > 0:
                        sign_samples.append(landmarks)
                        processed_count += 1
                
                # Progress every 200 images
                if (i + 1) % 200 == 0:
                    print(f"   ...Processed {i + 1}/{max_images} images")
                    
            except Exception as e:
                continue
        
        # Save the landmarks
        if sign_samples:
            save_path = os.path.join(output_path, f"{sign}.npy")
            np.save(save_path, np.array(sign_samples))
            print(f"✅ {sign}: Saved {len(sign_samples)} samples ({processed_count} successful)")
            total_samples += len(sign_samples)
            successful_signs += 1
        else:
            print(f"❌ {sign}: No hand landmarks detected")
    
    print(f"\n🎉 DATASET CONVERSION COMPLETE!")
    print(f"📊 Successfully converted {successful_signs}/{len(signs)} signs")
    print(f"📊 Total samples: {total_samples}")
    print(f"📊 Average samples per sign: {total_samples/successful_signs if successful_signs > 0 else 0:.0f}")
    
    return True

def create_final_model(num_features, num_classes):
    """Create the final optimized model"""
    model = Sequential([
        # Input layer
        Dense(512, activation='relu', input_shape=(num_features,)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layers
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def load_complete_data(data_path='data_complete/'):
    """Load the complete dataset"""
    X = []
    y = []
    
    if not os.path.exists(data_path):
        print(f"❌ Data path {data_path} does not exist!")
        return None, None
    
    print("📁 Loading complete dataset...")
    
    for filename in os.listdir(data_path):
        if filename.endswith('.npy'):
            label = filename.replace('.npy', '')
            file_path = os.path.join(data_path, filename)
            
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Ensure proper shape
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                print(f"   {label}: {len(data)} samples")
                
                for sample in data:
                    if sample.size == 63:  # 21 landmarks * 3 coordinates
                        X.append(sample)
                        y.append(label)
                        
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    if len(X) == 0:
        print("❌ No valid data found!")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Total samples: {len(X):,}")
    print(f"📊 Number of classes: {len(np.unique(y))}")
    print(f"📊 Classes: {sorted(np.unique(y))}")
    print(f"📊 Input shape: {X.shape}")
    
    return X, y

def train_final_model():
    """Train the final complete model"""
    print("🚀 TRAINING FINAL COMPLETE MODEL")
    print("=" * 60)
    
    # Step 1: Convert complete dataset
    if not convert_complete_dataset():
        return
    
    # Step 2: Load data
    X, y = load_complete_data()
    
    if X is None:
        print("❌ No data to train on.")
        return
    
    # Step 3: Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    num_classes = len(label_encoder.classes_)
    num_features = X.shape[1]
    
    print(f"\n📊 FINAL DATASET INFORMATION:")
    print(f"   Number of features: {num_features}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Total training samples: {len(X):,}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.15,  # More data for training
        random_state=42, 
        stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test,
        test_size=0.5,
        random_state=42
    )
    
    print(f"\n📊 DATA SPLIT:")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_val.shape[0]:,}")
    print(f"   Testing samples: {X_test.shape[0]:,}")
    
    # Step 5: Create final model
    model = create_final_model(num_features, num_classes)
    
    print(f"\n🧠 FINAL MODEL ARCHITECTURE:")
    model.summary()
    
    # Create models directory
    os.makedirs('models_final', exist_ok=True)
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint(
            'models_final/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=0.000001,
            verbose=1
        )
    ]
    
    # Step 6: Train final model
    print(f"\n🎯 STARTING FINAL TRAINING...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,  # Larger batch size for more data
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Step 7: Save everything
    model.save('models_final/sign_language_model_final.h5')
    print("✅ Final model saved: models_final/sign_language_model_final.h5")
    
    np.save('models_final/label_encoder.npy', label_encoder.classes_)
    print("✅ Label encoder saved")
    
    class_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    with open('models_final/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("✅ Class mapping saved")
    
    # Step 8: Comprehensive evaluation
    print(f"\n📈 FINAL MODEL EVALUATION:")
    
    # Test accuracy
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Test Precision: {test_precision:.4f}")
    print(f"   Test Recall: {test_recall:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Training accuracy
    train_accuracy = history.history['accuracy'][-1]
    print(f"   Final Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Step 9: Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Final Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Final Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models_final/final_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print(f"\n🎉 FINAL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"⭐ MODEL PERFORMANCE SUMMARY:")
    print(f"   - Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   - Number of Signs: {num_classes}")
    print(f"   - Total Training Samples: {len(X):,}")
    print(f"   - Model File: models_final/sign_language_model_final.h5")
    print(f"   - Signs: {list(label_encoder.classes_)}")
    
    return model, history, label_encoder

if __name__ == "__main__":
    # Train the final complete model
    model, history, label_encoder = train_final_model()
    
    print(f"\n🚀 NEXT STEP:")
    print(f"   Run: python real_time_detection_final.py")
    print(f"   to test the final model with all signs!")