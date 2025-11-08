import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json

def load_data(data_path='data/'):
    """
    Load all collected data from numpy files
    """
    X = []
    y = []
    
    if not os.path.exists(data_path):
        print(f"❌ Data path {data_path} does not exist!")
        return None, None
    
    print("📁 Loading data from files...")
    
    for filename in os.listdir(data_path):
        if filename.endswith('.npy'):
            label = filename.replace('.npy', '')
            file_path = os.path.join(data_path, filename)
            
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Handle both single samples and batches
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                print(f"   {label}: {len(data)} samples")
                
                for sample in data:
                    if sample.size == 63:  # 21 landmarks * 3 coordinates
                        X.append(sample)
                        y.append(label)
                    else:
                        print(f"   ⚠️ Skipping sample with incorrect shape: {sample.shape}")
                        
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    if len(X) == 0:
        print("❌ No valid data found!")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Total samples: {len(X)}")
    print(f"📊 Number of classes: {len(np.unique(y))}")
    print(f"📊 Classes: {np.unique(y)}")
    print(f"📊 Input shape: {X.shape}")
    
    return X, y

def create_advanced_model(num_features, num_classes):
    """
    Create an advanced neural network model
    """
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_shape=(num_features,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_model():
    """
    Main training function
    """
    print("🚀 Starting Sign Language Model Training")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    if X is None:
        print("❌ No data to train on. Please collect data first.")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    num_classes = len(label_encoder.classes_)
    num_features = X.shape[1]
    
    print(f"\n📊 Dataset Information:")
    print(f"   Number of features: {num_features}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # Create model
    model = create_advanced_model(num_features, num_classes)
    
    print(f"\n🧠 Model Architecture:")
    model.summary()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            'checkpoints/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🎯 Starting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Save final model
    model.save('models/sign_language_model.h5')
    print("✅ Final model saved as: models/sign_language_model.h5")
    
    # Save label encoder
    np.save('models/label_encoder.npy', label_encoder.classes_)
    print("✅ Label encoder saved as: models/label_encoder.npy")
    
    # Save class mapping
    class_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("✅ Class mapping saved as: models/class_mapping.json")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print(f"\n📈 Model Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Final summary
    print(f"\n🎉 Training Completed Successfully!")
    print(f"📊 Final Results:")
    print(f"   - Model: models/sign_language_model.h5")
    print(f"   - Accuracy: {test_accuracy*100:.2f}%")
    print(f"   - Classes: {list(label_encoder.classes_)}")
    print(f"   - Total training samples: {len(X)}")
    
    return model, history, label_encoder

def quick_test_model():
    """
    Quick test to verify the trained model works
    """
    print("\n🔍 Quick Model Test...")
    
    try:
        from tensorflow.keras.models import load_model
        import numpy as np
        
        # Load model and label encoder
        model = load_model('models/sign_language_model.h5')
        label_encoder = np.load('models/label_encoder.npy', allow_pickle=True)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {list(label_encoder)}")
        
        # Create dummy test data
        dummy_input = np.random.randn(1, 63)
        prediction = model.predict(dummy_input, verbose=0)
        predicted_class = label_encoder[np.argmax(prediction)]
        
        print(f"✅ Model inference test passed!")
        print(f"   Sample prediction: {predicted_class}")
        print(f"   Prediction shape: {prediction.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

if __name__ == "__main__":
    # Train the model
    model, history, label_encoder = train_model()
    
    # Quick test
    quick_test_model()
    
    print(f"\n🚀 Next step: Run 'python real_time_detection.py' for real-time sign detection!")