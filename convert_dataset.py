import os
import cv2
import numpy as np
from utils import HandDetector, create_folders

def convert_dataset():
    """Convert your ASL Alphabet dataset to our format"""
    print("🔄 Converting ASL Alphabet Dataset...")
    
    # Your exact dataset path
    dataset_path = "D:/sign language detection/dataset/asl_alphabet_train/asl_alphabet_train"
    output_path = "data"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    print(f"✅ Dataset found! Location: {dataset_path}")
    create_folders()
    detector = HandDetector()
    
    # Get all sign folders
    signs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Found {len(signs)} signs: {sorted(signs)}")
    
    total_samples = 0
    successful_signs = 0
    
    # Start with just A, B, C to test
    test_signs = ['A', 'B', 'C']
    
    for sign in test_signs:
        if sign not in signs:
            print(f"⚠️  Sign {sign} not found in dataset")
            continue
            
        sign_folder = os.path.join(dataset_path, sign)
        print(f"\n🔄 Processing {sign}...")
        
        sign_samples = []
        image_files = [f for f in os.listdir(sign_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"   Found {len(image_files)} images")
        
        # Process first 50 images to test
        for i, img_file in enumerate(image_files[:50]):
            img_path = os.path.join(sign_folder, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize for better performance
                    img = cv2.resize(img, (640, 480))
                    
                    # Detect hands and get landmarks
                    img = detector.find_hands(img, draw=False)
                    landmarks = detector.get_normalized_landmarks(img)
                    
                    if landmarks.size > 0:
                        sign_samples.append(landmarks)
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"   ...Processed {i + 1}/50 images")
                    
            except Exception as e:
                print(f"   Error with {img_file}: {e}")
        
        # Save the landmarks
        if sign_samples:
            save_path = os.path.join(output_path, f"{sign}.npy")
            np.save(save_path, np.array(sign_samples))
            print(f"✅ {sign}: Saved {len(sign_samples)} samples")
            total_samples += len(sign_samples)
            successful_signs += 1
        else:
            print(f"❌ {sign}: No hand landmarks detected")
    
    print(f"\n🎉 Conversion complete!")
    print(f"📊 Successfully converted {successful_signs} signs")
    print(f"📊 Total samples: {total_samples}")
    
    # Show what we have
    print(f"\n📁 Data folder contents:")
    data_files = [f for f in os.listdir(output_path) if f.endswith('.npy')]
    for file in data_files:
        data = np.load(os.path.join(output_path, file))
        print(f"   {file}: {len(data)} samples")
    
    return True

if __name__ == "__main__":
    convert_dataset()