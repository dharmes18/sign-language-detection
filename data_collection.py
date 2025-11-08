import cv2
import numpy as np
import os
import time
from utils import HandDetector, create_folders, save_data

def collect_data():
    create_folders()
    detector = HandDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Define the signs you want to collect
    signs = ['A', 'B', 'C', 'Hello', 'Thank You', 'I Love You', 'Yes', 'No', 'Please', 'Help']
    
    print("Available signs:", signs)
    label = input("Enter the sign label: ").strip()
    
    if label not in signs:
        print(f"Warning: {label} not in predefined signs")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    samples_per_sign = 300  # Reduced for manual collection
    collected_samples = 0
    sample_count = 0
    
    print(f"\n🎥 Collecting data for sign: {label}")
    print("=" * 50)
    print("CONTROLS:")
    print("• Press 's' to start/pause collecting")
    print("• Press 'c' to capture single sample")
    print("• Press 'q' to quit and save")
    print("• Press 'r' to reset counter")
    print("=" * 50)
    
    collecting = False
    last_capture_time = 0
    capture_delay = 0.5  # Minimum time between captures (seconds)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)  # Mirror effect
        img = detector.find_hands(img)
        landmarks = detector.get_normalized_landmarks(img)
        
        # Display information on screen
        cv2.putText(img, f"Sign: {label}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Samples: {collected_samples}/{samples_per_sign}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Sample #: {sample_count}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status with color coding
        status = "COLLECTING" if collecting else "PAUSED"
        color = (0, 0, 255) if collecting else (0, 255, 255)
        cv2.putText(img, f"Status: {status}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Hand detection status
        hand_status = "Hand Detected" if landmarks.size > 0 else "No Hand"
        hand_color = (0, 255, 0) if landmarks.size > 0 else (0, 0, 255)
        cv2.putText(img, f"Hand: {hand_status}", (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        
        # Instructions
        cv2.putText(img, "Press 's': Start/Pause, 'c': Capture, 'q': Quit", (10, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Data Collection - Sign Language", img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            collecting = not collecting
            print(f"{'Started' if collecting else 'Paused'} collecting...")
        elif key == ord('r'):
            sample_count = 0
            print("Counter reset")
        elif key == ord('c'):  # Manual capture
            current_time = time.time()
            if landmarks.size > 0 and (current_time - last_capture_time) > capture_delay:
                save_data(landmarks, label)
                collected_samples += 1
                sample_count += 1
                last_capture_time = current_time
                print(f"📸 Manually captured sample {sample_count}")
        
        # Auto-collect when enabled
        if collecting and landmarks.size > 0:
            current_time = time.time()
            if (current_time - last_capture_time) > capture_delay:
                save_data(landmarks, label)
                collected_samples += 1
                sample_count += 1
                last_capture_time = current_time
                print(f"📸 Auto-captured sample {sample_count}")
                
                if collected_samples >= samples_per_sign:
                    print(f"✅ Completed collecting {samples_per_sign} samples for {label}")
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Data collection completed for {label}!")
    print(f"📊 Total samples collected: {collected_samples}")

def batch_collect_signs():
    """Collect multiple signs in one session"""
    signs = ['A', 'B', 'C', 'Hello', 'Thank You', 'I Love You']
    
    print("🔄 Batch Data Collection Mode")
    print("Available signs:", signs)
    
    for sign in signs:
        print(f"\n{'='*50}")
        print(f"🔄 Now collecting: {sign}")
        print(f"{'='*50}")
        
        response = input(f"Press Enter to collect '{sign}' or 's' to skip: ")
        if response.lower() == 's':
            print(f"⏭️ Skipped {sign}")
            continue
            
        collect_single_sign(sign)

def collect_single_sign(sign_label):
    """Collect data for a single sign"""
    create_folders()
    detector = HandDetector()
    
    cap = cv2.VideoCapture(0)
    samples_per_sign = 200
    collected_samples = 0
    
    print(f"🎥 Collecting {samples_per_sign} samples for '{sign_label}'")
    print("Press 's' to start, 'q' to quit")
    
    collecting = False
    
    while collected_samples < samples_per_sign:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmarks = detector.get_normalized_landmarks(img)
        
        # Display info
        cv2.putText(img, f"Sign: {sign_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Progress: {collected_samples}/{samples_per_sign}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status = "COLLECTING" if collecting else "PAUSED"
        color = (0, 0, 255) if collecting else (0, 255, 255)
        cv2.putText(img, f"Status: {status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(f"Collecting: {sign_label}", img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            collecting = True
        
        if collecting and landmarks.size > 0:
            save_data(landmarks, sign_label)
            collected_samples += 1
            print(f"Collected {collected_samples}/{samples_per_sign}")
            time.sleep(0.1)  # Small delay
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Completed {sign_label}: {collected_samples} samples")

if __name__ == "__main__":
    print("Choose collection mode:")
    print("1. Single sign collection")
    print("2. Batch collection (multiple signs)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        batch_collect_signs()
    else:
        collect_data()