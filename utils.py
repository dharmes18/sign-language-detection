import cv2
import mediapipe as mp
import numpy as np
import os

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def get_landmarks(self, img, hand_number=0):
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
        return landmarks
    
    def get_normalized_landmarks(self, img, hand_number=0):
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks) if landmarks else np.array([])

def create_folders():
    folders = ['models', 'data', 'checkpoints']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def save_data(landmarks, label, data_path='data/'):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    filename = os.path.join(data_path, f'{label}.npy')
    
    if os.path.exists(filename):
        existing_data = np.load(filename, allow_pickle=True)
        # Ensure both are 2D arrays
        if existing_data.ndim == 1:
            existing_data = existing_data.reshape(1, -1)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        data = np.vstack((existing_data, landmarks))
    else:
        data = landmarks
        if data.ndim == 1:
            data = data.reshape(1, -1)
    
    np.save(filename, data)
    print(f"💾 Saved data for {label}. Total samples: {len(data)}")

def view_collected_data():
    """View statistics of collected data"""
    data_path = 'data/'
    if not os.path.exists(data_path):
        print("No data collected yet!")
        return
    
    print("\n📊 COLLECTED DATA STATISTICS:")
    print("=" * 40)
    
    total_samples = 0
    for filename in os.listdir(data_path):
        if filename.endswith('.npy'):
            label = filename.replace('.npy', '')
            file_path = os.path.join(data_path, filename)
            data = np.load(file_path, allow_pickle=True)
            samples = len(data) if data.ndim > 1 else 1
            total_samples += samples
            print(f"📁 {label}: {samples} samples")
    
    print("=" * 40)
    print(f"📈 TOTAL: {total_samples} samples across {len([f for f in os.listdir(data_path) if f.endswith('.npy')])} signs")