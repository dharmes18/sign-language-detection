import cv2
import numpy as np
import tensorflow as tf
from utils import HandDetector

class SignLanguageDetector:
    def __init__(self, model_path='models_500/sign_language_model_500.h5', 
                 labels_path='models_500/label_encoder.npy'):
        self.detector = HandDetector()
        
        # Load your 99.37% accurate model!
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.labels = np.load(labels_path, allow_pickle=True)
            print(f"✅ Model loaded with 99.37% accuracy!")
            print(f"📊 Signs available: {list(self.labels)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
    
    def predict(self, landmarks):
        if landmarks.size == 0:
            return None, 0
        
        # Reshape for model input
        landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        predictions = self.model.predict(landmarks, verbose=0)
        confidence = np.max(predictions)
        predicted_class = self.labels[np.argmax(predictions)]
        
        return predicted_class, confidence
    
    def run_detection(self):
        cap = cv2.VideoCapture(0)
        
        print("🚀 Starting real-time detection with 99.37% accurate model!")
        print("Press 'q' to quit")
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            # Flip for mirror effect
            img = cv2.flip(img, 1)
            
            # Detect hands
            img = self.detector.find_hands(img)
            landmarks = self.detector.get_normalized_landmarks(img)
            
            prediction = "Show your hand"
            confidence = 0
            
            if landmarks.size > 0:
                prediction, confidence = self.predict(landmarks)
            
            # Display results with color coding
            if confidence > 0.9:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.7:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            cv2.putText(img, f"Sign: {prediction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, "Press 'q' to quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show accuracy info
            cv2.putText(img, "Model Accuracy: 99.37%", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Sign Language Detection - 99.37% Accurate", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run_detection()