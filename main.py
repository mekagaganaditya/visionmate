import os
import tempfile
import threading
import cv2
import csv
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
from ultralytics import YOLO
from gtts import gTTS
import speech_recognition as sr
from queue import Queue
from kivy.lang import Builder
from kivy.uix.label import Label

class ObjectDetectionApp(App):
    def build(self):
        # Create the main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create and add the icon image at the top
        self.icon_image = Image(source='icon.png', size_hint=(None, None), size=(100, 100), pos_hint={'center_x': 0.5})
        layout.add_widget(self.icon_image)

        # Create and add the "Vision Mate" label
        vision_mate_label = Label(
            text="VisionMate",
            font_size=32,
            size_hint=(None, None),
            size=(300, 100),
            pos_hint={'center_x': 0.5}
        )
        layout.add_widget(vision_mate_label)

        # Spacer to push the instruction label to the bottom
        layout.add_widget(Label(size_hint_y=None, height=50))  # Adjust height as needed for spacing

        # Create and add the instruction label at the bottom
        self.instruction_label = Label(
            text="Please say start to begin",
            font_size=24,
            size_hint=(None, None),
            size=(300, 100),
            pos_hint={'center_x': 0.5}
        )
        layout.add_widget(self.instruction_label)

        # Create the image widget for displaying the camera feed
        self.image = Image()
        layout.add_widget(self.image)
        from google.colab import drive
        import torch

# Mount Google Drive
        drive.mount('/content/drive')






# Path to the .pt file on Google Drive
        file_path = '/content/drive/My Drive/yolov8l.pt'
        # Load YOLO model
        self.model = YOLO(file_path)

        # Load class names
        with open('coco.names', 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Load object sizes from CSV
        self.object_sizes = {}
        csv_file_path = 'coco.names.csv'
        
        with open(csv_file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.object_sizes[row['Object'].strip()] = float(row['Width(cm)']) / 100  # cm to meters

        # Camera parameters
        self.focal_length = 800  # Example focal length in pixels

        # Queue and flags for audio and voice command management
        self.current_guidance_text = ""  # Store the current guidance text
        self.audio_playing = False
        self.temp_audio_file_path = None  # To store the path of the temporary audio file
        self.capture_running = False  # Flag to track capture status

        # Start a thread to listen for voice commands
        threading.Thread(target=self.listen_for_voice_commands, daemon=True).start()

        return layout

    def listen_for_voice_commands(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        while True:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for commands...")
                audio = recognizer.listen(source)

            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {command}")

                if "start" in command and not self.capture_running:
                    self.start_capture()
                elif "stop" in command and self.capture_running:
                    self.stop_capture()
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    def start_capture(self):
        print("Starting capture...")
        self.capture_running = True
        self.instruction_label.text = "Object detection in progress..."
        threading.Thread(target=self.capture_frames, daemon=True).start()

    def stop_capture(self):
        print("Stopping capture...")
        if hasattr(self, 'cap'):
            self.cap.release()
            self.cap = None

        self.capture_running = False
        self.instruction_label.text = "Please say start to begin"
        App.get_running_app().stop()

    def capture_frames(self):
        self.cap = cv2.VideoCapture(0)  # Camera capture
        if not self.cap.isOpened():
            print("Error: Camera could not be opened.")
            return
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS

    def update(self, dt):
        if not hasattr(self, 'cap') or self.cap is None:
            print("Warning: Camera capture is not initialized.")
            return
        
        # Capture frame from the camera
        ret, frame = self.cap.read()
        if ret:
            # Apply CLAHE for histogram equalization
            frame = self.apply_clahe(frame)

            # Perform object detection
            results = self.model(frame)
            guidance_text = self.parse_results(frame, results)

            # Display guidance text on frame
            cv2.putText(frame, guidance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert image to Kivy texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

            # Queue or update audio guidance if the prompt has changed
            if guidance_text != self.current_guidance_text:
                self.current_guidance_text = guidance_text
                self.play_next_audio()
        else:
            print("Warning: Frame could not be captured.")

    def apply_clahe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def parse_results(self, frame, results):
        guidance_text = ""
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, box)
                    bbox_width = x2 - x1
                    class_name = self.class_names[int(cls_id)]
                    real_world_width = self.object_sizes.get(class_name, 0.1)

                    # Distance calculation
                    distance = (real_world_width * self.focal_length) / bbox_width if bbox_width > 0 else float('inf')
                    label = f"{class_name}: {conf:.2f}, Distance: {distance:.2f}m"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    center_x = (x1 + x2) / 2
                    center_x_norm = center_x / frame.shape[1]
                    left_threshold, right_threshold = self.adjust_thresholds(bbox_width, distance)

                    if left_threshold <= center_x_norm <= right_threshold:
                        left_space = x1
                        right_space = frame.shape[1] - x2
                        guidance_text = "Move left" if left_space > right_space else "Move right"
                    else:
                        left_space = x1 if center_x_norm < left_threshold else 0
                        right_space = frame.shape[1] - x2 if center_x_norm > right_threshold else 0
                        guidance_text = "Move right" if center_x_norm < left_threshold else "Move left"

        if guidance_text:
            object_name = self.class_names[int(cls_id)]
            direction = "right" if left_space > right_space else "left"
            guidance_text = f"'{object_name}' on '{direction}' at '{distance:.2f}m'. " + guidance_text
            
        return guidance_text

    def adjust_thresholds(self, bbox_width, distance):
        if distance < 1.0:
            return 0.4, 0.6
        elif bbox_width > 200:
            return 0.35, 0.65
        else:
            return 0.3, 0.7

    def play_next_audio(self):
        if not self.audio_playing:
            self.audio_playing = True
            tts = gTTS(self.current_guidance_text)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                self.temp_audio_file_path = temp_file.name
                tts.save(self.temp_audio_file_path)
            sound = SoundLoader.load(self.temp_audio_file_path)
            if sound:
                sound.play()
                sound.bind(on_stop=self.audio_playback_finished)
            else:
                self.audio_playing = False

    def audio_playback_finished(self, instance):
        if self.temp_audio_file_path and os.path.exists(self.temp_audio_file_path):
            os.remove(self.temp_audio_file_path)
        self.audio_playing = False

if __name__ == '__main__':
    ObjectDetectionApp().run()
