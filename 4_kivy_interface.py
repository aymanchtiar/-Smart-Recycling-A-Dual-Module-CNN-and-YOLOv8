import cv2
import tensorflow as tf
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from ultralytics import YOLO

# Load models
od_model_path = '1_R-CNN_model/saved_model'
od_model = tf.saved_model.load(od_model_path)

class_model_path = '2_CNN_none_organic'
class_model = tf.saved_model.load(class_model_path)
class_infer = class_model.signatures["serving_default"]

class_model_path1 = '3_CNN_binary_classification'
class_model1 = tf.saved_model.load(class_model_path1)
class_infer1 = class_model1.signatures["serving_default"]

yolo_model_path = '1_YOLOv8_model.pt'
yolo_model = YOLO(yolo_model_path)

classNames = ["ERIrotation", "Placards", "corrosive", "dangerous-when-wet", "explosive",
              "flammable", "flammable-solid", "infectious-substance", "license plate",
              "non-flammable-gas", "organic-peroxide", "oxidizer", "poison", "radioactive",
              "spontaneously-combustible"]

class_names = ['Cardboard and Paper', 'Glass', 'Metal', 'Cardboard and Paper', 'Plastics']
class_names_organic = ['organic', 'none organic']

def preprocess_frame(frame, target_size=(512, 384)):
    frame_resized = cv2.resize(frame, target_size)
    frame_preprocessed = tf.keras.applications.efficientnet.preprocess_input(frame_resized)
    return np.expand_dims(frame_preprocessed, axis=0)

def preprocess_frame1(frame, target_size=(512, 384)):
    frame_resized = cv2.resize(frame, target_size)
    frame_preprocessed = tf.keras.applications.efficientnet.preprocess_input(frame_resized)
    return np.expand_dims(frame_preprocessed, axis=0)

def process_and_display_frame(frame, widget):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_uint8 = tf.cast(frame_rgb, tf.uint8)
    input_tensor = tf.convert_to_tensor([frame_uint8], dtype=tf.uint8)

    detections = od_model(input_tensor)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()

    for i in range(len(detection_scores)):
        if detection_scores[i] >= 0.5:
            box = detection_boxes[i]
            image_height, image_width, _ = frame.shape
            y_min, x_min, y_max, x_max = box
            cv2.rectangle(frame, (int(x_min * image_width), int(y_min * image_height)),
                          (int(x_max * image_width), int(y_max * image_height)), (0, 255, 0), 2)

            crop_img = frame[int(y_min * image_height):int(y_max * image_height),
                             int(x_min * image_width):int(x_max * image_width)]
            if crop_img.size != 0:
                frame_processed = preprocess_frame(crop_img)
                predictions = class_infer(tf.convert_to_tensor(frame_processed, dtype=tf.float32))
                print(predictions.keys())  # Print the keys to identify the correct one
                predictions = predictions[list(predictions.keys())[0]].numpy()
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_names[predicted_class_index]

                cv2.putText(frame, f'Class: {predicted_class_name}', (int(x_min * image_width), int(y_min * image_height) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    buf = cv2.flip(frame, 0).tobytes()
    texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    widget.texture = texture

def process_and_display_frame1(frame, widget):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_uint8 = tf.cast(frame_rgb, tf.uint8)
    input_tensor = tf.convert_to_tensor([frame_uint8], dtype=tf.uint8)

    detections = od_model(input_tensor)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()

    for i in range(len(detection_scores)):
        if detection_scores[i] >= 0.5:
            box = detection_boxes[i]
            image_height, image_width, _ = frame.shape
            y_min, x_min, y_max, x_max = box
            cv2.rectangle(frame, (int(x_min * image_width), int(y_min * image_height)),
                          (int(x_max * image_width), int(y_max * image_height)), (0, 255, 0), 2)

            crop_img = frame[int(y_min * image_height):int(y_max * image_height),
                             int(x_min * image_width):int(x_max * image_width)]
            if crop_img.size != 0:
                frame_processed = preprocess_frame1(crop_img)
                predictions = class_infer1(tf.convert_to_tensor(frame_processed, dtype=tf.float32))
                print(predictions.keys())  # Print the keys to identify the correct one
                predictions = predictions[list(predictions.keys())[0]].numpy()
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_names_organic[predicted_class_index]

                cv2.putText(frame, f'Class: {predicted_class_name}', (int(x_min * image_width), int(y_min * image_height) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    buf = cv2.flip(frame, 0).tobytes()
    texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    widget.texture = texture

class OptionsScreen(Screen):
    def __init__(self, **kwargs):
        super(OptionsScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        btn1 = Button(text='Organic or Non-Organic', size_hint_y=None, height='48dp')
        btn2 = Button(text='Classify the Non-Organic Waste', size_hint_y=None, height='48dp')
        btn3 = Button(text='Detect Hazardous Component', size_hint_y=None, height='48dp')

        btn1.bind(on_press=self.to_camera_screen1)
        btn2.bind(on_press=self.to_camera_screen)
        btn3.bind(on_press=self.to_hazardous_component_screen)

        layout.add_widget(btn1)
        layout.add_widget(btn2)
        layout.add_widget(btn3)
        self.add_widget(layout)
        
    def to_camera_screen1(self, instance):
        self.manager.current = 'camera_screen1'
   
    def to_camera_screen(self, instance):
        self.manager.current = 'camera_screen'

    def to_hazardous_component_screen(self, instance):
        self.manager.current = 'hazardous_component_screen'

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        self.capture_button = Button(text='Capture Frame', size_hint_y=None, height='48dp')
        self.back_button = Button(text='Back', size_hint_y=None, height='48dp')

        self.capture_button.bind(on_press=self.capture_frame)
        self.back_button.bind(on_press=self.go_back)

        self.layout.add_widget(self.back_button)
        self.layout.add_widget(self.img1)
        self.layout.add_widget(self.capture_button)
        self.add_widget(self.layout)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)

    def capture_frame(self, instance):
        ret, frame = self.capture.read()
        if ret:
            App.get_running_app().process_frame = frame
            App.get_running_app().root.current = 'processed_screen'

    def go_back(self, instance):
        self.manager.current = 'options_screen'
        self.manager.transition.direction = 'right'

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture
            
class CameraScreen1(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen1, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        self.capture_button = Button(text='Capture Frame', size_hint_y=None, height='48dp')
        self.back_button = Button(text='Back', size_hint_y=None, height='48dp')

        self.capture_button.bind(on_press=self.capture_frame1)
        self.back_button.bind(on_press=self.go_back1)

        self.layout.add_widget(self.back_button)
        self.layout.add_widget(self.img1)
        self.layout.add_widget(self.capture_button)
        self.add_widget(self.layout)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update1, 1.0/30.0)

    def capture_frame1(self, instance):
        ret, frame = self.capture.read()
        if ret:
            App.get_running_app().process_frame = frame
            App.get_running_app().root.current = 'processed_screen1'

    def go_back1(self, instance):
        self.manager.current = 'options_screen'
        self.manager.transition.direction = 'right'

    def update1(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture
            
class HazardousComponentScreen(Screen):
    capture = None

    def __init__(self, **kwargs):
        super(HazardousComponentScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', size_hint=(1, .9))
        self.img1 = Image(size_hint=(1, 1))
        self.layout.add_widget(self.img1)

        back_btn = Button(text="Back to Options", size_hint=(1, .1))
        back_btn.bind(on_press=self.back_to_options)
        self.layout.add_widget(back_btn)

        self.add_widget(self.layout)
        
    def on_pre_enter(self):
        if not self.capture:
            self.capture = cv2.VideoCapture(0)
        self.event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    def on_pre_leave(self):
        Clock.unschedule(self.event)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        
    def back_to_options(self, instance):
        self.manager.current = 'options_screen'
        
    def update(self, dt):
        ret, img = self.capture.read()
        if ret:
            results = yolo_model(img, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    confidence = box.conf[0]
                    cls = int(box.cls[0])
                    text = f"{classNames[cls]} {confidence:.2f}"
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            buf = cv2.flip(img, 0).tobytes()
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture
            
class ProcessedScreen(Screen):
    def on_enter(self, *args):
        self.layout = BoxLayout(orientation='vertical')
        processed_frame = App.get_running_app().process_frame
        self.img2 = Image()
        if processed_frame is not None:
            process_and_display_frame(processed_frame, self.img2)
        self.back_button = Button(text='Back', size_hint_y=None, height='48dp')
        self.back_button.bind(on_press=self.go_back)
        
        self.layout.add_widget(self.img2)
        self.layout.add_widget(self.back_button)
        self.add_widget(self.layout)

    def go_back(self, instance):
        self.manager.current = 'camera_screen'
        self.manager.transition.direction = 'right'
        
class ProcessedScreen1(Screen):
    def on_enter(self, *args):
        self.layout = BoxLayout(orientation='vertical')
        processed_frame = App.get_running_app().process_frame
        self.img2 = Image()
        if processed_frame is not None:
            process_and_display_frame1(processed_frame, self.img2)
        self.back_button = Button(text='Back', size_hint_y=None, height='48dp')
        self.back_button.bind(on_press=self.go_back1)
        
        self.layout.add_widget(self.img2)
        self.layout.add_widget(self.back_button)
        self.add_widget(self.layout)

    def go_back1(self, instance):
        self.manager.current = 'camera_screen1'
        self.manager.transition.direction = 'right'

class CamApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(OptionsScreen(name='options_screen'))
        sm.add_widget(CameraScreen(name='camera_screen'))
        sm.add_widget(CameraScreen1(name='camera_screen1'))
        sm.add_widget(ProcessedScreen(name='processed_screen'))
        sm.add_widget(ProcessedScreen1(name='processed_screen1'))
        sm.add_widget(HazardousComponentScreen(name='hazardous_component_screen'))
        return sm

if __name__ == '__main__':
    CamApp().run()
