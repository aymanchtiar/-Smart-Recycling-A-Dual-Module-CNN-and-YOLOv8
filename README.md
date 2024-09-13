
# README

## Overview

This project consists of multiple models and a Kivy-based graphical interface for processing and classifying images. The main components of the project are:

1. **1_R-CNN_model**: A Region-based Convolutional Neural Network (R-CNN) model for object detection.
2. **2_CNN_none_organic**: A Convolutional Neural Network (CNN) model for classifying images as organic or non-organic.
3. **3_CNN_binary_classification**: A CNN model for binary classification of non-organic images.
4. **1_YOLOv8_model.pt**: A YOLOv8 model for detecting hazardous components.
5. **4_kivy_interface.py**: A Kivy-based graphical user interface for interacting with the models.
6. **5_training_and_testing.ipynb**: A Jupyter notebook for training and testing the models, accessible via Google Colab.

## Libraries and Dependencies

To run the code, you need to install the following libraries:

```bash
pip install opencv-python-headless
pip install tensorflow
pip install numpy
pip install kivy
pip install ultralytics
```

## Running the Code

### Setting Up the Environment

1. **Install the required libraries**:


### Running the Kivy Interface

To run the Kivy interface, execute the following command:

```bash
python 4_kivy_interface.py
```

### Training and Testing the Models

If you want to train and test the models, you can use the Jupyter notebook provided. This notebook can be accessed via Google Colab for ease of use.

**Google Colab Link**: [Training and Testing Notebook](https://colab.research.google.com/drive/1ERXQ2x_v7BGaOX8-juy2IeJBWXvlpicC?usp=sharing)

### Code Structure

1. **Model Loading**:

   ```python
   import tensorflow as tf
   from ultralytics import YOLO

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
   ```

2. **Frame Preprocessing**:

   ```python
   import cv2
   import numpy as np

   def preprocess_frame(frame, target_size=(512, 384)):
       frame_resized = cv2.resize(frame, target_size)
       frame_preprocessed = tf.keras.applications.efficientnet.preprocess_input(frame_resized)
       return np.expand_dims(frame_preprocessed, axis=0)
   ```

3. **Frame Processing and Display**:

   ```python
   from kivy.graphics.texture import Texture

   def process_and_display_frame(frame, widget):
       # Frame processing code
       buf = cv2.flip(frame, 0).tobytes()
       texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
       texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
       widget.texture = texture
   ```

4. **Kivy Interface**:

   ```python
   from kivy.app import App
   from kivy.uix.boxlayout import BoxLayout
   from kivy.uix.button import Button
   from kivy.uix.image import Image
   from kivy.uix.screenmanager import ScreenManager, Screen

   class OptionsScreen(Screen):
       # Screen setup code

   class CameraScreen(Screen):
       # Camera screen setup code

   class HazardousComponentScreen(Screen):
       # Hazardous component screen setup code

   class CamApp(App):
       def build(self):
           sm = ScreenManager()
           sm.add_widget(OptionsScreen(name='options_screen'))
           sm.add_widget(CameraScreen(name='camera_screen'))
           sm.add_widget(HazardousComponentScreen(name='hazardous_component_screen'))
           return sm

   if __name__ == '__main__':
       CamApp().run()
   ```

## Additional Notes

- Ensure that your camera is connected and properly configured to use the camera functionalities in the Kivy interface.


