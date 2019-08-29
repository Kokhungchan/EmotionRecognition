
# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2
import dlib
import pickle
import os
import numpy as np
from keras.models import load_model
from statistics import mode
from UI_emotion import *

from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input

# # Hardcoded the directory path to models
model_dir = '/Users/kokhungchan/Desktop/emotion_recognition/model'
#
# # Declaring dlib models to detect face and generate face encodings (all three models obtained from dlib library)
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
# face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))
face_detector = dlib.get_frontal_face_detector()
#
# # Name and path of the predictor model previously trained
# classifier_file = os.path.join(model_dir,'emotion_classifier_v1_43.pkl')
emotion_model_path = os.path.join(model_dir, 'emotion_model.hdf5')

emotion_classifier = load_model(emotion_model_path)
face_cascade = cv2.CascadeClassifier(os.path.join(model_dir, 'haarcascade_frontalface_default.xml'))

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

emotion_labels = get_labels('fer2013')

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]


# starting lists for calculating modes
emotion_window = []

def predict_emotion(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        emotion_mode = emotion_mode + ' ' + str(emotion_probability)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return rgb_image


# # brief:    This function is created to load the classifier
# def load_classifier(classifier_filename):
#     print('loading model...')
#     if not os.path.exists(classifier_filename):
#         raise ValueError('Classifier Not Found!')
#
#     with open(classifier_filename, 'rb') as f:
#         model, class_names = pickle.load(f)
#         return model, class_names

# # brief :   Make predictions using the previously trained model & indicate names in picture
# # param :   emb_array: a list of face_encodings
# #           model and class_names: the dir to the pre-trained model
# #           image: a cv2 image
# #           rects: Dlib Detection Object
# def predict_classifier(emb_array, model, class_names, image, rects):
#     # Read model
#         if emb_array == []:
#             return image
#         # Make predictions on the face encodings
#         predictions = model.predict_proba(emb_array)
#
#         best_class_indices = np.argmax(predictions, axis=1)
#         best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#
#         for index,rect in enumerate(rects):
#             # Draw rectangle over faces detected
#             draw_rectangle(image, rect)
#
#
#             text = class_names[best_class_indices[index]]
#             prob = best_class_probabilities[index]
#             draw_text(image, text+' '+str(prob), rect)
#
#         return image


# # brief :   Draw a rectangle over a given face position
# # param :   img: a cv2 image
# #           rect: a Dlib Detection Object
# def draw_rectangle(img, rect):
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
#
#
# # brief :   Draw a text above a given face position, purpose: to indicate name
# # param :   img: a cv2 image
# #           text: string to write above the given face
# #           rect: a Dlib Detection Object
# def draw_text(img, text, rect):
#     x = rect.left()
#     y = rect.top()
#     cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),2)


# brief:    QtPy UI classv
class MainWindow(QWidget):
    # Class constructor
    def __init__(self):
        # Call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Create a timer
        self.timer = QTimer()
        # Set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # Set control_bt callback clicked  function
        self.ui.pushButton.clicked.connect(self.controlTimer)

    # View camera
    def viewCam(self):
        # model, class_names = load_classifier(classifier_file)
        # Read image in BGR format
        ret, image = self.cap.read()
        # Convert image to RGB format
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = face_detector(image, 0)  # Detect the faces in the image
        # shapes = [shape_predictor(image, face) for face in detections]
        # face_encodings = [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes]
        for k, d in enumerate(detections):  # For each detected face
            shape = shape_predictor(image, d)  # Get coordinates
            for i in range(1, 68):  # There are 68 landmark points on each face
                cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)

        # image = predict_classifier(face_encodings, model, class_names, image, detections)
        image = predict_emotion(image)

        # Get image infos
        height, width, channel = image.shape
        step = channel * width
        # Create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # Show image in img_label
        self.ui.graphicsView.setPixmap(QPixmap.fromImage(qImg))

    # Start/stop timer
    def controlTimer(self):
        # If timer is stopped
        if not self.timer.isActive():
            # Create video capture
            self.cap = cv2.VideoCapture(0)
            # Start timer
            self.timer.start(20)
            # Update control_bt text
            self.ui.pushButton.setText("Stop")
        # If timer is started
        else:
            # Stop timer
            self.timer.stop()
            # Release video capture
            self.cap.release()
            # Update control_bt text
            self.ui.pushButton.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())