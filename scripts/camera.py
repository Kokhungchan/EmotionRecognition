
import dlib
import numpy as np
import os
import pickle
import cv2
import time


# Hardcoded the directory path to models
# Change this path to your model directory
model_dir = '/Users/kokhungchan/Desktop/emotion_recognition/model'

# Declaring dlib models to detect face and generate face encodings
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))
face_detector = dlib.get_frontal_face_detector()

# Name of the predictor model previously trained
classifier_file = os.path.join(model_dir,'emotion_classifier_v8.pkl')


def load_classifier(classifier_filename):
    print('loading model...')
    if not os.path.exists(classifier_filename):
        raise ValueError('Classifier Not Found!')

    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)
        return model, class_names


# brief :   Make predictions using the previously trained model & indicate names in picture
# param :   emb_array: a list of face_encodings
#           classifier_filename: the dir to the pre-trained model
#           image: a cv2 image
#           rects: Dlib Detection Object
def predict_classifier(emb_array, model, class_names, image, rects):

    if emb_array == []:
        return image
    # Make predictions on the face encodings
    predictions = model.predict_proba(emb_array)
    
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    
#    for i in range(len(best_class_indices)):
#        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

    for index,rect in enumerate(rects):
        # Draw rectangle over faces detected
        draw_rectangle(image, rect)
        
        # Draw name of faces
        name = class_names[best_class_indices[index]]
        prob = best_class_probabilities[index]
        draw_text(image, name+' '+str(prob), rect)
    return image


# brief :   Generates face encodings of an image
# param :   path_to_image: absolute path to an image
# return:   face_encodings: a list of np array storing face encodings in 128 dimensions
#           test_img: a copy of the target image in cv2 format, purpose: to not overwrite original image
#           detected_faces: a list of Dlib Detection Object that provides the position of the faces
def get_face_encodings(frame):
    detected_faces = face_detector(frame,0)
    shaped_faces = [shape_predictor(frame,face) for face in detected_faces]
    face_encodings = [np.array(face_recognition_model.compute_face_descriptor(frame,face_pose,1)) for face_pose in shaped_faces]
    
    return face_encodings, frame, detected_faces


# brief :   Draw a rectangle over a given face position
# param :   img: a cv2 image
#           rect: a Dlib Detection Object
def draw_rectangle(img, rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)


# brief :   Draw a text above a given face position, purpose: to indicate name
# param :   img: a cv2 image
#           text: string to write above the given face
#           rect: a Dlib Detection Object
def draw_text(img, text, rect):
    x = rect.left()
    y = rect.top()
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),2)


# MAIN
model, class_names = load_classifier(classifier_file)

# brief: open camera and run prediction on every frame
cap = cv2.VideoCapture(0)
print('predicting...')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    face_encodings, image, rects = get_face_encodings(frame)
    
    gray = predict_classifier(face_encodings, model, class_names, image, rects)
    # Display the resulting frame
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


