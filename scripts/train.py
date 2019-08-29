import dlib
import numpy as np
import os
import pickle
import cv2
import time
import sys
from random import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

start_time = time.time()

# Hardcoded directory to model file
model_dir = '/Users/kokhungchan/Desktop/emotion_recognition/model'
# Hardcoded input directory
input_dir = '/Users/kokhungchan/Desktop/emotion_recognition/fer2013_data_v2'

# Declaring dlib models to detect face and generate face encodings
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
emotion_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))

# Name of model to train and evaluate
classifier_filename = os.path.join(model_dir,'emotion_classifier_v3.pkl')

# Minimum number of images per class
min_nrof_image_per_class = 10


# brief :   Shows the current progress of the program
# param :   count: an int that represents current progress
#           total: an int that represents total work to be done
#           status: a string that describe the current progress
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%50s' % (bar, percents, '%', status))
    sys.stdout.flush()


# brief :   Generates face_encodings
# param :   image: a cv2 image
# return:   a list of face encodings for all detected faces in image
def get_face_encodings(image):

    detected_faces = face_detector(image,1)

    shaped_faces = [shape_predictor(image,face) for face in detected_faces]

    return [np.array(emotion_recognition_model.compute_face_descriptor(image,face_pose,1)) for face_pose in shaped_faces]


# brief :   Read the images in input dir and format it nicely into a list
# param :   input_dir: absolute path to the input directory
# return:   data_array: a list that contains all the encodings and labels
#           total: an int that indicates how many images were supposed to process
#           lost_counter: an int that indicates how many images were not processed
def get_data(input_dir):
    lost_counter = 0
    total = 0
    data_array = []
    class_names = get_class_names(input_dir)

    for dir in os.listdir(input_dir):
        if not dir.startswith('.'):
            for image_path in os.listdir(os.path.join(input_dir, dir)):
                if len(os.listdir(os.path.join(input_dir,dir))) <min_nrof_image_per_class:
                    continue
                else:
                    if not image_path.startswith('.'):
                        full_path_to_image = os.path.join(input_dir, dir)
                        full_path_to_image = full_path_to_image + '/' + image_path
                        if image_path.endswith('.jpg'):
                            image = cv2.imread(full_path_to_image)
                            total += 1

                            status = str(total) + '/' + str(get_total_image(input_dir)) + ' ' + image_path + ' ' + dir
                            progress(total, get_total_image(input_dir), status=status)
                            emb = get_face_encodings(image)
                            if len(emb) <1:
                                status = 'Error: ' + str(total) + '/' + str(get_total_image(input_dir)) + ' ' + image_path + ' ' + dir
                                progress(total, get_total_image(input_dir), status=status)
                                lost_counter+=1
                                continue
                            data_array.append([emb[0],class_names.index(dir)])


    return data_array, total, lost_counter


# brief :   Split the data into test and train set
# param :   dataset: a list that contains all the dataset
#           split_ratio: ratio of train dataset
# return:   train_emb: a list of face encodings to train on
#           train_label: a list of int that act as labels to the face encodings
#           test_emb: a list of face encodings to test on
#           test_label: a list of int that act as answers to the test data
def split_test_train_set(dataset, split_ratio = 0.9):
    shuffle(dataset)
    train_emb = []
    train_label = []
    test_emb = []
    test_label = []

    progress(100, 100, status='Splitting Test & Train Set')

    split = int(round(len(dataset) * split_ratio))
    train_set = dataset[0:split]
    test_set = dataset[split:]

    for data in train_set:
        train_emb.append(data[0])
        train_label.append(data[1])

    for data in test_set:
        test_emb.append(data[0])
        test_label.append(data[1])

    return train_emb, train_label, test_emb, test_label


# brief :   Goes through the input dir and get a list of qualified names, qualified means > 10 images
# param :   input_dir: the absolute path to the training data directory
# return:   names: a sorted list of qualified names
def get_class_names(input_dir):
    names = []
    for dir in os.listdir(input_dir):
        if not dir.startswith('.'):
            dir_path = os.path.join(input_dir,dir)
            if len(os.listdir(dir_path)) < min_nrof_image_per_class:
                continue
            names.append(dir)

    names.sort()
    return names


# brief :   Calculate the total number of images to process, purpose to track progress
# param :   input_dir: the absolute path to the training data directory
# return:   an int that represents the total images to process
def get_total_image(input_dir):
    total = 0
    for dir in os.listdir(input_dir):
        if not dir.startswith('.'):
            dir_path = os.path.join(input_dir,dir)
            if len(os.listdir(dir_path)) < min_nrof_image_per_class:
                continue
            total += len(os.listdir(dir_path))
    return total


# brief :   Train the model and save it
# param :   emb_array: face encodings to train on
#           label_array: labels to the face encodings
#           classifier_filename: name to save the trained model
#           class_names: a list of names of qualified directories
def train_and_save_classifier(emb_array, label_array, classifier_filename, class_names):
    status = 'Training on ' + str(len(emb_array)) + ' embeddings'
    progress(100, 100, status=status)
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)

    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)


# brief :   Evaluate the accuracy of the model on unseen data
# param :   emb_array: face encodings to evaluate using the pre-trained model
#           label_array: answers to the emb_array, purpose: to check whether prediction is correct
#           classifier_filename: path to the pre-trained model
def evaluate_classifier(emb_array, label_array, classifier_filename):
    status = 'Evaluating on ' + str(len(emb_array)) + ' embeddings'
    progress(100, 100, status=status)
    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        accuracy = model.score(emb_array,label_array)
        print('\nAccuracy: %.3f' % accuracy)


def confusion_report(emb_array, label_array, classifier_filename):
    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        report = confusion_matrix(label_array, best_class_indices.tolist())
        print('Confusion Matrix :')
        print(report)

        for i,name in enumerate(class_names):
            print('%d. %s' % (i, name))



    # MAIN
# get the valid classes from input dir
classes = get_class_names(input_dir)
# read the images in input dir and format it
dataset, total_img, lost_img = get_data(input_dir)
# split the dataset into test and train set
train_emb, train_label, test_emb, test_label = split_test_train_set(dataset)
# train the model using the train set
train_and_save_classifier(train_emb, train_label, classifier_filename, classes)
# evaluate the model using the test set
evaluate_classifier(test_emb, test_label, classifier_filename)
# generate confusion report
confusion_report(test_emb, test_label, classifier_filename)
print('Total: {}' .format(total_img))
print('Lost: {}' .format(lost_img))
print('Completed in {} seconds'.format(time.time() - start_time))
