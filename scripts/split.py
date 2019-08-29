import csv
import os

legend_dir = '/Users/seanlai/Documents/ai/emotion_recognition/data/legend.csv'
input_dir = '/Users/seanlai/Documents/ai/emotion_recognition/images/'
output_dir = '/Users/seanlai/Documents/ai/emotion_recognition/training_data/'

with open(legend_dir, 'r') as f:
    reader = csv.reader(f)

    for i, row in enumerate(reader):
        filename = row[1]
        emotion = row[2]

        file_dir = input_dir + filename
        if os.path.isfile(file_dir):
            output_name = output_dir + emotion + '/' + filename
            os.rename(file_dir,output_name)
