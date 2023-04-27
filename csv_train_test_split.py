import csv, os
import cv2
import glob, random

all_files = glob.glob('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class7/*.jpg')
random.shuffle(all_files)


total_files = len(all_files)
num_train = 0.75*total_files
num_test = 0.25*total_files

counter = 0
for file in all_files:
    if(counter<num_train):
        img = cv2.imread(file)
        cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class7/train/'+file.split('/')[-1], img)
    else:
        img = cv2.imread(file)
        cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class7/test/'+file.split('/')[-1], img)
    
    counter+=1
