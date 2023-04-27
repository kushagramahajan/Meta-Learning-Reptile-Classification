import csv
import cv2

with open('/home/ilab/Downloads/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    	#print(row)
    	if(line_count == 0):
    		line_count+=1
    		continue
    	image_name = row[0]
    	
    	label0 = int(float(row[1]))
    	label1 = int(float(row[2]))
    	label2 = int(float(row[3]))
    	label3 = int(float(row[4]))
    	label4 = int(float(row[5]))
    	label5 = int(float(row[6]))
    	label6 = int(float(row[7]))
    	print(label0, label1, label2, label3, label4, label5, label6)

    	if(label0 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		print('image read')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class1/'+image_name+'.jpg', img)
    		print('image written')
    	if(label1 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class2/'+image_name+'.jpg', img)
    	if(label2 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class3/'+image_name+'.jpg', img)
    	if(label3 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class4/'+image_name+'.jpg', img)
    	if(label4 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class5/'+image_name+'.jpg', img)
    	if(label5 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class6/'+image_name+'.jpg', img)
    	if(label6 == 1):
    		img = cv2.imread('/home/ilab/Downloads/ISIC2018_Task3_Training_Input/'+image_name+'.jpg')
    		cv2.imwrite('/home/ilab/Downloads/ISIC2018_Task3_Training_class/class7/'+image_name+'.jpg', img)


    	line_count+=1

    print(f'Processed {line_count} lines.')