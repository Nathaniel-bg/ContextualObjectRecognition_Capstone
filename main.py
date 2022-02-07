# Author: Nathaniel Basque-Giroux
# Carleton University 4th year Capston Project
# References: 
# Nicholas Renotte: Deep Drowsiness Detection using YOLO, Pytorch and Python, https://www.youtube.com/watch?v=tFNJGim3FXw 

from statistics import mode
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import yaml
import os
import time
from collections import Counter

from zmq import device

def collect_data_pandas(results, collect = False):

    if collect == True:
        #Detect basic bounding box information and convert to pandas dataframe
        detected_data = results.pandas().xyxy[0]
        #detected information results (list)
        xmin= pd.Series(detected_data['xmin']),
        ymin= pd.Series(detected_data['ymin']),
        xmax= pd.Series(detected_data['xmax']),
        ymax= pd.Series(detected_data['ymax']),
        confidence= pd.Series(detected_data['confidence']),
        item_class= pd.Series(detected_data['class']),
        name= pd.Series(detected_data['name'])

        name_count = Counter(name)

        return(name_count)

def create_experiment_folder(parent_save, save_folder):
    run = os.listdir(parent_save)
    if len(run) == 0:
        #Create the folder path for the run
        save_folder_path = os.path.join(parent_save, save_folder)
    else:
        i = 1
        #temp save folder for base name of folder
        temp_save = save_folder
        for r in run:
            #Check to see if folder already exists
            if r == save_folder:
                #create a new folder name with an incrementing name
                save_folder = temp_save + str(i)
                save_folder_path = os.path.join(parent_save, save_folder)
            i = i+1
    print('Creating custom_runs save folder: ',save_folder_path)
    #Create the folder
    os.mkdir(save_folder_path)
    #Return the save path
    return save_folder_path

def read_yaml(filter_name):
    
    print('obtaining list of names...')
    #Load the yaml file
    with open('ContextFilters.yaml') as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

    names_list = dict[filter_name]['names']
    #Return a list of names
    print('done')
    return names_list

def filtered_classes(model, filter):

    print('Turning filter on')
    filter_names = read_yaml(filter)
    print('obtaining index position...')
    # Creating an empty list
    index_pos = []
    #for all filter values, compare to the model values
    for names in filter_names:
        index_pos.append(model.names.index(names))

    print('done')
    #Returns a list of classe positions
    return index_pos

def process_data(model, total_counter, time_inference, SAVE_F_PATH):
        #create a counter of the full model name list
    print('Creating Graph')
    model_counter = Counter(model.names)
        #set all the starting values of model name counter to 0
    for values in model_counter.elements():
        model_counter[values] = 0
    #add the measured total values to the model counter list
    model_counter.update(total_counter)

    #plot and save the results
    SAVE_FOLDER_GRAPH = os.path.join(SAVE_F_PATH, 'Graphs')
    os.mkdir(SAVE_FOLDER_GRAPH)

    #Create a horizontal bar plot of results
    objects = list(model_counter.keys())
    objects_detected = list(model_counter.values())
    total_detections = sum(objects_detected)
    object_percentage = []
    for val in objects_detected:
        object_percentage.append(round(val / total_detections, 4))
    
    #Creating a horizontal bar plot of inference results
    print('Creating Graph')
    plt.figure(figsize=(25,15))
    plt.barh(objects, object_percentage)
    plt.xlim([0,1])
    plt.xlabel('Percentage of Occurence')
    plt.ylabel('Model Objects')
    plt.savefig(SAVE_FOLDER_GRAPH + '\custommygraph.png')

    #Saving relevant data to textfile
    print('Saving Data')
    textfile = open(SAVE_FOLDER_GRAPH + '\Detection_Results.txt', 'w')
    textfile.write('Total inference time' + ' , ' + str(time_inference) + '\n')
    textfile.write('Total detections'+ ' , ' + str(total_detections) + '\n')
    for i, count in model_counter.items():
        textfile.write(str(objects[i]) + ', ' + str(objects_detected[i]) + ', ' + str(object_percentage[i])+ '\n')
    textfile.close()

###############################################################
#
#                      Live video capture
#
###############################################################        
def live_video(model, switch_filters):

    #Start video capture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if switch_filters == True:
            #Obtain the filtered classes from model and filter
            fc = filtered_classes(model, 'Nate_basement')
            #Set the filtered classes
            model.classes = fc
            switch_filters = False

        t0 = time.time()
        #Apply model to the image/frame
        results = model(frame)
        t1 = time.time()
        print('time for inference ', round(t1-t0, 4))

        #Collect information for inference
        collect_data_pandas(results,False)
        
        #Render the results onto the live view
        cv2.imshow('Contextual Object Detection', np.squeeze(results.render()))

        #Check for inputs during video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.waitKey(10) & 0xFF == ord('w'):
            print('turning filter off')
            model.classes = list(range(0, len(model.names)))

        if cv2.waitKey(10) & 0xFF == ord('e'):
            switch_filters = True
    cap.release()
    cv2.destroyAllWindows()

###############################################################
#
#                      Processing Images
#
###############################################################
def process_images(model, switch_filters, folder_dir):

    #Create a blank list to add all image paths
    imgs = []
    #Check for the name of each image
    file = os.listdir(folder_dir)
    for img in file:
        #create the full path for each image
        full_path = os.path.join(folder_dir, img)
        #Add the full path to the list
        imgs.append(full_path)
    print('images to process:', len(imgs))

    SAVE_F_PATH = create_experiment_folder('custom_runs', 'exp')

    file_pos = 0
    t0 = time.time()
    total_counter = Counter()
    while file_pos <= 100:# len(imgs):
        
        if switch_filters == True:
            #Obtain the filtered classes from model and filter
            fc = filtered_classes(model, 'City_Street')
            #Set the filtered classes
            model.classes = fc
            switch_filters = False
        results = model(imgs[file_pos-1])

        name_count = collect_data_pandas(results, True)
        total_counter = total_counter + name_count
        print(total_counter)
        
        results.save(save_dir= SAVE_F_PATH)


        file_pos = file_pos +1

    t1 = time.time()
    #total inference time for the images
    time_inference = round(t1-t0, 4)
    print('Total inference time: ', time_inference)

    # Create a graph of the inference results
    process_data(model, total_counter, time_inference, SAVE_F_PATH)

def main():
    # Set the model used for detection
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0', )
    #switch the model to use cuda drivers instead of cpu
    #model.cuda()
    # model.conf = 0.6
    # Determine if we want filter on/off
    switch_filters = True
    # Live video using webcam
    #live_video(model, switch_filters)
    #multi image processing
    folder_dir = 'D:\Documents\School\Fourth Year Carleton\Capstone\Alpha_demo_Initial_imgs'
    process_images(model, switch_filters, folder_dir)

if __name__ == "__main__":
    main()