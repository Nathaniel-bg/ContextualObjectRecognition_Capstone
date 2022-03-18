# Author: Nathaniel Basque-Giroux
# Carleton University 4th year Capston Project
# References: 
# Nicholas Renotte: Deep Drowsiness Detection using YOLO, Pytorch and Python, https://www.youtube.com/watch?v=tFNJGim3FXw 

from operator import mod
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
import argparse
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Function to collect pandas from single inference result
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
        print('no foolder exists with that name')
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
                save_folder = temp_save + '(' +str(i) + ')'
                save_folder_path = os.path.join(parent_save, save_folder)
            else:
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

def dynamic_filter():

    print('')


def process_data(model, total_counter, time_inference, SAVE_F_PATH):
        #create a counter of the full model name list
    print('Processing information')
    #make a basic save of the inference counter data
    print(total_counter)

    #Create a counter for all model names
    model_counter = Counter(model.names)
        #set all the starting values of model name counter to 0
    for values in model_counter.elements():
        model_counter[values] = 0
    #add the measured total values to the model counter list
    model_counter.update(total_counter)

    #plot and save the results
    SAVE_FOLDER_GRAPH = os.path.join(SAVE_F_PATH, 'Graphs')
    os.mkdir(SAVE_FOLDER_GRAPH)

    #data processing to be stored in csv
    objects = list(model_counter.keys())
    objects_detected = list(model_counter.values())
    total_detections = sum(objects_detected)
    object_percentage = []
    for val in objects_detected:
        object_percentage.append(round(val / total_detections, 4))
    
    #data processing to be displayed in graph
    graph_objects = list(total_counter.keys())
    graph_detected = list(total_counter.values())
    graph_percentage = []
    for val in graph_detected:
        graph_percentage.append(round(val / total_detections, 4))

    #Create a dataframe using the processed information
    d = {'Objects':objects, 'Objects Detected':objects_detected, 'percentage of detection':object_percentage,
    'inference time':time_inference, 'total detections':total_detections}
    df = pd.DataFrame(data=d)
    #Save the data to a csv in the data folder
    df.to_csv(SAVE_FOLDER_GRAPH + '\dataframe.csv')

    #Creating a horizontal bar plot using the built dataframe
    print('Creating Graph')
    #df.plot.barh(x='Objects', y='percentage of detection', figsize=(25,15))
    plt.figure(figsize=(25,15))
    plt.barh(graph_objects, graph_percentage)
    plt.xlabel('Percentage of Occurence')
    plt.ylabel('Model Objects')
    plt.savefig(SAVE_FOLDER_GRAPH + '\custommygraph.png')


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
            filter_type = 'Nate_basement'
            fc = filtered_classes(model, filter_type)
            print('filter ON: ' + filter_type)
            #Set the filtered classes
            model.classes = fc
            switch_filters = False

        #Apply model to the image/frame
        results = model(frame)

        #Collect information for inference
        #collect_data_pandas(results,False)
        
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
#               Processing Images - static filter
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

    file_pos = 1
    t0 = time.time()
    total_counter = Counter()
    while file_pos <= 100:#len(imgs):
        
        if switch_filters == True:
            #Obtain the filtered classes from model and filter
            fc = filtered_classes(model, 'City_Street')
            #Set the filtered classes
            model.classes = fc
            switch_filters = False
            
        results = model(imgs[file_pos-1])

        #Collect the results inference data
        name_count = collect_data_pandas(results, True)
        total_counter = total_counter + name_count
        
        #Save the inference results to path
        results.save(save_dir= SAVE_F_PATH)

        file_pos = file_pos +1

    t1 = time.time()
    #total inference time for the images
    time_inference = round(t1-t0, 4)
    print('Total inference time: ', time_inference)
    # Create a graph of the inference results
    process_data(model, total_counter, time_inference, SAVE_F_PATH)

#main run properties
def run(weights='yolov5s',
        type_detect='',
        filter=False,
        images_path='',
        conf_thres=0.25,
        iou_thres=0.45):

    if weights == 'yolov5s' or weights == 'yolov5m' or weights == 'yolov5l':
        model = torch.hub.load('ultralytics/yolov5', weights)
    #Run the custom model values
    if weights == 'custom':
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\Documents\CapstoneProject\ContextualObjectRecognition_Capstone\last.pt')
    
    model.cuda()
    model.conf = conf_thres
    model.iou = iou_thres
    switch_filters = filter
    
    if type_detect == 'images':
        process_images(model, switch_filters, images_path)
    
    if type_detect == 'video':
        live_video(model, switch_filters)
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s', help='model to use (yolov5, custom')
    parser.add_argument('--type-detect', default='video', help='images or video')
    parser.add_argument('--filter', type=bool, default=False, help='True/False for filter on/off')
    parser.add_argument('--images-path', type=str, help='Specify path for images, video')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    opt = parser.parse_args()
    return opt

def main(opt):
    #Create a run with desired variables
    run(**vars(opt))

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)