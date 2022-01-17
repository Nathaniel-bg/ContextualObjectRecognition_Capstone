# Author: Nathaniel Basque-Giroux
# Carleton University 4th year Capston Project
# References: 
# Nicholas Renotte: Deep Drowsiness Detection using YOLO, Pytorch and Python, https://www.youtube.com/watch?v=tFNJGim3FXw 

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import yaml

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

        print(name)

def read_yaml(filter_name):
    #Load the yaml file
    print('obtaining list of names...')
    with open('ContextFilters.yaml') as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

    names_list = dict[filter_name]['names']
    #Return a list of names
    print('done')
    return names_list


def filtered_classes(model, filter):
    
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
        

def main_live():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    switch_filters = True

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

        #Apply model to the image/frame
        results = model(frame)
        
        #Collect information for inference
        collect_data_pandas(results)
        
        #Render the results onto the live view
        cv2.imshow('Contextual Object Detection', np.squeeze(results.render()))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_live()