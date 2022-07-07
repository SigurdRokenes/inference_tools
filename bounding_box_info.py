## Find number of each class
import os
import struct
import numpy as np
import pandas as pd
import io
from PIL import Image

import os
import struct


path = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test/annotations/'
image_path = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test/images/'
labels = os.listdir(path)
i = 0
porpoise = 0
mc = 0

small_thresh = 32**2
med_thresh = 96**2

small = 0
med = 0
large = 0

large_mc = 0
med_mc = 0

for doc in labels:
    #Get image dimensions
    im_path = image_path+doc[:-3]+'jpg'
    im = Image.open(im_path)
    im_w, im_h = im.size

    #work on labels:
    label_path = path+doc
    #for testing
    if i > 50:
        break
    

    file = np.asarray(pd.read_csv(label_path, sep=' ', header=None))
    
    cls = file[0:, 0]

    porpoise += len(cls[cls==0])
    #print(len(cls[cls==0]))
    mc += len(cls[cls==1])

    #x = file[:, 1]
    #y = file[:, 2]
    w = file[:, 3] * im_w
    h = file[:, 4] * im_h
    for i in len(w):
        classification = cls[i]
        area = w[i]*h[i]

        if area <= small_thresh:
            small += 1

        elif area > small_thresh and area <= med_thresh:
            med += 1
            if classification == 1:
                med_mc += 1
            
                
        else:
            large += 1
        
    #area = w * h

    print(area)
    """
    
    #for index in w:

    """
    #w = 
    #width = 

    #print('index: ', i)
    #print(cls)
    #print(file[0][0])
    #if cls == 0:
    #    porpoise+=1

    #if cls == 1:
    #    mc += 1
    #with open(label_path, 'r') as file:
        #print(file)
    i += 1
#print(len(labels))
print('total porpoise: ', porpoise)
print('total mc: ', mc)

