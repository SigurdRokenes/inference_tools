import pandas as pd
import numpy as np
import sys
import os

DATASET_NAME = 'Porpoises'
IMAGE_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test/images/'
###Faster RCNN
"""
RESULT_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/fasterrcnn_testresults/'
RESULT_NAME = 'result.json'
SAVE_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/fasterrcnn_testresults/yolo_format/'
"""
### YOLO
RESULT_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/yolov4_testresults/annotations/'
RESULT_NAME = 'coco_results.json'
SAVE_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/yolov4_testresults/annotations/absolute_format/'

THRESHOLD = 0.45

SAVE = True

print(os.getcwd())
class JsonWriter:
    def __init__(self):
        print(RESULT_PATH+RESULT_NAME)
        try:
            self.df = pd.read_json(RESULT_PATH+RESULT_NAME)
        except:
            print('File not found. Exiting...')
            exit()


    def find_char_location(self, original, character):
        return [index for index, char in enumerate(original) if char == character]


    def find_image_name(self, image):

        path_index = self.find_char_location(image, '/')[-1] + 1
        #path = image[:path_index]
        image_name = image[path_index:]
        return image_name


    def populate_annotations(self, arr, threshold):
        #id, category_id, bbox = (0,0,0)

        detections = self.df[arr]

        valid_detections = detections[detections['score'] >= threshold]
        #nr of valid detections
        #n = len(valid_detections)
        #coordinates
        bbox = [coords for coords in valid_detections['bbox']]
        confidence = [conf for conf in valid_detections['score']]
        class_id = [cl_id-1 for cl_id in valid_detections['category_id']]

        return class_id, confidence, bbox


    def write_yolo_file(self, name, id, conf, bbox, save_path=SAVE_PATH):
        n = len(id)
        with open(save_path+name[:-4]+'.txt', 'w') as file:
            for i in range(n):
                left, top, width, height = bbox[i]

                file.write('{} {} {} {} {} {} \n'.format(id[i], round(conf[i], 4), round(left),
                                                         round(top), round(width), round(height)))



    def __call__(self):
        image_ids = np.unique(self.df['image_id'])
        n_img = len(image_ids)
        print('Processing image:')
        for index, image in enumerate(image_ids):
            sys.stdout.write('\r {}%'.format(round((index+1)/n_img * 100)))

            image_name = self.find_image_name(image)

            class_id, confidence, bbox = self.populate_annotations(arr=(self.df['image_id'] == image),
                                                                   threshold=THRESHOLD)
            if SAVE:
                self.write_yolo_file(image_name, class_id, confidence, bbox)
            sys.stdout.flush()
        #self.populate_dictionary()


def main():
    writer = JsonWriter()
    writer()


if __name__ == "__main__":
    main()


