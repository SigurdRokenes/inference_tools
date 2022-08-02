import argparse
import numpy as np
import pandas as pd
#from mean_average_precision import MetricBuilder
import os
import re
def json_to_df(json, conf_thresh = None):
    df = pd.read_json(json)

    if conf_thresh is not None:
        pred = df[df['score'] >= conf_thresh]
        return pred
    else:
        return df


def gt_converter(gt):
    df_col = gt['bbox']
    #ImageID,LabelName,XMin,XMax,YMin,YMax
    names = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']
    df = pd.DataFrame(index=range(df_col.size),columns=names)
    
    #df['XMin']

    for i, ele in enumerate(df_col):
        #xmin
        df['XMin'][i] = ele[0] / 1024 #input size just hardcoded since lazy
        #xmax
        df['XMax'][i] = ele[2] / 1024
        #ymin
        df['YMin'][i] = ele[1] / 1024
        #ymax
        df['YMax'][i] = ele[3] / 1024
        #df[1][i] = (ele[0] + ele[2]) / 1024
        #df[3][i] = (ele[1] + ele[3]) / 1024
    df['ImageID'] = gt['image_id']
    df['LabelName'] = 'Porpoise'
    return df


def pred_converter(pred):
    ###Input: xmin, ymin, w, h
    ###Output: xim,xmax,ymin,ymax (between [0,1])
    try:
        df_col = pred['bbox']
        names = ['ImageID', 'LabelName','Conf', 'XMin', 'XMax', 'YMin', 'YMax']
        df = pd.DataFrame(index=range(df_col.size),columns=names)
        #find index of name from first '_'
        txt = pred['image_id'][0]
        idx = txt.find(r'_') + 1

        for i, ele in enumerate(df_col):
            #xmin
            df['XMin'][i] = ele[0] / 1024 #input size just hardcoded since lazy
            #xmax = xmin + w
            df['XMax'][i] = (ele[0] + ele[2])/ 1024
            #ymin
            df['YMin'][i] = ele[1] / 1024
            #ymax
            df['YMax'][i] = (ele[1] + ele[3]) / 1024

            df['ImageID'][i] = pred['image_id'][i][idx:]

        #print(txt[idx:])
        
        #df['ImageID'] = pred['image_id'][idx:]
        df['LabelName'] = 'Porpoise'
        df['Conf'] = pred['score']

        return df.sort_values(by=['ImageID'])
    except:
        print("No valid inputs in {}".format(pred))


def main():

    #model = "efficientDet_d1"
    #model = "faster_rcnn"
    model = "yolov4_1024"
    def pred():
        input_dir = os.path.join('result_json',model) 
        
        input_list = os.listdir(input_dir)
        for input_path in input_list:
            input = os.path.join(input_dir, input_path)

            try:
                pred = json_to_df(input)#.sort_values(by=['image_id'])
                pred_fixed = pred_converter(pred)
                result_name = input_path[:-4]+'.csv'
                pred_fixed.to_csv(os.path.join('result_csv', model+'_'+result_name), sep = ",", index=False)

            except:
                print('Input not valid for {}'.format(input))




    def gt():
        gt_df = json_to_df("result_json/Annotations_1024.json")
        gt = gt_converter(gt_df)
        gt.to_csv('groundtruth.csv', sep = ",", index=False)
    
    #gt()
    pred()
    print("Process complete")
if __name__ == "__main__":
    main()