import pandas as pd
import json
import re

input = "Labels/result_efficientDetd1.json"
output = "objMet_Labels/annotations.json"

annotations = False
WRITE_TO_FILE = False

def json_to_df(json, conf_thresh = None):
    df = pd.read_json(json)

    if conf_thresh is not None:
        pred = df[df['score'] >= conf_thresh]
        return pred
    else:
        return df

if annotations:
    df = json_to_df(input)
else:
    df = json_to_df(input, conf_thresh=0.1)
    confidence = df['score']




labels = df['image_id'].to_list()
bbox = df['bbox'].to_list()
category_id = df['category_id'].to_list()

json_result = []

print(labels[0])
file_number = re.findall(r'\d+', labels[0])[0]
print(type(int(file_number)))
for i in range(len(labels)):
    id = re.findall(r'\d+', labels[i])[0]

    json_result.append({
        'id': id
        "category_id": category_id[i]
        "bbox": [bbox]
    })
if WRITE_TO_FILE:

    #print(len(json_result))
    with open(output, 'w') as outfile:
        outfile.write('[')
        outfile.write('\n')
        for i in range(len(json_result)):
            json.dump(json_result[i], outfile, separators=(',', ':'))
            if i < len(json_result) - 1:
                outfile.write(',\n')
            else:
                outfile.write('\n')
        outfile.write(']')

    print('Writing to file complete.')
