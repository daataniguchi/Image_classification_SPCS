import os
import cv2 as cv
import collections
import numpy as np
from keras.models import load_model

def resize(img, resize_size=(150, 150), num_color_channels=3):
    # Resize
    # First resize while keeping aspect ratio
    rows,cols = img.shape[:2] # Define in input num_color_channels in case want black and white
    rc_ratio = rows/cols
    if resize_size[0] > int(resize_size[1]*rc_ratio):# if resize rows > rows with given aspect ratio
        img = cv.resize(img, (resize_size[1], int(resize_size[1]*rc_ratio))) #NB: resize dim arg are col,row
    else:
        img = cv.resize(img, (int(resize_size[0]/rc_ratio), resize_size[0]))

    # Second, pad to final size
    rows,cols = img.shape[:2] #find new num rows and col of resized image
    res = np.zeros((resize_size[0], resize_size[1], num_color_channels), dtype=np.uint8) #array of zeros
    res[(resize_size[0]-rows)//2:(resize_size[0]-rows)//2+rows,
        (resize_size[1]-cols)//2:(resize_size[1]-cols)//2+cols,:] = img # fill in image in middle of zeros

    return res/255.

label_map = {0:'Ciliate', 1:'L_Poly', 2:'Other'}

model = load_model('saved_models/model_dense-units=8_dropout=0.4_val-acc=0.86.h5')

root_dir = 'SPCP2_unlabeled_data_2020_bloom_no_bloom'

counts = collections.defaultdict(dict)
for root, dirs, files in os.walk(root_dir):
    if len(files) > 0:
        print(root)

        counts[root] = {'Ciliate': 0, 'L_Poly': 0, 'Other': 0, 'Total': 0}
        for f in files:
            img = cv.imread(os.path.join(root,f))
            img = resize(img)
            img = np.expand_dims(img, 0)
            out = model.predict(img)
            y_pred = np.argmax(out[0])
            counts[root][label_map[y_pred]] += 1
            counts[root]['Total'] += 1

for k,v in counts.items():
    print('{}:{}'.format(k,v))

for k,v in counts.items():
    for kv, vv in v.items():
        counts[k][kv] = float(vv)/counts[k]['Total']
    print('{}:{}'.format(k,counts[k]))





