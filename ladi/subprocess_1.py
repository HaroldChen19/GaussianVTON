#!/root/miniconda3/envs/GsE/bin/python
import os
import subprocess
import sys
sys.path.append('/root/autodl-tmp/GaussianVTON/ladi/preprocess/pytorch_openpose')
os.chdir('ladi')
os.chdir('preprocess')
os.chdir('pytorch_openpose')
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob
import json

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('/root/autodl-tmp/GaussianVTON/ladi/preprocess/pytorch_openpose/model/body_pose_model.pth')
for s in ['upper_body','lower_body','dresses']:
    input_path = '/root/autodl-tmp/GaussianVTON/ladi/input/' + s + '/images/'
    output_path = '/root/autodl-tmp/GaussianVTON/ladi/input/'+ s + '/skeletons/'
    keypoint_path = '/root/autodl-tmp/GaussianVTON/ladi/input/'+ s + '/keypoints/'

    pattern = os.path.join(input_path, '*')

    # for images in glob.glob('*', root_dir = input_path):
    for images in glob.glob(pattern):
        oriImg = cv2.imread(images) #BGR order
        images = os.path.basename(images)
        candidate, subset = body_estimation(oriImg)
        canvas = util.draw_bodypose(np.zeros_like(oriImg), candidate, subset)
        arr = candidate.tolist()
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        for i in range(0,18):
            if len(arr)==i or arr[i][3] != vals[i]:
               arr.insert(i, [-1, -1, -1, vals[i]])

        keypoints = {'keypoints':arr[:18]}
        cv2.imwrite(output_path +images.replace('_0','_5'),canvas)
        with open(keypoint_path + os.path.splitext(images)[0].replace('_0','_2') + ".json", "w") as fin:
            fin.write(json.dumps(keypoints))