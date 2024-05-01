#!/root/miniconda3/envs/GsE/bin/python
import os
import subprocess
import sys
import distutils.core
sys.path.append('/root/autodl-tmp/GaussianVTON/ladi/preprocess')
os.chdir('ladi')
os.chdir('preprocess')

dist = distutils.core.run_setup("./detectron2/setup.py")
subprocess.run(['python', '-m', 'pip', 'install'] + dist.install_requires)

# Add the repository path to sys.path
repo_path = os.path.abspath('./detectron2')
sys.path.insert(0, repo_path)


import sys, os, distutils.core
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

os.chdir('detectron2/projects/DensePose')

#show
command_list = [
    'python', 'apply_net.py', 'show',
    'configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/images',
    'dp_segm', '-v',
    '--output', '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/dense/'
]
subprocess.run(command_list, check=True)

command_list = [
    'python', 'apply_net.py', 'show',
    'configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/images',
    'dp_segm', '-v',
    '--output', '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/dense/'
]
subprocess.run(command_list, check=True)

command_list = [
    'python', 'apply_net.py', 'show',
    'configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/images',
    'dp_segm', '-v',
    '--output', '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/dense/'
]
subprocess.run(command_list, check=True)

#dump
command_list = [
    'python', 'apply_net.py', 'dump',
    'configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/images',
    'dp_segm', '-v',
    '--output', '/root/autodl-tmp/GaussianVTON/ladi/input/upper_body/dense/'
]
subprocess.run(command_list, check=True)

command_list = [
    'python', 'apply_net.py', 'dump',
    'configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/images',
    'dp_segm', '-v',
    '--output', '/root/autodl-tmp/GaussianVTON/ladi/input/lower_body/dense/'
]
subprocess.run(command_list, check=True)

command_list = [
    'python', 'apply_net.py', 'dump',
    'configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/images',
    'dp_segm', '-v',
    '--output', '/root/autodl-tmp/GaussianVTON/ladi/input/dresses/dense/'
]
subprocess.run(command_list, check=True)

