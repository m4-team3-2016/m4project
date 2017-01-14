import cv2
import os
import numpy as np
import json
from numpy import trapz
import sys
import glob

sys.path.append('../week2')
import evaluation as ev

dataset = '../../../highwayDataset/'

files = glob.glob(dataset + '*')
