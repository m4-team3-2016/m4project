import cv2
import os
import numpy as np
import json
from numpy import trapz
import sys
import glob
sys.path.append('../week2')
import evaluation as ev
import bwareaopen

dataset = '../../../datasets-week3/highwayDataset/'
files = glob.glob(dataset + '*')
ID = "Highway"
#cv2.imshow('test',cv2.imread(files[0]))
results = dict()
TP = []
TN = []
FP = []
FN = []
Precision = []
Recall = []
F1  = []

for p in range(0,1000,20):
    print "Filtering objects smaller than " + str(p)
    for f in files:
        cv2.imwrite('results/areaFiltering/' + ID + f.split('/')[-1],bwareaopen.bwareaopen(files[0],p))
        #print 'results/areaFiltering/' + f.split('/')[-1]
    tp,tn,fp,fn,precision,recall,f1 = ev.evaluateFolder('results/areaFiltering/',ID)
    for f in glob.glob('results/areaFiltering/*'):
        os.remove(f)
    print tp,tn,fp,fn,precision,recall,f1
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    Precision.append(precision)
    Recall.append(recall)
    F1.append(f1)

results['TP'] = TP
results['TN'] = TN
results['FP'] = FP
results['FN'] = FN
results['Precision'] = Precision
results['Recall'] = Recall
results['F1'] = F1



with open('results/resultsAreaFiltering' + ID + '.json','w') as f:
    json.dump(results,f)
