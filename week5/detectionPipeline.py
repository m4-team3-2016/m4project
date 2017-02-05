import cv2
import sys
import numpy as np
import dataReader
sys.path.append("../")
sys.path.append('../tools')

for i in range(1,5):
    sys.path.append("../week" + str(i))

import configuration as conf
import week5configuration as finalConf

import holefillingFunction as hf
import shadowRemoval as sr
import morphology as mp
import bwareaopen as bw



def getObjectsFromFrame(frame,mu,sigma,alpha):

    colorSpace = finalConf.colorSpace
    # Background Substraction:
    if colorSpace != 'gray':
        out = np.abs(frame[:,:,0] - mu[:,:,0]) >= alpha * (sigma[:,:,0] + 2)
        for channel in range(1,frame.shape[2]):
            out = np.bitwise_or(np.abs(frame[:,:,channel] - mu[:,:,channel]) >= alpha * (sigma[:,:,channel] + 2),out)
    else:
        out = np.abs(frame[:,:] - mu[:,:]) >= alpha * (sigma[:,:] + 2)

    out = out.astype(np.uint8)

    # Hole filling
    if conf.isHoleFilling:
        out = hf.holefilling(out, conf.fourConnectivity)

    if conf.isAreFilling:
        out = bw.bwareaopen(out,conf.areaOptimal)


    # Morpholoy filters
    if conf.isMorphology:
        out = mp.apply_morphology_noise(out, conf.noise_filter_size)
        out = mp.apply_morphology_vertline(out, conf.vert_filter_size)
        out = mp.apply_morphology_horzline(out, conf.horz_filter_size)
    # Shadow removal
    if False:#conf.isShadowremoval:
        out = sr.inmask_shadow_removal(frame, out)
    # Shadow removal tends to remove some car components such as
    if False:#conf.isHoleFilling and conf.isShadowremoval:
        out = hf.holefilling(out, conf.fourConnectivity)

    return out

def getMuSigma(data,trainingRange):

    frame = dataReader.getSingleFrame(data,0)


    mu = np.zeros_like(frame).ravel()
    sigma = np.zeros_like(frame).ravel()
    nFrames = 0
    for idx in trainingRange:
        frame = dataReader.getSingleFrame(data,idx)

        mu = ((idx) * mu + frame.ravel())/float(idx + 1)
        nFrames +=1

    for idx in trainingRange:
        frame = dataReader.getSingleFrame(data,idx)
        sigma = sigma + (frame.ravel() - mu)**2

    sigma = np.sqrt(sigma / max(0,int(nFrames)))

    if finalConf.colorSpace == 'gray':
        mu = mu.reshape(frame.shape[0],frame.shape[1])
        sigma = sigma.reshape(frame.shape[0],frame.shape[1])
    else:
        mu = mu.reshape(frame.shape[0], frame.shape[1], frame.shape[2])
        sigma = sigma.reshape(frame.shape[0], frame.shape[1], frame.shape[2])

    return mu, sigma
