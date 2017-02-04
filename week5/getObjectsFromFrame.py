import cv2
import sys
import numpy as np

sys.path.append('../')
sys.path.append('../tools')

for i in range(1,5):
    sys.path.append('../week' + str(i))

import configuration as conf
import week5configuration as finalConf

import holefillingFunction as hf
import shadowRemoval as sr
import morphology as mp


def getObjectsFromFrame(frame,mu,sigma,alpha):
    #TODO: FIX THIS
    colorSpace = 'gray'
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

def getMuSigma(vid):
    colorSpace = 'gray'
    ret,frame = vid.read()
    vid.set(1,0)

    if colorSpace != 'BGR':
        frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])

    mu = np.zeros_like(frame).ravel()
    sigma = np.zeros_like(frame).ravel()
    nFrames = 500
    for idx in range(nFrames):
        ret,frame = vid.read()
        if colorSpace != 'BGR':
            frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])
        mu = ((idx) * mu + frame.ravel())/float(idx + 1)

    for idx in range(0,max(0,int(nFrames))):
        ret,frame = vid.read()
        if colorSpace != 'BGR':
            frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])
        sigma = sigma + (frame.ravel() - mu)**2

    sigma = np.sqrt(sigma / max(0,int(nFrames)))

    if colorSpace == 'gray':
        mu = mu.reshape(frame.shape[0],frame.shape[1])
        sigma = sigma.reshape(frame.shape[0],frame.shape[1])
    else:
        mu = mu.reshape(frame.shape[0], frame.shape[1], frame.shape[2])
        sigma = sigma.reshape(frame.shape[0], frame.shape[1], frame.shape[2])

    return mu, sigma