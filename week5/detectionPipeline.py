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



def getObjectsFromFrame(frame,mu,sigma,alpha, rho):

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

    if colorSpace != 'gray':
        outExtraDimension = np.stack([out, out, out], axis=-1)
        outFlat = outExtraDimension.ravel()
    else:
        outFlat = out.ravel()

    muFlat = mu.ravel()
    sigmaFlat = (sigma.ravel()) ** 2
    frameFlat = frame.ravel()

    muFlat = np.multiply(outFlat, muFlat) + np.multiply((rho * frameFlat + (1 - rho) * muFlat), (1 - outFlat))
    sigmaFlat = np.multiply(outFlat, sigmaFlat) + np.multiply((rho * (frameFlat - muFlat) ** 2 + (1 - rho) * sigmaFlat),
                                                              (1 - outFlat))
    sigmaFlat = np.sqrt(sigmaFlat)

    if colorSpace == 'gray':
        mu = muFlat.reshape(mu.shape[0], mu.shape[1])
        sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1])
    else:
        mu = muFlat.reshape(mu.shape[0], mu.shape[1], frame.shape[2])
        sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1], frame.shape[2])

    return out, mu, sigma



def getMuSigma(data,trainingRange):
    colorSpace = finalConf.colorSpace

    if finalConf.ID is 'Video':
        print 'Do something with empty frames.' # Frames without cars
        # MUST BE DONE
        mu = np.zeros_like(10).ravel()
        sigma = np.zeros_like(10).ravel()
        return mu, sigma
    else:

        frame = dataReader.getSingleFrame(data,0)

        mu = np.zeros_like(frame).ravel()
        sigma = np.zeros_like(frame).ravel()

        #Background estimation
        for idx in trainingRange:
            frame = dataReader.getSingleFrame(data, idx, False)
            if colorSpace != 'BGR':
                frame = cv2.cvtColor(frame, finalConf.colorSpaceConversion[colorSpace])
            mu = ((idx) * mu + frame.ravel())/float(idx + 1)

        for idx in trainingRange:
            frame = dataReader.getSingleFrame(data, idx, False)
            if colorSpace != 'BGR':
                frame = cv2.cvtColor(frame, finalConf.colorSpaceConversion[colorSpace])
            sigma = sigma + (frame.ravel() - mu) ** 2

        sigma = np.sqrt(sigma / len(trainingRange))

        if colorSpace == 'gray':
            mu = mu.reshape(frame.shape[0],frame.shape[1])
            sigma = sigma.reshape(frame.shape[0],frame.shape[1])
        else:
            mu = mu.reshape(frame.shape[0], frame.shape[1], frame.shape[2])
            sigma = sigma.reshape(frame.shape[0], frame.shape[1], frame.shape[2])

        return mu, sigma
