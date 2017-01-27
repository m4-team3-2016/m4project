# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:53:22 2017

@author: Group 3
"""

import cv2
import numpy as np
import sys

def create_compensated_image(prev_img, motion_matrix, block_size, x_blocks, y_blocks):
    x_size = prev_img.shape[0]
    y_size = prev_img.shape[1]

    comp_img = np.zeros([x_size, y_size])

    for x_pos in range(x_blocks):
        for y_pos in range(y_blocks):
            comp_img[x_pos*block_size:x_pos*block_size+block_size,y_pos*block_size:y_pos*block_size+block_size]=prev_img[x_pos*block_size-motion_matrix[x_pos,y_pos,0]:x_pos*block_size+block_size-motion_matrix[x_pos,y_pos,0],y_pos*block_size+motion_matrix[x_pos,y_pos,1]:y_pos*block_size+block_size+motion_matrix[x_pos,y_pos,1]]
    return comp_img

def compute_error(block1, block2):
    return sum(sum(abs(block1-block2)**2))

def block_search(region_to_explore, block_to_search, block_size):
    x_size = region_to_explore.shape[0]
    y_size = region_to_explore.shape[1]

    min_diff = sys.float_info.max

    for row in range(x_size-area_size):
        for column in range(y_size-area_size):
            block2analyse = region_to_explore[row:row+block_size, column:column+block_size]
            diff = compute_error(block2analyse, block_to_search)
            if diff < min_diff:
                min_diff = diff
                x_mot = - row + area_size
                y_mot = column - area_size
    return x_mot, y_mot
    
def compute_block_matching(prev_img, curr_img, block_size, area_size, compensation):
    #We will apply backward compensation
    if compensation=='backward':
        img2xplore = curr_img
        searchimg = prev_img
    else:
        img2xplore = prev_img
        searchimg = curr_img
    
    x_blocks = img2xplore.shape[0]/block_size
    y_blocks = img2xplore.shape[1]/block_size

    #Add padding in the search image
    pad_searchimg = np.zeros([img2xplore.shape[0]+2*area_size,img2xplore.shape[1]+2*area_size])
    pad_searchimg[area_size:area_size+img2xplore.shape[0],area_size:area_size+img2xplore.shape[1]] = searchimg[:,:]

    motion_matrix = np.zeros([x_blocks, y_blocks, 2])
    
    for row in range(x_blocks):
        for column in range(y_blocks):
            print "Computing block " + str(column)
            block_to_search = img2xplore[row*block_size:row*block_size+block_size, column*block_size:column*block_size+block_size]
            region_to_explore = pad_searchimg[row*block_size:row*block_size+block_size+2*area_size, column*block_size:column*block_size+block_size+2*area_size]
            x_mot, y_mot = block_search(region_to_explore, block_to_search, block_size)
            
            motion_matrix[row,column,0] = x_mot
            motion_matrix[row,column,1] = y_mot
            
    return motion_matrix, x_blocks, y_blocks
    
if __name__ == "__main__":
    # 1241 x 376 --> 155 x 47
    sequence = 45 #157
    folder = 'image_0/'
    prev_img = cv2.imread(folder + '0000'+str(sequence)+'_10.png')
    curr_img = cv2.imread(folder + '0000'+str(sequence)+'_11.png')
    
    # We resize from 1241 to 1240 in order to obtain an even number of pixels
    # We choose the color channel 0 because is the same as the others
    prev_img = np.array(prev_img)
    prev_img = prev_img[:,0:1240,0]
    curr_img = np.array(curr_img)
    curr_img = curr_img[:,0:1240,0]
    
    block_size = 8
    area_size = 16
    compensation = 'backward' #or 'forward'
        
    #Apply block matching
    motion_matrix, x_blocks, y_blocks = compute_block_matching(prev_img, curr_img, block_size, area_size, compensation)
    
    if compensation=='backward':
        prev_img = prev_img
    else:
        prev_img = curr_img

    comp_img = create_compensated_image(prev_img, motion_matrix, block_size, x_blocks, y_blocks)
    cv2.imwrite('compensated.jpg', comp_img)
    
    #cv2.line(curr_img, (x[0], y[0]), (x[-1], y[-1]), (0,0,0))
    #cv2.imshow("foo",img)
    #cv2.waitKey()
