"""
Problem statement:
Given a starting point in an image, keep adding points from the image to it that fulfill a certain condition.
Step 1: condition is that the pixels have a value greater than a given threshold
"""
from pprint import pprint

import numpy as np
from skimage import io
from queue import Queue

def display_image(image, cmap='gray'):
    io.imshow(np.array(image), cmap=cmap)
    io.show()

def display_img_collection(collection):
    io.imshow_collection(collection)
    io.show()

#-----------------------------------------------------------------------------------------------------------------------
def update_region_pixels_mean_intensity(old_mean, new_entry):
    if (old_mean == 0):
        return new_entry
    return (old_mean+new_entry)/2


def region_growing_s1(image, seed, threshold):

    region = image.copy()
    region[:,:] = 0  # blacken out the image

    region_pixel_intensities = []

    # build a queue to hold all the pixels from the image to be processed
    q = Queue()
    q.put(seed)  # initially it only contains the seed pixel

    orientations = ([-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1])

    # initialization -> could be optimized to be done directly in the while, i guess
    cpx = seed[0]
    cpy = seed[1]
    region[cpx, cpy] = 255  # whiten the pixels that belong to the segmented region
    region_pixel_intensities.append(image[cpx, cpy])
    # add the neighbors of the current pixel to the queue (only if the current pixel is part of the region)
    for o in orientations:
        neigh_line = o[0] + cpx
        neigh_col = o[1] + cpy
        is_in_image = image.shape[0] > neigh_line >= 0 and image.shape[1] > neigh_col >= 0
        if (is_in_image):
            not_visited = region[
                              neigh_line, neigh_col] == 0  # to use a third matrix since probably using region doesn't cover all the cases
            if (not_visited):
                q.put([neigh_line, neigh_col])

    t = 0
    while not q.empty():
        t=t+1
        if (t%30000==0):
            print(t)
            print ("Queue size: %d", q.qsize())
            # display_image(region)
        current_point = q.get()
        cpx = current_point[0]
        cpy = current_point[1]
        if (region[cpx, cpy]==0):
            # check if the current point can be added to the region
            current_pixel_intensity = image[cpx, cpy]
            region_pixels_mean_intensity = np.mean(region_pixel_intensities)
            if (abs(region_pixels_mean_intensity-current_pixel_intensity)<threshold):   # the condition for adding a new pixel to the region
                region[cpx, cpy] = 255  # whiten the pixels that belong to the segmented region
                region_pixel_intensities.append(image[cpx, cpy])
                # add the neighbors of the current pixel to the queue (only if the current pixel is part of the region)
                for o in orientations:
                    neigh_line = o[0] + cpx
                    neigh_col = o[1] + cpy
                    is_in_image = image.shape[0]>neigh_line>=0 and image.shape[1]>neigh_col>=0
                    if (is_in_image):
                        not_visited = region[neigh_line, neigh_col] == 0  # to use a third matrix since probably using region doesn't cover all the cases
                        if (not_visited):
                            q.put([neigh_line, neigh_col])

    return region


def with_simple_test_image():
    image = [[0, 0,   0,   0,   100, 150, 111],
             [110, 100, 225, 100, 100, 0,   0],
             [0, 0,   100, 100, 210, 0,   0],
             [0, 0,   0,   60,  0,   0,   0]]

    image = np.array(image)
    region = np.array(region_growing_s1(image, [2,3], 100))
    display_img_collection([image, region])

def with_real_image(filename):
    image_real = io.imread(filename)
    region = np.array(region_growing_s1(image_real, [100, 10], 50))
    display_img_collection([image_real, region])

with_real_image("mias-2.pgm")
