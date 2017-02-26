"""
Problem statement:
Given a starting point in an image, keep adding points from the image to it that fulfill a certain condition.
Step 1: condition is that the pixels have a value greater than a given threshold
"""
from pprint import pprint

import numpy
from skimage import io
from queue import Queue

def display_image(image, cmap='gray'):
    io.imshow(numpy.array(image), cmap=cmap)
    io.show()
#-----------------------------------------------------------------------------------------------------------------------
def region_growing_s1(image, seed, threshold):

    region = image.copy()
    region[:,:] = 0  # blacken out the image

    # build a queue to hold all the pixels from the image to be processed
    q = Queue()
    q.put(seed)  # initially it only contains the seed pixel

    orientations = ([-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1])
    while not q.empty():
        current_point = q.get()
        cpx = current_point[0]
        cpy = current_point[1]
        # check if the current point can be added to the region
        if (image[cpx, cpy]>threshold):
            region[cpx, cpy] = 255  # whiten the pixels that belong to the segmented region
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

image = [[0, 0,   0,   0,   100, 150, 111],
         [110, 100, 225, 100, 100, 0,   0],
         [0, 0,   100, 100, 210, 0,   0],
         [0, 0,   0,   60,  0,   0,   0]]
image = numpy.array(image)

display_image(image)
region = region_growing_s1(image, [2,3], 1)
display_image(region)