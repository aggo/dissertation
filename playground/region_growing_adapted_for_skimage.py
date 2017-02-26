import numpy
from skimage import io

def simple_region_growing(img, seed, threshold=1):
    # via http://www.lengrand.fr/2011/11/simple-region-growing-implementation-in-python, adapted for skimage
    region = img.copy()
    region[:,:]=0  #initialize it all with black

    orig_image_dimensions = img.shape

    #parameters
    mean_intensity_of_the_region = float(img[seed[0], seed[1]])
    size_of_region = 1
    orig_image_area = orig_image_dimensions[0]*orig_image_dimensions[1]
    contour_of_segmented_region = [] # will be [ [[x1, y1], val1],..., [[xn, yn], valn] ]  ## what does contour really refer to, only contour pixels or all pixels?
    region_contour_intensity_values = []
    distance_between_avg_intensity_of_region_and_current_neighbor_intensity = 0  # the u in the formula |u-I(x,y)|<RT

    # TODO: may be enhanced later with 8th connectivity
    neighbor_orientation = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
    current_pixel = [seed[0], seed[1]]

    #Spreading
    while(distance_between_avg_intensity_of_region_and_current_neighbor_intensity<threshold and size_of_region<orig_image_area):
        #adding pixels (parsing neighbors)
        for j in range(len(neighbor_orientation)):
            #select neighbor
            neighbor_coordinates = [current_pixel[0] +neighbor_orientation[j][0], current_pixel[1] +neighbor_orientation[j][1]]
            #check if it belongs to the image
            is_in_img = orig_image_dimensions[0]>neighbor_coordinates[0]>0 and orig_image_dimensions[1]>neighbor_coordinates[1]>0 #returns boolean
            #candidate is taken if not already selected before
            # if is_in_img and the region value for it is still 0 (meaning it hasn't been visited) (or added?)
            if (is_in_img and (region[neighbor_coordinates[0], neighbor_coordinates[1]]==0)):   ## Why 1,0 and not 0,1? is it an opencv thingy? -> changed them to 0,1 because was getting an error
                contour_of_segmented_region.append(neighbor_coordinates)
                region_contour_intensity_values.append(img[neighbor_coordinates[0], neighbor_coordinates[1]] )
                region[neighbor_coordinates[0], neighbor_coordinates[1]] = 150

        #add the nearest pixel of the contour in it
        distance_between_avg_intensity_of_region_and_current_neighbor_intensity = abs(numpy.mean(region_contour_intensity_values) - mean_intensity_of_the_region)

        list_of_distances = [abs(i - mean_intensity_of_the_region) for i in region_contour_intensity_values ]
        if (len(list_of_distances)>0):
            index = list_of_distances.index(min(list_of_distances)) #mean distance index
            size_of_region += 1 # updating region size
            region[current_pixel[0], current_pixel[1]] = 255

            #updating mean MUST BE FLOAT
            mean_intensity_of_the_region = (mean_intensity_of_the_region*size_of_region + float(region_contour_intensity_values[index]))/(size_of_region+1)
            #updating seed
            current_pixel = contour_of_segmented_region[index]

            #removing pixel from neigborhood
            del contour_of_segmented_region[index]
            del region_contour_intensity_values[index]

    return region

def display_image(image, cmap='gray'):
    io.imshow(image, cmap=cmap)
    io.show()

path = "mdb001.pgm"
image = io.imread(path)
seed_coord = [50,500]
region = simple_region_growing(image, seed_coord)
display_image(region)