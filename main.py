from scipy.ndimage import generic_filter, morphology
from skimage import measure

from skimage import data, io, filters
from skimage.filters import median
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import closing
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import opening
from skimage.morphology import reconstruction
from skimage.morphology import remove_small_objects
from skimage.morphology import square


def display_image(image, cmap='gray'):
    io.imshow(image, cmap=cmap)
    io.show()


def denoise(image):
    return median(image, disk(3))


def binary_image_with_threshold(image, threshold, display=False):
    thresh = threshold_otsu(image)
    binary = image > thresh
    if (display):
        display_image_histogram_and_binary(image, thresh, binary)
    return binary


def display_image_histogram_and_binary(image, thresh, binary):
    fig = plt.figure(figsize=(8, 2.5))
    ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1, adjustable='box-forced')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')

    ax2.hist(image)
    ax2.set_title('Histogram')
    ax2.axvline(thresh, color='r')

    ax3.imshow(binary, cmap=plt.cm.gray)
    ax3.set_title('Thresholded')
    ax3.axis('off')
    plt.show()


def label_connected_components(image, display=False):
    all_labels = measure.label(image)
    blobs_labels = measure.label(image, background=0)
    if (display):
        plt.figure(figsize=(9, 3.5))
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(all_labels, cmap='spectral')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(blobs_labels, cmap='spectral')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return blobs_labels


def morphological_opening_to_remove_extra_objects(image, structuring_element):
    """
    neighborhood = structuring element = selem
    :param image:
    :param structuring_element:
    :return:
    """
    return opening(image, structuring_element)


def display_all(images):
    array_length = len(images)
    plt.figure(figsize=(5 * array_length, 5.5))
    index = 1
    for image in images:
        plt.subplot(3, 2, index)  # 3 lines, 2 columns
        index += 1
        plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.show()

def smoothen_image(image):
    """
    sets a pixel's value to one where 5 or more pixels in its 3x3 neighborhood are 1
    :param image:
    :return:
    """
    window = square(3)
    def _replace_center_with_one_if_five_neighbors_are_one(values):
        """
        For each location in the input image, the value returned by the function is the value assigned to that location.
        That's why, naturally, the function needs to return a scalar.
        :param values:
        :return: a scalar representing the value to be set at the current location in the input image
        """
        ones_count = 0
        for entry in values:
            if entry == 1:
                ones_count += 1
        if ones_count >= 5:
            return 1
        else:
            return 0

    """
    This call will take windows of the shape given by the footprint, send them as an 1D array to the _replace function
    and return the value that is to be set in the center of the window. The edges are ignored (for now)
    """
    new_image = generic_filter(image, _replace_center_with_one_if_five_neighbors_are_one, footprint = window)
    return new_image


def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()

def morphological_closing(image, structuring_element=disk(5)):
    return closing(image, structuring_element)

def fill_holes(image):
    seed_image = np.copy(image)  # the max possible values of the original image
    seed_image[1:-1, 1:-1] = image.max() # along the borders, use original values
    # (the border pixels will be the starting point for the erosion process)
    mask = image
    filled = reconstruction(seed_image, mask, method="erosion")
    return filled

    # fig, ax = plt.subplots(1, 2, figsize=(5, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    # ax = ax.ravel()
    #
    # ax[0].imshow(image)
    # ax[0].set_title('Original image')
    # ax[0].axis('off')
    # ax[1].imshow(filled)
    # ax[1].set_title('after filling holes')
    # ax[1].axis('off')
    # show the holes
    # ax[2].imshow(image - filled)
    # ax[2].set_title('holes')
    # ax[2].axis('off')

    # if we wanted bright spots, we'd use dilation instead of erosion
    # (dilation expands the maximal values of the seed image until it encounters a max image)
    # for this, init the seed image with the minimum image intensity instead of the maximum

def crop_biggest_object(image):
    # via https://gist.github.com/wkentaro/016b6bbb32e8663ac673
    imgray = image
    img_denoised = denoise_tv_chambolle(imgray, weight=.1)
    thresh = threshold_otsu(img_denoised)
    # get label_img
    bw = closing(img_denoised > thresh, square(2))
    cleared = bw.copy()
    label_img = measure.label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_img[borders] = -1
    # get biggest area
    max_region = sorted((region.area, region)
        for region in regionprops(label_img))[-1][1]
    minr, minc, maxr, maxc = max_region.bbox
    return image[minr:maxr, minc:maxc]


def determine_sum_of_pixels(image, size_of_corner=10):
    left_matrix = image[0:size_of_corner,0:size_of_corner]
    right_matrix = image[0:size_of_corner,image.shape[0]-size_of_corner:image.shape[0]]
    left_sum = sum(sum(left_matrix))
    right_sum = sum(sum(right_matrix))
    return (left_sum, right_sum)

# returns 0 for left breast orientation and 1 for right breast orientation
def determine_breast_orientation(image,size_of_corner=10):
    sums_of_pixels = determine_sum_of_pixels(image, size_of_corner)
    if (sums_of_pixels[0]>sums_of_pixels[1]):
        return 1
    return 0


def remove_pectoral_muscle_image(cropped_image):
    breast_orientation = determine_breast_orientation(cropped_image)

    pass


if __name__ == "__main__":
    path = "all-mias/mdb001.pgm"
    image = io.imread(path)
    original = image.copy()
    # display_image(image)
    denoised_image = denoise(image)
    # display_image(denoised_image)
    binary_thresholded_image = binary_image_with_threshold(denoised_image, threshold=18, display=False)
    # display_image(binary_thresholded_image)
    connected_components_image = label_connected_components(binary_thresholded_image, display=False)
    # display_image(connected_components_image, cmap='spectral')
    largest_area_only_image = morphological_opening_to_remove_extra_objects(connected_components_image,
                                                                            structuring_element=square(10))
    # display_image(largest_area_only_image)
    small_objects_removed_image = remove_small_objects(largest_area_only_image, min_size=100)
    # display_all([imageL, denoised_image, binary_thresholded_image, connected_components_image, largest_area_only_image])
    # display_image(small_objects_removed_image)
    smoothened_image = smoothen_image(small_objects_removed_image)
    # plot_comparison(small_objects_removed_image, smoothened_image, "Smoothened")
    morphological_closing_image = morphological_closing(smoothened_image, structuring_element=disk(5))
    # plot_comparison(image, morphological_closing_image, "Morphological closing")
    holes_filled_image = fill_holes(morphological_closing_image)
    # plot_comparison(morphological_closing_image, holes_filled_image, "Holes filled")

    multiplied_image = holes_filled_image*image

    cropped_image = crop_biggest_object(multiplied_image)
    # plot_comparison(original, cropped_image, "Cropped image")
    display_image(cropped_image)
    print(determine_breast_orientation(cropped_image))

    # image_without_pectoral_muscle = remove_pectoral_muscle_image(cropped_image);

    # display_image(smoothened_image)
    # display_all([largest_area_only_image, small_objects_removed_image])
    print("Done!")
