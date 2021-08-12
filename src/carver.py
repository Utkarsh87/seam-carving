"""
Content-aware image resizing(Seam Carving)
Author: Utkarsh Kumar Singh
Some code borrowed from https://github.com/dtfiedler/seam-carving-python

To run this script:
python src/carver.py --image_name castle --num_seams 100 --filter_type laplace
Make sure to keep the image to be resized in the images dir, resized images along with other files will be saved in the results dir.
"""

import os
import argparse
# import warnings
# warnings.filterwarnings("ignore")
from pathlib import Path
Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from tqdm import trange # progress bar

from giffer import GIFMake

def get_energy_map(img, filter_type):
    # use the gradient magnitude function to generate the energy map
    img = img.astype(np.uint8)

    # apply gaussian smoothing
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.int8)

    # turn into greyscale
    grey = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if(filter_type == "laplace"): # use Laplace filter
        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        energy_map = abs_laplacian

    else: # use Sobel filter; default
        sobel_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        # merge into energy map
        energy_map = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    # write color map(only for original image)
    color_map = cv2.applyColorMap(energy_map, cv2.COLORMAP_JET)
    if not os.path.isfile(os.path.join(results_dir, f"{args.image_name}_colormap_{args.filter_type}.jpg")):
        cv2.imwrite(results_dir+f"{args.image_name}_colormap_{args.filter_type}.jpg", color_map)

    # write energy map(only for original image)
    if not os.path.isfile(os.path.join(results_dir, f"{args.image_name}_energy_{args.filter_type}.jpg")):
        cv2.imwrite(results_dir+f"{args.image_name}_energy_{args.filter_type}.jpg", energy_map)

    return energy_map

def delete_seam(img, seam):
    rows, cols, _ = img.shape
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]

    output_img = img[:, 0:cols-1]
    return output_img

def overlay_seam(seam, file):
    img = cv2.imread(results_dir+file)
    img_with_seam = np.copy(img)
    x, y = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    img_with_seam[x, y] = (0, 0, 255)
    cv2.imwrite(results_dir+file, img_with_seam)

def get_min_seam(img, energy_map):
    rows, cols, _ = img.shape
    seam = np.zeros(rows)
    dist = np.zeros((rows, cols)) + 100000000
    dist[0,:] = np.zeros(cols)
    edge = np.zeros((rows, cols))

    for row in range(rows-1):
        for col in range(cols):
            if col != 0:
                first = dist[row+1, col-1]
                prev = dist[row, col] + energy_map[row+1, col-1]
                dist[row+1, col-1] = min(first, prev)

                if first>prev:
                    edge[row+1, col-1] = 1

            second = dist[row+1, col]
            cur = dist[row, col] + energy_map[row+1, col]
            dist[row+1, col] = min(second, cur)
            if second>cur:
                edge[row+1, col] = 0

            if col != cols-1:
                third = dist[row+1, col+1]
                next_ = dist[row, col] + energy_map[row+1, col+1]
                dist[row+1, col+1] = min(third, next_)
                if third>next_:
                    edge[row+1, col+1] = -1

    seam[rows-1] = np.argmin(dist[rows-1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i-1] = seam[i] + edge[i, int(seam[i])]

    return seam

def carve_seams(output_img):
    num_seams = args.num_seams # how many seams to remove from the image
    energy_map = get_energy_map(output_img, args.filter_type)

    for _ in trange(num_seams):
        seam = get_min_seam(output_img, energy_map)

        overlay_seam(seam, f"{args.image_name}_with_seams_{args.filter_type}.jpg") # to show all seams removed overlayed on the original image

        # the image file "cur_with_next_seam.jpg" is a temp file used as a frame in creating gif of the entire seam carving process
        cv2.imwrite(results_dir+"cur_with_next_seam.jpg", output_img)
        overlay_seam(seam, "cur_with_next_seam.jpg") # overlay seam on current image
        img_for_gif = cv2.imread(results_dir+"cur_with_next_seam.jpg")
        image_list.append(img_for_gif)
        os.remove(os.path.join(results_dir, "cur_with_next_seam.jpg"))

        output_img = delete_seam(output_img, seam)
        image_list.append(output_img)

        energy_map = get_energy_map(output_img, args.filter_type) # get energy map of new image

    cv2.imwrite(results_dir+f"{args.image_name}_resized_{args.filter_type}.jpg", output_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_name', help="filename of the image for resizing. Image must be present in the images directory. This name will be used to generate resultant filenames.", type=str)
    parser.add_argument('--num_seams', help="number of vertical seams to be removed from the image.", default=100, const=100, nargs='?', type=int)
    parser.add_argument('--filter_type', help="type of filter to be used for generating energy map.", default="sobel", choices=set(("sobel", "laplace")), nargs='?', const="sobel", type=str)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/") # dir to store resultant images and energy maps etc
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images/") # dir to read images from

    # list to store the frames for creating the gif
    image_list = []

    # read said image
    img = cv2.imread(images_dir+args.image_name+".jpg")

    # create 2 output image files to be overwritten every time a new seam is computed and carved
    cv2.imwrite(results_dir+f"{args.image_name}_resized_{args.filter_type}.jpg", img)
    cv2.imwrite(results_dir+f"{args.image_name}_with_seams_{args.filter_type}.jpg", img)

    # perform seam carving
    carve_seams(img)

    # create gif
    run = GIFMake(image_list, args)
    run.gif_make()
