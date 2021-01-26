"""
Content-aware image resizing(Seam Carving)
Author: Utkarsh Kumar Singh
Most of the code has been replicated from https://github.com/dtfiedler/seam-carving-python
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from tqdm import trange # progress bar

from giffer import GIFMake

results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/")
images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images/")
image_list = []

def get_energy_map(img):
    # use the gradient magnitude function to generate the energy map
    img = img.astype(np.uint8)

    # apply gaussian smoothing
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.int8)

    # turn into greyscale
    grey = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # use sobel filters(3*3) to get gradients
    sobel_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # merge into energy map
    energy_map = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    # write color map
    color_map = cv2.applyColorMap(energy_map, cv2.COLORMAP_JET)
    cv2.imwrite(results_dir+"colormap.jpg", color_map)

    # write energy map
    cv2.imwrite(results_dir+"energy.jpg", energy_map)

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
    num_seams = 150 # how many seams to remove from the image
    energy_map = get_energy_map(output_img)

    for _ in trange(num_seams):
        seam = get_min_seam(output_img, energy_map)

        overlay_seam(seam, "castle_with_seams.jpg") # to show all seams removed overlayed on the original image

        cv2.imwrite(results_dir+"cur_castle_with_next_seam.jpg", output_img)
        overlay_seam(seam, "cur_castle_with_next_seam.jpg") # overlay seam on current image(used for making gif)
        img_for_gif = cv2.imread(results_dir+"cur_castle_with_next_seam.jpg")
        image_list.append(img_for_gif)

        output_img = delete_seam(output_img, seam)
        image_list.append(output_img)

        energy_map = get_energy_map(output_img) # get energy map of new image

    cv2.imwrite(results_dir+"castle_new.jpg", output_img)

img = cv2.imread(images_dir+"castle.jpg")
output_img = np.copy(img)
cv2.imwrite(results_dir+"castle_new.jpg", output_img)
cv2.imwrite(results_dir+"castle_with_seams.jpg", img)
carve_seams(output_img)

# create gif
run = GIFMake(image_list)
run.save_imgs()
run.gif_make()
