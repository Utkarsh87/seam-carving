"""
Content-aware image resizing(Seam Carving)
Author: Utkarsh Kumar Singh
Most of the code has been replicated from https://github.com/dtfiedler/seam-carving-python
"""


import numpy as np
import cv2
import sys
from scipy.ndimage.filters import convolve
from tqdm import trange # progress bar
import warnings
warnings.filterwarnings("ignore")

def get_energy_map(img):
    img = img.astype(np.uint8)
    # use the gradient magnitude function to generate the energy map
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
    cv2.imwrite("colormap.jpg", color_map)
    # write energy map
    cv2.imwrite("energy.jpg", energy_map)

    return energy_map

# # compute the enery map of the image
# def calc_energy(img):
#     # E = Ex + Ey   (E: energy of a pixel)
#     # Ex = deltaRx^2 + deltaGx^2 + deltaBx^2
#     # Ey = deltaRy^2 + deltaGy^2 + deltaBy^2

#     img = img.astype(np.uint8)
#     # use the gradient magnitude function to generate the energy map
#     # apply gaussian smoothing
#     img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.int8)
#     # turn into greyscale
#     grey_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

#     # use Sobel filters for this computation
#     # create 2D Sobel filters and stack 3 for each colour channel
#     hor_filter = np.array([
#             [1.0, 2.0, 1.0],
#             [0.0, 0.0, 0.0],
#             [-1.0, -2.0, -1.0],
#     ])
#     # hor_filter = np.stack([hor_filter]*3, axis=2) # 2D -> 3D filter

#     ver_filter = np.array([
#         [1.0, 0.0, -1.0],
#         [2.0, 0.0, -2.0],
#         [1.0, 0.0, -1.0],
#     ])
#     # ver_filter = np.stack([ver_filter]*3, axis=2)

#     conv_out = np.absolute(convolve(grey_img, hor_filter)) + np.absolute(convolve(grey_img, ver_filter)) # E = Ex + Ey
#     # energy_map = conv_out.sum(axis=2) # sum gradients for each channel
#     energy_map = conv_out

#     # write color map
#     color_map = cv2.applyColorMap(energy_map, cv2.COLORMAP_JET)
#     cv2.imwrite("colormap.jpg", color_map)

#     cv2.imwrite("energy.jpg", energy_map)
#     return energy_map


def delete_seam(img, seam):
    rows, cols, _ = img.shape
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]

    output_img = img[:, 0:cols-1]
    return output_img

def overlay_seam(seam, file):
    img = cv2.imread(file)
    img_with_seam = np.copy(img)
    x, y = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    img_with_seam[x, y] = (0, 0, 255)
    cv2.imwrite(file, img_with_seam)

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
    num_seams = 10
    energy_map = get_energy_map(output_img)

    for _ in trange(num_seams):
        seam = get_min_seam(output_img, energy_map)
        overlay_seam(seam, "castle_with_seams.jpg")
        output_img = delete_seam(output_img, seam)
        energy_map = get_energy_map(output_img)

    cv2.imwrite("castle_new.jpg", output_img)

if __name__ == "__main__":
    img = cv2.imread("castle.jpg")
    output_img = np.copy(img)
    cv2.imwrite("castle_with_seams.jpg", img)

    carve_seams(output_img)
