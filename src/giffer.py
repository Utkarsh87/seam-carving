from PIL import Image
import glob
import os
import cv2

class GIFMake():
    def __init__(self, img_list):
        self.img_list = img_list
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/")

    def save_imgs(self):
        base_img_name = "for_gif_"
        extension = ".png"
        i = 1
        for img in self.img_list:
            cv2.imwrite(self.results_dir+base_img_name+str(i)+extension, img)
            i = i+1

    def gif_make(self):
        frames = []
        imgs = glob.glob(self.results_dir+"*.png")

        # sort the filenames(filename: "for_gif_i.png")
        list.sort(imgs, key=lambda x: int(x.split('_')[2].split('.png')[0]))

        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        frames[0].save('out_new.gif', format='GIF', append_images=frames[1:], save_all=True, duration=40, loop=0)
