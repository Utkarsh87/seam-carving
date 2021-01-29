from PIL import Image
import glob
import os
import cv2

class GIFMake():
    def __init__(self, img_list):
        self.img_list = img_list
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/")

    def save_imgs(self):
        """
        Creates temporary png files to be used for creating the gif.
        These files will be stored(temporarily) in the results dir and deleted once the gif has been created.
        Returns none.
        """
        base_img_name = "for_gif_"
        extension = ".png"
        i = 1
        for img in self.img_list:
            cv2.imwrite(self.results_dir+base_img_name+str(i)+extension, img)
            i = i+1

    def gif_make(self):
        """
        Reads the saved png files and stores them in the apt order into a list.
        Makes a gif from the list and then deletes the png files.
        Returns none.
        """
        self.save_imgs()

        frames = [] # will store a list of images in the order required to make a gif
        imgs = glob.glob(self.results_dir+"*.png")

        # sort the filenames(filename: "for_gif_i.png")
        list.sort(imgs, key=lambda x: int(x.split('_')[2].split('.png')[0]))

        # populate the frames list
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        # create a gif from the images stored in "frames" list
        frames[0].save('out_new.gif', format='GIF', append_images=frames[1:], save_all=True, duration=40, loop=0)

        # delete the png files used for creating the gif
        for files in os.listdir(self.results_dir):
            if files.endswith(".png"):
                os.remove(os.path.join(self.results_dir, files))

