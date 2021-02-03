# Seam-Carving
Python implementation of Content-Aware Image Resizing.<br>
Uses a dynamic programming approach to find the minimum energy "seam" to "carve" out from the image.<br>

---
### Run the script ###
To view command line options and default arguments:<br>
```console
foo@bar:~/Seam-Carving$ python src/carver.py -h
```

To run the script on a test image:<br>
Ensure that the test image is in the "images" dir. The results dir will be automatically created if it 
doesn't already exist and all the results will be saved here. A gif of the entire process will be saved
in the current directory as well.<br>
```console
foo@bar:~/Seam-Carving$ python src/carver.py --image_name castle --num_seams 100 --filter_type laplace
```

---
TODO:
- reduce I/O operations during seam overlaying operation, should ideally reduce compute time
- add gif, current gif too big
---
- <strike>generalise file names(DONE)</strike>
- <strike>add argument parser, generalise other functionality in script(DONE, mostly..?)</strike>
- <strike>in giffer.py: all png files should auto-delete after the gif has been created(DONE)</strike>
- <strike>add laplace filter for energy map creation(DONE)</strike>
- <strike>energymap and colormap should be written only once, for the initial image. For subsequent calls to get_energy_map, don't re-write(DONE)</strike>
- <strike>modularize; separate carver functions from driver/main script(DONE)</strike>
- <strike>better naming for generated files(DONE)</strike>
