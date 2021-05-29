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
- add option to remove horizontal seams
- instead of asking number of seams from user, ask required image dimensions and resize accordingly
