import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from .dst_io import *

# -------- 
#  Generate plots for the DST13 lightscape project
#
#  2013/12/16 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def image_hires(inpath='11/28/14.28.45/', 
                infile='oct08_2013-10-25-175504-292724.raw',
                wpath=os.environ['DST_WRITE'], 
                wfile='day_img_112813_1430_hires.png'):

    """ Generate a high resolution png.  Default is a daytime image on
    11/28/13 at ~2:30pm. """

    # -- read in the data
    rpath = os.path.join(os.environ['DST_DATA'],inpath)

    print("DST_PLOTS: reading in image")
    print("DST_PLOTS:   path = {0}".format(rpath))
    print("DST_PLOTS:   file = {0}".format(infile))

    img = read_raw(infile, rpath)


    # -- make the plot
    plt.figure(figsize=[7.50*3.5,3.955*3.5])
    plt.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()


    # -- write to file
    print("DST_PLOTS: writing output file")
    print("DST_PLOTS:   path = {0}".format(wpath))
    print("DST_PLOTS:   file = {0}".format(wfile))

    plt.savefig(os.path.join(wpath,wfile),clobber=True)
    plt.close()

    return
