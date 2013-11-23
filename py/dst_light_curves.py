import os
import pickle as pkl
import numpy as np
from .dst_window_labels import *


# -------- 
#  Class holding (night time) light curves for a given start and end time
#
#  2013/11/22 - Writen by Greg Dobler (CUSP/NYU)
# -------- 
class LightCurves():


    def __init__(cls, start, end, dpath=os.environ['DST_DATA'], 
                 wpath=os.environ['DST_WRITE']):
        """ Make night time light curves for start and end times """

        # -- get the full file list
        fopen = open(os.path.join(wpath,'filelist.pkl'),'rb')
        fl    = pkl.load(fopen)
        fopen.close()

        # -- take a time slice
        paths, files, times = fl.time_slice(start, end)

        # -- get the window labels
        labels = window_labels(hand=True)
        wpix   = np.where(labels)
        labvec = labels[wpix].flatten()
        sind   = np.argsort(labvec)
        labsrt = labvec[sind]
        lbound = [0] + list(np.where((labsrt-np.roll(labsrt,1))==1)[0]) + 
                 [labsrt.size]

        # -- initialize the light curves
        ntime = len(times)
        npix  = labels[labels>0].size


# get a list of images (night time)

# read in the image, extract just the brightness of window pixels,
# average over pixels with a given label, compute the bootstrap
# (remove 2) error, compute the standard deviation

# times, amplitudes, errors
