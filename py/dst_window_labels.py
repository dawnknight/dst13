import numpy as np
import scipy.ndimage as nd
from .dst_window_pix import *

# -------- 
#  Create a map of window labels
#
#  2013/11/22 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def window_labels(hand=False):
    """ Create a map of window labels """

    # -- get the window pixels
    winpix = window_pix(hand=hand)

    # -- label the windows
    labels, nobj = nd.measurements.label(winpix)

    return labels.astype(np.int)


# -------- 
#  Class containing window labels and useful parameters
# -------- 
class WindowLabels():

    def __init__(cls, hand=False, nopos=False):
        """ class containing window labels and useful parameters """

        # -- get the window labels
        cls.labels = window_labels(hand=hand) # labels map
        cls.wpix   = np.where(cls.labels) # window pixels
        cls.labvec = cls.labels[cls.wpix].flatten() # labels vector
        cls.sind   = np.argsort(cls.labvec) # sorting indices
        cls.labsrt = cls.labvec[cls.sind] # sorted labels vector
        lbound     = list(np.where((cls.labsrt-np.roll(cls.labsrt,1))==1)[0])
        cls.lbound = [0] + lbound + [cls.labsrt.size] # boundaries of labels
        cls.nwin   = len(cls.lbound) - 1 # total number of windows

        # -- set the position labels
        cls.rvec = np.zeros(cls.nwin)
        cls.cvec = np.zeros(cls.nwin)

        if not nopos:
            for iwin in range(1,cls.nwin+1):
                index            = cls.labvec==iwin
                cls.rvec[iwin-1] = cls.wpix[0][index].astype(np.float).mean()
                cls.cvec[iwin-1] = cls.wpix[1][index].astype(np.float).mean()


        return
