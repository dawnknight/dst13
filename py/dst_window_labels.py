import numpy as np
import scipy.ndimage as nd
from .dst_window_pix import *

# -------- 
# create a map of window labels
# -------- 
def window_labels(hand=False):
    """ Create a map of window labels """

    # -- get the window pixels
    winpix = window_pix(hand=hand)

    # -- label the windows
    labels, nobj = nd.measurements.label(winpix)

    return labels.astype(np.int)
