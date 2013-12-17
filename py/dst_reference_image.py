import os
from .dst_io import *

# -------- 
#  Get the reference image
#
#  2013/12/16 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def reference_image(bord=None):

    """ Get the reference image used for regestration """

    rpath = os.path.join(os.environ['DST_DATA'],'11/15/19.03.48')
    rfile = 'oct08_2013-10-25-175504-182135.raw'

    if bord==None:
        return read_raw(rfile,rpath)
    else:
        return read_raw(rfile,rpath)[bord:-bord,bord:-bord,:]

    return ref
