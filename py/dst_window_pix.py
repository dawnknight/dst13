import os
import numpy as np
from .dst_read_bin import *

# -------- 
# load window pixel labels
# -------- 
def window_pix(hand=False, create=False):
    """ Load window pixel labels """

    if hand:
        print("DST_WINDOW_PIX: loading hand selected window list...")

        # -- define the path and get the file list
        path  = os.path.join(os.environ['DST_WRITE'],'window_chooser')
        files = [i for i in os.listdir(path) if 'reg' in i]

        # -- utilities
        sfile  = 'window_labels.out'
        reg    = range(1,33)
        nfile  = len(files)
        nrow   = 2120
        ncol   = 4056
        winpix = np.zeros([nrow,ncol])
        dtype  = np.float

        # -- check if stacked window list is alread available
        if not create and (sfile in os.listdir(path)):
            return read_bin(sfile,np.int,[nrow,ncol],path)

        # -- loop through files and add to mask
        for i, ifile in enumerate(files):
            winpix[:,:] += read_bin(ifile,dtype,[nrow,ncol],path)

        winpix = 1*(winpix > 0.0)

        # -- split merged windows
        winpix = winpix*read_bin('split_merged.out',np.int,[nrow,ncol],path)

        # -- write to file
        fout = os.path.join(path,sfile)

        print("DST_WINDOW_PIX: writing combined labels to")
        print("DST_WINDOW_PIX:   {0}".format(fout))

        fopen = open(os.path.join(path,sfile),'wb')
        fopen.write(winpix)
        fopen.close()

        return winpix
    else:
        print("DST_WINDOW_PIX: only hand selected windows are available!!!")
        return -1
