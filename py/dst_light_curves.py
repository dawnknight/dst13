import os
import pickle as pkl
import numpy as np
from .dst_window_labels import *
from .dst_io import *


# -------- 
#  Class holding (night time) light curves for a given start and end time
#
#  2013/11/22 - Writen by Greg Dobler (CUSP/NYU)
# -------- 
class LightCurves():


    def __init__(cls, start, end, dpath=os.environ['DST_DATA'], 
                 wpath=os.environ['DST_WRITE']):
        """ Make night time light curves for start and end times """

        # -- utilities
        bord = 20
        nrow = 2160
        ncol = 4096

        # -- get the full file list
        fopen = open(os.path.join(wpath,'filelist.pkl'),'rb')
        fl    = pkl.load(fopen)
        fopen.close()

        # -- take a time slice
        cls.paths, cls.files, cls.times = fl.time_slice(start, end)

        # -- get the window labels
        labels = window_labels(hand=True)
        wpix   = np.where(labels)
        labvec = labels[wpix].flatten()
        sind   = np.argsort(labvec)
        labsrt = labvec[sind]
        lbound = list(np.where((labsrt-np.roll(labsrt,1))==1)[0])
        lbound = [0] + lbound + [labsrt.size]

        # -- initialize the light curves
        cls.ntime = len(cls.times)
        cls.nwin  = len(lbound) - 1
        cls.lcs   = np.zeros([cls.nwin,cls.ntime,3])
        cls.std   = np.zeros([cls.nwin,cls.ntime,3])
        cls.err   = np.zeros([cls.nwin,cls.ntime,3])
        cls.bse   = np.zeros([cls.nwin,cls.ntime,3])

        # -- initialize color vectors
        red = np.zeros(sind.size)
        grn = np.zeros(sind.size)
        blu = np.zeros(sind.size)
        img = np.zeros([nrow-2*bord,ncol-2*bord,3]).astype(np.uint8)

        # -- loop through images
        for itime, (p,f,t) in enumerate(zip(cls.paths,cls.files,cls.times)):
            print("DST_LIGHT_CURVES: extracting window brightnesses for " + 
                  "file {0} of {1}".format(itime+1,len(cls.times)))

            img[:,:,:] = read_raw(f,p)[bord:-bord,bord:-bord,:]

            red[:] = ((img[:,:,0])[wpix])[sind].astype(np.float)
            grn[:] = ((img[:,:,1])[wpix])[sind].astype(np.float)
            blu[:] = ((img[:,:,2])[wpix])[sind].astype(np.float)

            for iwin in range(cls.nwin):
                ilo = lbound[iwin]
                ihi = lbound[iwin+1]
                rtn = np.sqrt(float(ihi-ilo+1))

                cls.lcs[iwin,itime,0] = red[ilo:ihi].mean()
                cls.lcs[iwin,itime,1] = grn[ilo:ihi].mean()
                cls.lcs[iwin,itime,2] = blu[ilo:ihi].mean()

                cls.std[iwin,itime,0] = red[ilo:ihi].std()
                cls.std[iwin,itime,1] = grn[ilo:ihi].std()
                cls.std[iwin,itime,2] = blu[ilo:ihi].std()

                cls.std[iwin,itime,0] = cls.std[iwin,itime,0]/rtn
                cls.std[iwin,itime,1] = cls.std[iwin,itime,1]/rtn
                cls.std[iwin,itime,2] = cls.std[iwin,itime,2]/rtn

        return
