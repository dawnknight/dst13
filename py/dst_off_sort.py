import os
import numpy as np
import pickle as pkl
from .dst_light_curves import *

# -------- 
#  Sort light curves according to "big off" times
#
#  2014/01/26 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def off_sort(night, lcs=None, residential=True, indices=None, band=0):

    # -- check input
    if lcs==None:
        lcs = LightCurves('','',infile='lcs_night_'+str(night).zfill(2), 
                          noerr=True)
    else:
        print("DST_OFF_SORT: using input LightCurves instance...")


    # -- get window labels
    if residential:
        print("DST_OFF_SORT: reading window labels...")

        fopen = open(os.path.join(os.environ['DST_WRITE'],
                                  'window_labels.pkl'), 'rb')
        labs  = pkl.load(fopen)
        fopen.close()


    # -- get the on/off transitions and convert to array
    fopen     = open(os.path.join(os.environ['DST_WRITE'], 
                                  'ind_onoff_night_'+
                                  str(night).zfill(2)+'.pkl'), 'rb')
    ind_onoff = pkl.load(fopen)
    fopen.close()


    # -- if not input, get the indices of the light curves
    if indices==None:
        sort = True

        if residential:
            indices = np.arange(lcs.lcs.shape[0])[np.array([
                        rpos>1000 and len(onoff[onoff<0])>0 for 
                        rpos,onoff in zip(labs.rvec, ind_onoff)])]
        else:
            indices = np.arange(lcs.lcs.shape[0])[np.array([
                        rpos>1000 and len(onoff[onoff<0])>0 for 
                        rpos,onoff in zip(labs.rvec, ind_onoff)])]
    else:
        sort = False
        print("DST_OFF_SORT: using input light curves indices...")


    # -- pull out the subset of light curves
    sub = lcs.lcs[indices].mean(1) if band==3 else lcs.lcs[indices,:,band]


    # -- normalize according to max/min value of light curve
    sub[:,:] = (sub.T-sub.min(1)).T
    sub[:,:] = (sub.T/sub.max(1)).T


    # -- identify largest transition
    big_onoff = [[i] for i in indices]

    for ionoff, onoff_arr in enumerate([ind_onoff[i] for i in indices]):
        bigoff, left = 0, 0.0
        bigon, right = 0, 0.0
        for onoff in onoff_arr:
            if onoff<0: # off transition
                tleft = sub[ionoff,:-onoff].mean()-sub[ionoff,-onoff:].mean()
                if tleft>left:
                    bigoff = onoff
                    left   = tleft
            elif onoff>0: # on transition
                tright = sub[ionoff,onoff:].mean()-sub[ionoff,:onoff].mean()
                if tright>right:
                    bigon = onoff
                    right = tright

        big_onoff[ionoff] += [bigon, bigoff]


    # -- sort by big off transition and return
    if sort:
        return [big_onoff[i] for i in np.argsort(np.abs(zip(*big_onoff)[2]))]
    else:
        return big_onoff
