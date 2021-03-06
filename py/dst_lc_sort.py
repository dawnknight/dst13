import os
import numpy as np
import pickle as pkl
from .dst_light_curves import *
import matplotlib.pyplot as plt

def lc_sort(night, norm=1, band=4, residential=True, delta=True, lc_ind=None, 
            index=None):

    # -- utilities
    ngtst = str(night).zfill(2)


    # -- read in the light curves
    lcs = LightCurves('','',infile='lcs_night_'+ngtst,noerr=True)


    # -- get window labels
    if residential:
        print("DST_LC_SORT: reading window labels...")

        fopen = open(os.path.join(os.environ['DST_WRITE'],'window_labels.pkl'),
                     'rb')
        labs  = pkl.load(fopen)
        fopen.close()


    # -- get the on/off transitions and convert to array
    ind_onoff = []
    for i in range(1,10):
        fopen = open('../output/ind_onoff_night_'+ngtst+'_'+str(i)+'.pkl','rb')
        ind_onoff += pkl.load(fopen)
        fopen.close()

    ind_arr = np.array([i for i in ind_onoff])


    # -- get subset of lightcurves
    if lc_ind==None:
        lc_ind = []
        if residential:
            lc_ind = [((j>1000) and (len(k[k<0])>0)) for (k,j) in
                      zip(ind_arr,labs.rvec)]
        else:
            lc_ind = [len(k[k<0])>0 for k in ind_arr]
    else:
        print("DST_PLOTS: using input light curve indices...")

    lc_vind = np.array([i for i in lc_ind])
    if band==4:
        sub = lcs.lcs[lc_vind].mean(2)
    else:
        sub = lcs.lcs[lc_vind,:,band]


    # -- pull out the sublist of off transitions
    ind_sub = ind_arr[lc_vind]


    # -- normalize according to maximum value of the lightcurve
    if norm==1:
        snorm = (sub.T-sub.min(1)).T
        snorm = (snorm.T/snorm.max(1)).T
    elif norm==2:
        snorm = (sub.T/sub.max(1)).T
    else:
        snorm = sub


    # -- identify largest transition and sort
    xx_off, yy_off = [], []
    xx_on, yy_on   = [], []

    for i, sublist in enumerate(ind_sub):
        bigoff, left = 0, 0.0
#        bigon, right = 0, 0.0 if delta else sub[i].mean()
        bigon, right = 0, 0.0

        plt.figure()
        plt.plot(range(sub[i].size),sub[i])
        plt.title(np.arange(lc_vind.size)[lc_vind][i])
        for j in sublist:
            if j<0:
                plt.plot(-j,sub[i,-j],'ro')
                tleft = sub[i,:-j].mean()-sub[i,-j:].mean() if delta else \
                    sub[i,:-j].mean()
                if tleft>left:
                    bigoff = j
                    left   = tleft
            elif j>0:
                plt.plot(j,sub[i,j],'go')
                tright = sub[i,j:].mean()-sub[i,:j].mean() if delta else \
                    sub[i,j:].mean()
                if tright>right:
                    bigon = j
                    right = tright
        plt.show()
        xx_off.append(i)
        yy_off.append(bigoff)
        xx_on.append(i)
        yy_on.append(bigon)


    # -- sort according to big off
    if index==None:
        index = []
        for iy in np.argsort(np.abs(yy_off)):
            index.append(iy)
    else:
        print("DST_PLOTS: using input sorting indices...")

    vindex = np.array([i for i in index])
    yy_off = np.array([i for i in yy_off])[vindex[::-1]]
    yy_on  = np.array([i for i in yy_on])[vindex[::-1]]

    return lc_ind, index, yy_on, yy_off
