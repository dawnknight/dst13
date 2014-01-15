import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import dst13


# -------- 
#  Use on/off transitions to define discrete light curves
#
#  2014/01/04 - Written by Greg Dobler
# -------- 

def discrete_lc(night):

    # -- utilities
    lcbase    = 'lcs_night_' + str(night).zfill(2)
    oobase    = 'ind_onoff_night_' + str(night).zfill(2)
    nind_oo   = 9
    ind_onoff = []


    # -- read the intrinsic light curve parameters
    print("DST_DISCRETE_LC:   reading par...")

    infile = os.path.join(os.environ['DST_WRITE'],lcbase+'_par.pkl')
    fopen  = open(infile,'rb')
    start  = pkl.load(fopen)
    end    = pkl.load(fopen)
    paths  = pkl.load(fopen)
    files  = pkl.load(fopen)
    times  = pkl.load(fopen)
    samp   = pkl.load(fopen)
    reg    = pkl.load(fopen)
    ntime  = pkl.load(fopen)
    nwin   = pkl.load(fopen)
    fopen.close()


    # -- read the on/off transitions files
    print("DST_DISCRETE_LC: reading transition files for night " +
          "{0}...".format(night))

    for ioo in range(nind_oo):
        oofile     = oobase + '_' + str(ioo+1) + '.pkl'
        fopen      = open(os.path.join(os.environ['DST_WRITE'],oofile),'rb')
        ind_onoff += pkl.load(fopen)
        fopen.close()


    # -- convert to discrete light curves
    print("DST_DISCRETE_LC: converting to discrete light curves...")

    lcs_dis = np.zeros([nwin,ntime],dtype=np.int)

    for ilc in range(lcs_dis.shape[0]):
        if ind_onoff[ilc].size==0: # no transitions
            continue

        for ind in ind_onoff[ilc]:
            lcs_dis[ilc,ind:] += 1 if ind>0 else -1

        lcs_dis[ilc] -= lcs_dis[ilc].min() # define "min" as off


    return lcs_dis
