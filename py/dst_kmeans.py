import os
import pickle as pkl
from .dst_night_times import *
from .dst_light_curves import *

# -------- 
#  Run K-Means on night time window light curves
#
#  2013/12/07 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def kmeans(wpath=os.environ['DST_WRITE'], n_clusters=12, band=[0,1,2,3]):

    """ run K-Means clustering on night time window light curves """

    # -- defaults
    if type(band)==int:
        band = [band]


    # -- utilities
    bname  = ['r','g','b','rgb']
    nnight = len(night_times()[0])


    # -- loop through nights
    for inight in range(nnight):
        infile  = 'lcs_night_' + str(inight).zfill(2)
        lcs     = LightCurves('','',infile=infile,noerr=True)
        kmeanss = lcs.k_means(n_clusters,band=band)


        # -- write to file
        for iband in band:
            wname  = 'kmeans_night_'+str(inight).zfill(2)+'_'+str(iband)+'.pkl'

            print("DST_KMEANS: Writing solution to ")
            print("DST_KMEANS:   path = {0}".format(wpath))
            print("DST_KMEANS:   file = {0}".format(wname))

            fopen = open(os.path.join(wpath,wname),'wb')
            pkl.dump(kmeanss[iband],fopen)
            fopen.close()

        # -- free lcs from memory
        lcs = []

    return
