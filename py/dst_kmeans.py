import os
import pickle as pkl
import matplotlib.cm as cm
from .dst_night_times import *
from .dst_light_curves import *
from .dst_window_labels import *

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
        for ii,iband in enumerate(band):
            wname  = 'kmeans_' + str(n_clusters).zfill(2) + '_night_' + \
                str(inight).zfill(2) + '_' + str(iband) + '.pkl'

            print("DST_KMEANS: Writing solution to ")
            print("DST_KMEANS:   path = {0}".format(wpath))
            print("DST_KMEANS:   file = {0}".format(wname))

            fopen = open(os.path.join(wpath,wname),'wb')
            pkl.dump(kmeanss[ii],fopen)
            fopen.close()

        # -- free lcs from memory
        lcs = []

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def kmeans_labels_map(night, band):

    """ Make a map of the labels for the K-Means clusters """

    # -- read in the K-Means file
    kmfile = 'kmeans_night_' + str(night).zfill(2) + '_' + str(band) + '.pkl'
    wpath  = os.environ['DST_WRITE']

    print("DST_KMEANS: reading in K-Means file")
    print("DST_KMEANS:   path = {0}".format(wpath))
    print("DST_KMEANS:   file = {0}".format(kmfile))

    fopen = open(os.path.join(wpath,kmfile),'rb')
    km    = pkl.load(fopen)
    fopen.close()


    # -- utilities
    labs       = WindowLabels(hand=True, nopos=True) # window labels
    nrow, ncol = labs.labels.shape
    maps       = np.zeros([km.n_clusters,nrow,ncol],dtype=np.uint8)


    # -- loop through clusters and generate the maps
    for iclus in range(km.n_clusters):
        print("DST_KMEANS: generating map for cluster {0}...".format(iclus+1))

        iwin = [i for i,j in enumerate(km.labels_) if j==iclus]

        for ii in iwin:
            maps[iclus] += (iclus+1)*(labs.labels==(ii+1))

        outfile = 'kmeans_night_' + str(night).zfill(2) + '_' + \
            str(band) + '_labels_' + str(iclus).zfill(2) + '.out'

        print("DST_KMEANS: writing labels map to file")
        print("DST_KMEANS:   path = {0}".format(wpath))
        print("DST_KMEANS:   file = {0}".format(outfile))

        fopen = open(os.path.join(wpath,outfile),'wb')
        fopen.write(maps[iclus])
        fopen.close()


    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def read_kmeans_maps(night,band):

    """ Read in the K-Means labels maps """

    # -- read in the K-Means file
    kmfile = 'kmeans_night_' + str(night).zfill(2) + '_' + str(band) + '.pkl'
    wpath  = os.environ['DST_WRITE']

    print("DST_KMEANS: reading in K-Means file")
    print("DST_KMEANS:   path = {0}".format(wpath))
    print("DST_KMEANS:   file = {0}".format(kmfile))

    fopen = open(os.path.join(wpath,kmfile),'rb')
    km    = pkl.load(fopen)
    fopen.close()


    # -- utilities
    labs       = WindowLabels(hand=True, nopos=True) # window labels
    nrow, ncol = labs.labels.shape
    maps       = np.zeros([km.n_clusters,nrow,ncol],dtype=np.uint8)


    # -- loop through and read
    for iclus in range(km.n_clusters):
        mfile = os.path.join(os.environ['DST_WRITE'],
                             'kmeans_night_' + str(night).zfill(2) + '_' +
                             str(band) + '_labels_' + str(iclus).zfill(2) + 
                             '.out')

        fopen = open(mfile,'rb')
        maps[iclus] = np.fromfile(fopen, dtype=np.uint8, count=-1
                                  ).reshape(nrow,ncol)
        fopen.close()

    return maps


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 

def kmeans_labels_overlay(night,band,cluster,cmap='bone',
                          wincolor='#0099FF'):

    """ Plot an overlay of the K-Means tagged windows """

    # -- get the maps of the cluster tags
    maps = read_kmeans_maps(night,band)


    # -- get an image
    bkg = np.ma.array(read_raw('oct08_2013-10-25-175504-181179.raw',
                               os.path.join(os.environ['DST_DATA'],
                                            '11/15/16.23.43')
                               )[20:-20,20:-20,:].astype(np.float).mean(2))


    # -- set the mask
    bkg.mask = maps[cluster]>0


    # -- plot
    mn = bkg.min() + 0.2*np.abs(bkg.min())
    mx = bkg.max() - 0.2*np.abs(bkg.max())

    color = cm.get_cmap(cmap)
    color.set_bad(color=wincolor)

    plt.figure()
    plt.imshow(bkg,cmap=color,clim=[mn,mx])
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

    return
