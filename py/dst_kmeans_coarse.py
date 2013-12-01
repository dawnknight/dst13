import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from .dst_io import *
from .dst_imtools import *
from .dst_night_times import *

# -------- 
#  coarse K-Means clustering on whole image to reproduce Brumby's plots
#
#  2013/11/25 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def kmeans_coarse(start,end):

    """ K-Means clustering on a la Brumby 2013-04-10 (see cusp.pro) """

    # -- utilities
    nrow = 2160
    ncol = 4096
    fac  = 4
    nrow_bin = nrow/fac
    ncol_bin = ncol/fac

    # -- get the file list
    fopen = open(os.path.join(os.environ['DST_WRITE'],'filelist.pkl'),'rb')
    fl    = pkl.load(fopen)
    fopen.close()


    # -- pull out a time slice
    paths, files, times = fl.time_slice(start,end)


    # -- initialize features matrix (r-band only), img container, and ones
    lcs = np.zeros([nrow_bin*ncol_bin,files[::25].size])
    img = np.zeros([nrow_bin*ncol_bin])
    ons = np.ones(img.shape)


    # -- rebin (r-band only), divide by L2-norm, and insert
    for i, (p,f,t) in enumerate(zip(paths[::25],files[::25],times[::25])):
        if (i+1)%5==0:
            print("DST_KMEANS_COARSE: Loading file " + 
                  "{0} of {1}".format(i+1,lcs.shape[1]))
        img[:]   = rebin(read_raw(f,p).astype(np.float)[:,:,0],fac).flatten()
        img[:]  /= np.max([np.sqrt(np.dot(img**2,ons)),1e-6])
        lcs[:,i] = img


    # -------- kmeans
    print("DST_KMEANS_COARSE: Running K-Means...")
    kmeans_coar = KMeans(init='random', n_clusters=12, n_init=10)
    kmeans_coar.fit(lcs)


    return kmeans_coar



# -------- 
#  plot the K-Means results
# -------- 
def kmeans_coarse_plot(kfile,outname):

    """ Plot the coarse K-Means results """

    # -- read in the kmeans file and extract the cluster centers
    fopen  = open(kfile,'rb')
    kmcoar = pkl.load(fopen)
    fopen.close()

    nclus, ntime = kmcoar.cluster_centers_.shape


    # -- utilities
    outp   = os.environ['DST_WRITE']
    outn   = [outname + '_' + str(i+1).zfill(2) + '.png' for i in range(nclus)]
    times  = (np.arange(ntime)*250.)/60.
    htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]
    labs   = np.zeros(540*1024,dtype=np.int)
    clmin  = kmcoar.cluster_centers_.min()
    clmax  = kmcoar.cluster_centers_.max()


    # -- get the labels map
    for i in range(1,nclus):
        labs[kmcoar.labels_==i] = i


    # -- plot
    linec = '#990000'
    fillc = '#F5B473'

    for i in range(nclus):

        outfile = os.path.join(outp,outn[i])

        print("DST_KMEANS_COARSE: writing file {0}".format(outfile))

        plt.figure(figsize=[7.5,8.5])
        plt.subplot(212)
        plt.fill_between(times,kmcoar.cluster_centers_[i],clmin,color=fillc)
        plt.plot(times,kmcoar.cluster_centers_[i],color=linec,lw=2)
        plt.xticks([60.*j for j in range(10+1)], htimes)
        plt.xlabel('time [HH:MM]')
        plt.ylabel('amplitude [arb units]')
        plt.figtext(0.15,0.5,'cluster #' + str(i+1), fontsize=15)
        plt.grid(b=1)
        plt.xlim([0,600])
        plt.ylim([clmin,clmax])

        plt.subplot(211)
        plt.imshow(labs.reshape([540,1024])==i)
        plt.axis('off')

        plt.savefig(outfile, clobber=True)
        plt.close()



# -------- 
#  run coarse K-Means for each night
# -------- 
def kmeans_coarse_run(wpath=os.environ['DST_WRITE']):

    """ Run the coarse K-Means for the night time runs """

    # -- get the night times
    start, end = night_times()


    # -- loop through nights, compute K-Means, and write to file
    for i, (st,en) in enumerate(zip(start,end)):
        print("DST_KMEANS_COARSE_RUN: Running K-Means for night {0}".format(i))
        print("DST_KMEANS_COARSE_RUN:   {0}  {1}".format(st,en))

        km    = kmeans_coarse(st,en)
        wfile = os.path.join(wpath,'kmean_coarse_night_' + 
                             str(i).zfill(2) + '.pkl')

        print("DST_KMEANS_COARSE_RUN: Writing solution to")
        print("DST_KMEANS_COARSE_RUN:   {0}".format(wfile))

        fopen = open(wfile,'wb')
        pkl.dump(km,fopen)
        fopen.close()


    return
