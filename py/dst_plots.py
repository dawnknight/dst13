import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .dst_io import *
from .dst_light_curves import *
from .dst_time_ticks import *
from .dst_kmeans import *

# -------- 
#  Generate plots for the DST13 lightscape project
#
#  2013/12/16 - Written by Greg Dobler (CUSP/NYU)
# -------- 


# -- some global color utilities
linec = ['#990000','#006600', '#0000FF']
fillc = ['#FF6600','#99C299', '#0099FF']


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def image_hires(inpath='11/28/14.28.45/', 
                infile='oct08_2013-10-25-175504-292724.raw',
                wpath=os.environ['DST_WRITE'], 
                wfile='day_img_112813_1430_hires.png'):

    """ Generate a high resolution png.  Default is a daytime image on
    11/28/13 at ~2:30pm. """

    # -- read in the data
    rpath = os.path.join(os.environ['DST_DATA'],inpath)

    print("DST_PLOTS: reading in image")
    print("DST_PLOTS:   path = {0}".format(rpath))
    print("DST_PLOTS:   file = {0}".format(infile))

    img = read_raw(infile, rpath)


    # -- make the plot
    plt.figure(figsize=[7.50*3.5,3.955*3.5])
    plt.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()


    # -- write to file
    print("DST_PLOTS: writing output file")
    print("DST_PLOTS:   path = {0}".format(wpath))
    print("DST_PLOTS:   file = {0}".format(wfile))

    plt.savefig(os.path.join(wpath,wfile),clobber=True)
    plt.close()

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def lc_matrix_plot(lcs, outfile, km=None, band=0, title=None):

    """ Plot the light curve matrix """

    # -- check input
    if type(lcs)==str:
        lcs = LightCurves('','',infile=lcs,noerr=True)


    # -- utilities
    tcks, htimes = time_ticks()
    bstr         = (['r','g','b','rgb'])[band]
    nwin, ntime  = lcs.lcs.shape[0:2]


    # -- open the figure
    plt.figure(figsize=[10.*float(ntime)/float(nwin),10.])


    # -- if K-Means solution has been passed, get the index sorting
    if km!=None:
        if type(km)==str:
            kmfile = os.path.join(os.environ['DST_WRITE'],
                                  km+'_'+str(band)+'.pkl')
            fopen  = open(kmfile,'rb')
            km     = pkl.load(fopen)
            fopen.close()

        ind   = np.argsort(km.labels_)
        trind = np.where(km.labels_[ind] - 
                         np.roll(km.labels_[ind],
                                 1))[0][1:]+0.5 # transition rows

        for itr in trind:
            plt.plot([0,ntime],[itr,itr],color=fillc[0],lw=1)
    else:
        ind = range(nwin)


    # -- plot the light curves
    if band==3:
        plt.imshow(lcs.lcs[ind])
    else:
        plt.imshow((lcs.lcs[:,:,band])[ind])

    plt.yticks([0,lcs.lcs.shape[0]-1],'')
    plt.xticks(tcks,htimes,rotation=30)

    if title:
        plt.title(title+' ('+bstr+'-band)')


    # -- write to file
    print("DST_PLOTS: writing output file")
    print("DST_PLOTS:   path = {0}".format(os.environ['DST_WRITE']))
    print("DST_PLOTS:   file = {0}".format(outfile))

    plt.show()
    plt.savefig(os.path.join(os.environ['DST_WRITE'], outfile), clobber=True)
    plt.close()

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def kmeans_plot(night, band):

    """ Plot the K-Means cluster center and tagged windows overlay """

    # -- read in the K-Means file
    kmfile = 'kmeans_night_' + str(night).zfill(2) + '_' + str(band) + '.pkl'
    wpath  = os.environ['DST_WRITE']

    print("DST_PLOTS: reading in K-Means file")
    print("DST_PLOTS:   path = {0}".format(wpath))
    print("DST_PLOTS:   file = {0}".format(kmfile))

    fopen = open(os.path.join(wpath,kmfile),'rb')
    km    = pkl.load(fopen)
    fopen.close()


    # -- utilities
    nclus, ntime = km.cluster_centers_.shape
    tcks, htimes = time_ticks()

    outname = 'kmeans_night_' + str(night).zfill(2) + '_' + str(band)
    nclus  = km.n_clusters
    outp   = os.environ['DST_WRITE']
    outn   = [outname + '_' + str(i+1).zfill(2) + '.png' for i in range(nclus)]
    times  = (np.arange(ntime)*10.)/60.
    clmin  = km.cluster_centers_.min()
    clmax  = km.cluster_centers_.max()
    maps   = read_kmeans_maps(night,band)


    # -- get the background image
    bkg = np.ma.array(read_raw('oct08_2013-10-25-175504-181179.raw',
                               os.path.join(os.environ['DST_DATA'],
                                            '11/15/16.23.43')
                               )[20:-20,20:-20,:].astype(np.float).mean(2))


    # -- set limits and colors
    mn    = bkg.min() + 0.2*np.abs(bkg.min())
    mx    = bkg.max() - 0.2*np.abs(bkg.max())
    color = cm.get_cmap('bone')
    color.set_bad(color='#FF6600')


    # -- make the plot
    for i in range(nclus):

        bkg.mask = maps[i]>0 # set window labels

        outfile = os.path.join(outp,outn[i])

        print("DST_PLOTS: writing file {0}".format(outfile))

        plt.figure(figsize=[7.5,8.5])
        plt.subplot(212)
        if band==3:
            plt.fill_between(range(ntime/3), km.cluster_centers_[i,0:ntime/3],
                             color='#990000',alpha=0.17)
            plt.fill_between(range(ntime/3), 
                             km.cluster_centers_[i,ntime/3:2*ntime/3],
                             color=fillc[1],alpha=0.17)
            plt.fill_between(range(ntime/3), 
                             km.cluster_centers_[i,2*ntime/3:ntime],
                             color=fillc[2],alpha=0.17)
            plt.plot(range(ntime/3), km.cluster_centers_[i,0:ntime/3],
                             color=linec[0])
            plt.plot(range(ntime/3), 
                             km.cluster_centers_[i,ntime/3:2*ntime/3],
                             color=linec[1])
            plt.plot(range(ntime/3), 
                             km.cluster_centers_[i,2*ntime/3:ntime],
                             color=linec[2])
        else:
            plt.fill_between(range(ntime),km.cluster_centers_[i],clmin,
                             color=fillc[2],alpha=0.5)
            plt.plot(range(ntime), km.cluster_centers_[i],color=linec[2],lw=2)
        plt.xticks(tcks, htimes)
        plt.xlabel('time [HH:MM]')
        plt.ylabel('amplitude [arb units]')
        plt.figtext(0.15,0.5,'cluster #' + str(i+1), fontsize=15)
        plt.grid(b=1)
        plt.xlim([0,3600])
        plt.ylim([clmin,clmax])

        plt.subplot(211)
        plt.imshow(bkg,cmap=color,clim=[mn,mx])
        plt.axis('off')

        plt.savefig(outfile, clobber=True)
        plt.close()


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def make_plots():

    """ Run all plots """

    # -- light curve matrix and sorted
    lcs01 = LightCurves('','',infile='lcs_night_01',noerr=True)
    km01  = 'kmeans_night_01'

    lc_matrix_plot(lcs01,'lc_matrix_01.png',
                   title='19:00 10/27 to 05:00 10/28')

    lc_matrix_plot(lcs01,'lc_matrix_01_srt.png', km=km01,
                   title='19:00 10/27 to 05:00 10/28')


    # -- K-means clusters
    for night in range(22):
        for band in range(4):
            dst13.kmeans_plot(night,band)


    return
