import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from .dst_io import *
from .dst_light_curves import *
from .dst_time_ticks import *

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


def make_plots():

    """ Run all plots """

    # -- light curve matrix and sorted
    lcs01 = LightCurves('','',infile='lcs_night_01',noerr=True)
    km01  = 'kmeans_night_01'

    lc_matrix_plot(lcs01,'lc_matrix_01.png',
                   title='19:00 10/27 to 05:00 10/28')

    lc_matrix_plot(lcs01,'lc_matrix_01_srt.png', km=km01,
                   title='19:00 10/27 to 05:00 10/28')

