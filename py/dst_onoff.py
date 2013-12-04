import os, time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# -------- 
#  Find the on/off transitions for windows
#
#  2013/12/04 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def onoff(data, thresh=5.0, see=False):

    # -- utilities
    linec    = ['#990000','#006600', '#0000FF']
    fillc    = ['#FF6600','#99C299', '#0099FF']


    # -- set the data
    if type(data) is str:
        infile = os.path.join(os.environ['DST_WRITE'],data+'_lcs.pkl')

        print("DST_ONOFF: Reading light curves from")
        print("DST_ONOFF:   {0}".format(infile))
        fopen  = open(infile, 'rb')
        lcs    = pkl.load(fopen)
        fopen.close()
    else:
        lcs = data


    # -- initialize median filtered arrays
    lc_m = np.ma.zeros([3,lcs.shape[1]])
    lc_g = np.ma.zeros([3,lcs.shape[1]])
    sig  = np.zeros(3)
    avg  = np.zeros(3)


    # -- initialize the on and off lists
    ind_on  = []
    ind_off = []


    # -- loop through windows 
    if see:
        plt.figure(3,figsize=[7,10])
        plt.grid(b=1)

    for iwin in range(100,200):

        # -- pull out the three bands and median filter
        for ib in (0,1,2):
            lc_m[ib,:] = np.ma.array(
                nd.filters.median_filter(
                    lcs[iwin,:,ib],20
                    )
                )


        # -- mask bad images
        lc_m.mask = lc_m < 1e-6


        # -- set the gradient, standard deviation, and mean
        lc_g[:,:] = np.roll(lc_m,1,1) - lc_m
        sig[:]    = lc_g.std(1)
        avg[:]    = lc_g.mean(1)


        # -- define the on and off times
        iwin_on = np.where(
            (lc_g[0] < (avg[0]-thresh*sig[0])) & 
            (lc_g[1] < (avg[1]-thresh*sig[1])) & 
            (lc_g[2] < (avg[2]-thresh*sig[2])) & 
            ~lc_g.mask[0]
            )[0]

        iwin_off = np.where(
            (lc_g[0] > (avg[0]+thresh*sig[0])) & 
            (lc_g[1] > (avg[1]+thresh*sig[1])) & 
            (lc_g[2] > (avg[2]+thresh*sig[2])) & 
            ~lc_g.mask[0]
            )[0]


        # -- append to lists
        ind_on.append(iwin_on)
        ind_off.append(iwin_off)


        # -- plot if desired
        if see:
            plt.clf()
            plt.subplot(211)

            sml = lc_g[0,iwin_on]
            big = lc_g[0,iwin_off]

            plt.plot(lc_g[0],linec[1])
            plt.plot(iwin_on,sml,'k+',ms=20)
            plt.plot(iwin_off,big,'k+',ms=20)
            plt.grid(b=1)
            plt.ylim([-20,20])

            plt.subplot(212)

            plt.fill_between(np.arange(lc_m[0].size), lc_m[0], 
                             0.5*(lc_m[0].max()+lc_m[0].min()),
                             facecolor=fillc[2],alpha=0.5)
            plt.plot(np.arange(lc_m[0].size)/1.,lc_m[0],linec[2])

            for i in iwin_on:
                plt.plot([i,i],[20,110],linec[0])
            for i in iwin_off:
                plt.plot([i,i],[20,110],linec[1])

            plt.draw()
            plt.show()

            time.sleep(1)


    # -- return the on/off indices
    return ind_on, ind_off
