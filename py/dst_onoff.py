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
def onoff(data, thresh=5.0, fsize=20, see=False, write=False):

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


    # -- initialize light curve arrays
    lc   = np.ma.zeros([3,lcs.shape[1]])
    lc_m = np.ma.zeros([3,lcs.shape[1]])
    lc_g = np.ma.zeros([3,lcs.shape[1]])
    sig  = np.zeros(3)
    avg  = np.zeros(3)


    # -- initialize the on and off lists
    ind_on  = []
    ind_off = []


    # -- loop through windows 
    if see or write:
        plt.figure(3,figsize=[7,10])
        plt.grid(b=1)

    for iwin in range(100,200):

        # -- pull out the three bands and median filter for plotting
        lc = np.ma.array(lcs[iwin].T)

        for ib in (0,1,2):
            lc_m[ib,:] = nd.filters.median_filter(lc[ib], fsize)


        # -- mask bad images
        lc.mask = lc < 1e-6


        # -- median filter the gradient
        lc_g[:,:] = np.roll(lc,1,1) - np.roll(lc,-1,1)


        # -- make sure the gradient mask is the right size
        if lc_g.mask.size==1:
            lc_g.mask = lc.mask


        # -- get the standard deviation, and mean
        sig[:]    = lc_g.std(1)
        avg[:]    = lc_g.mean(1)


        # -- define the on and off times
        glo = avg - thresh*sig
        ghi = avg + thresh*sig

        iwin_on = np.where(
            (lc_g[0] < (glo[0])) & 
            (lc_g[1] < (glo[1])) & 
            (lc_g[2] < (glo[2])) & 
            ~lc_g.mask[0]
            )[0]

        iwin_off = np.where(
            (lc_g[0] > (ghi[0])) & 
            (lc_g[1] > (ghi[1])) & 
            (lc_g[2] > (ghi[2])) & 
            ~lc_g.mask[0]
            )[0]


        # -- append to lists
        ind_on.append(iwin_on)
        ind_off.append(iwin_off)


        # -- plot if desired
        if see or write:
            plt.clf()
            plt.subplot(211)
            plt.ylabel(r'$\bigtriangledown$ intensity [arb units]')
            plt.figtext(0.12,0.91,'window ID: '+str(iwin),fontsize=15)

            sml = lc_g[0,iwin_on]
            big = lc_g[0,iwin_off]
            off = [-30,0,30]

            plt.ylim([-50,50])

            plt.plot(lc_g[0]+off[0],linec[0])
            plt.plot([0,lc_g[0].size],[glo[0]+off[0],glo[0]+off[0]],
                     '--',color='k')
            plt.plot([0,lc_g[0].size],[ghi[0]+off[0],ghi[0]+off[0]], 
                     '--',color='k')

            plt.plot(lc_g[1]+off[1],fillc[1])
            plt.plot([0,lc_g[1].size],[glo[1]+off[1],glo[1]+off[1]],
                     '--',color='k')
            plt.plot([0,lc_g[1].size],[ghi[1]+off[1],ghi[1]+off[1]], 
                     '--',color='k')
            plt.plot(iwin_on,sml+off[1],'k+',ms=20)
            plt.plot(iwin_off,big+off[1],'k+',ms=20)

            plt.plot(lc_g[2]+off[2],fillc[2])
            plt.plot([0,lc_g[2].size],[glo[2]+off[2],glo[2]+off[2]],
                     '--',color='k')
            plt.plot([0,lc_g[2].size],[ghi[2]+off[2],ghi[2]+off[2]], 
                     '--',color='k')


            plt.subplot(212)

            plt.xlabel('time [10s]')
            plt.ylabel('intensity [arb units]')

            plt.fill_between(np.arange(lc_m[0].size), lc_m[0], 
                             0.5*(lc_m[0].max()+lc_m[0].min()),
                             facecolor=fillc[0],alpha=0.5)
            plt.plot(np.arange(lc_m[0].size)/1.,lc_m[0],linec[0])

            for i in iwin_on:
                plt.plot([i,i],[20,110],linec[0])
            for i in iwin_off:
                plt.plot([i,i],[20,110],linec[2])

            plt.draw()

            if see:
                plt.show()
                time.sleep(1)
            else:
                wpath = os.path.join(os.environ['DST_WRITE'],'onoff')
                wfile = 'onoff_' + str(iwin).zfill(4) + '.png'
                print("DST_ONOFF: writing plot to file {0}")
                print("DST_ONOFF:   PATH - {0}".format(wpath))
                print("DST_ONOFF:   FILE - {0}".format(wfile))

                plt.savefig(os.path.join(wpath,wfile), clobber=True)

    # -- return the on/off indices
    return ind_on, ind_off
