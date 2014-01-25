import dst13
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf

def canny1d(lcs, indices=None, width=30, delta=2, see=False, sig_clip_iter=10, 
            sig_clip_amp=2.0, sig_peaks=10.0):

    # -- defaults
    if indices==None:
        nwin    = lcs.lcs.shape[0]
        indices = range(nwin)
        print("DST_CANNY1D: running edge detector for all " + 
              "{0} windows...".format(nwin))
    else:
        nwin = len(indices)
        print("DST_CANNY1D: running edge detector for " + 
              "{0} windows...".format(nwin))


    # -- utilities
    lcg       = np.zeros(lcs.lcs.shape[1:])
    dlcg      = np.ma.zeros(lcg.shape)
    dlcg.mask = dlcg>314
    ind_onoff = []
    ints      = np.arange(lcs.lcs.shape[0])


    # -- loop through windows
    for ii, index in enumerate(indices):
        if ii%100==0:
            print("DST_CANNY1D:   {0} of {1}".format(ii,nwin))

        # -- smooth each band
        for band in [0,1,2]:
            lcg[:,band] = gf(lcs.lcs[index,:,band],width)


        # -- compute Gaussian difference and set mask edges
        dlcg[:,:]          = np.roll(lcg,-delta,0)-np.roll(lcg,delta,0)
        dlcg.mask[:width]  = True
        dlcg.mask[-width:] = True


        # -- plot
        if see:
            plt.figure(6)
            plt.clf()
            plt.subplot(2,2,2)
            plt.plot(dlcg[:,0], lw=2)
            plt.ylim([1.2*dlcg.min(),1.2*dlcg.max()])
            plt.subplot(2,2,3)
            plt.plot(dlcg[:,1], lw=2)
            plt.ylim([1.2*dlcg.min(),1.2*dlcg.max()])
            plt.subplot(2,2,4)
            plt.plot(dlcg[:,2], lw=2)
            plt.ylim([1.2*dlcg.min(),1.2*dlcg.max()])

        # -- sigma clip
        for _ in range(10):
            avg = dlcg.mean(0)
            sig = dlcg.std(0)
            dlcg.mask = np.abs(dlcg-avg) > sig_clip_amp*sig

            if see:
                plt.subplot(2,2,2)
                plt.plot(dlcg[:,0], lw=2)
                plt.subplot(2,2,3)
                plt.plot(dlcg[:,1], lw=2)
                plt.subplot(2,2,4)
                plt.plot(dlcg[:,2], lw=2)


        # -- set mean and std and reset the mask
        avg                = dlcg.mean(0)
        sig                = dlcg.std(0)
        dlcg.mask[:,:]     = False
        dlcg.mask[:width]  = True
        dlcg.mask[-width:] = True


        # -- find peaks in RGB
        ind_on_rgb, ind_off_rgb = [], []

        tags_on  = (dlcg-avg > sig_peaks*sig) & \
            (dlcg>np.roll(dlcg,1,0)) & \
            (dlcg>np.roll(dlcg,-1,0)) & \
            ~dlcg.mask

        tags_off = (dlcg-avg < -sig_peaks*sig) & \
            (dlcg<np.roll(dlcg,1,0)) & \
            (dlcg<np.roll(dlcg,-1,0)) & \
            ~dlcg.mask

        for band in [0,1,2]:
            ind_on_rgb.append([i for i in ints[tags_on[:,band]]])
            ind_off_rgb.append([ i for i in ints[tags_off[:,band]]])


        # -- collapse RGB indices
        for iind in ind_on_rgb[0]:
            for jind in ind_on_rgb[1]:
                if abs(iind-jind)<=2:
                    ind_on_rgb[1].remove(jind)
            for jind in ind_on_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_on_rgb[2].remove(jind)

        for iind in ind_on_rgb[1]:
            for jind in ind_on_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_on_rgb[2].remove(jind)

        ind_on_list = ind_on_rgb[0] + ind_on_rgb[1] + ind_on_rgb[2]

        for iind in ind_off_rgb[0]:
            for jind in ind_off_rgb[1]:
                if abs(iind-jind)<=2:
                    ind_off_rgb[1].remove(jind)
            for jind in ind_off_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_off_rgb[2].remove(jind)

        for iind in ind_off_rgb[1]:
            for jind in ind_off_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_off_rgb[2].remove(jind)

        ind_off_list = ind_off_rgb[0] + ind_off_rgb[1] + ind_off_rgb[2]



        # -- add to on/off list
        tind_onoff = np.array([i for i in ind_on_list+[-j for j in 
                                                        ind_off_list]])

        ind_onoff.append(tind_onoff[np.argsort(np.abs(tind_onoff))])


#        if see:
#            plt.subplot(2,2,2)
#            plt.plot(np.arange(dlcg.shape[0])[on_ind[:,0]],
#                     dlcg[on_ind[:,0],0], 'go')


    return ind_onoff
