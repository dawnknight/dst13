import pickle as pkl
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks_cwt
from matplotlib.pyplot import *


# -------- 
#  Identify all ON/OFF transitions from step function fits to light curve 
#  matrices.
#
#  2013/12/17 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def fit_step(lcs, width=180, smooth=False, see=False, wnum=None, 
             wrng=None, chism=None, xcheck=True):

    # -- utilities
    npix      = width
    nband     = 3
    step_l    = np.zeros(npix)
    step_r    = np.zeros(npix)
    offset    = np.ones(npix)
    tvals     = np.arange(float(npix))
    dvals     = np.ma.zeros(npix)
    ilc_mn    = wnum if wnum!=None else 0
    ilc_mx    = wnum+1 if wnum!=None else lcs.lcs.shape[0]
    lc        = np.ma.zeros(lcs.lcs[0].shape)
    mask      = np.zeros(lcs.lcs[0].shape)
    chisq_thr = 1.0
    ind_onoff = []

    if wrng!=None:
        ilc_mn, ilc_mx = wrng


    # -- set the step function
    step_l[:npix/2] = 1.0
    step_r[npix/2:] = 1.0


    # -- smooth if desired
    if smooth:
        tvals  = gaussian_filter(tvals,smooth)
        step_l = gaussian_filter(step_l,smooth)
        step_r = gaussian_filter(step_r,smooth)


    # -- generate the two models and initialize some utilities
    tmpl_1 = np.vstack([tvals,step_l,step_r])
    tmpl_2 = np.vstack([tvals,offset])

    ptpinv_1 = np.linalg.inv(np.dot(tmpl_1,tmpl_1.T))
    ptpinv_2 = np.linalg.inv(np.dot(tmpl_2,tmpl_2.T))

    dpt_1 = tmpl_1.shape[0]
    dpt_2 = tmpl_2.shape[0]

    mvals_1 = np.zeros(npix)
    mvals_2 = np.zeros(npix)


    # -- find the range of analysis and initialize chisq arrays
    ioff    = lc.shape[0] % npix
    imax    = lc.shape[0]-ioff-npix
    chisq_1 = np.zeros([nband,imax])
    chisq_2 = np.zeros([nband,imax])
    onoff   = np.zeros([nband,imax],dtype=np.int8)


    # -- loop through light curves
    for ilc in range(ilc_mn,ilc_mx):

        # -- alert the user
        print("DST_FIT_STEP: running lightcurve " + 
              "{0} of {1}...".format(ilc+1,ilc_mx))

        # -- set the light curve
        lc[:,:] = np.ma.array(lcs.lcs[ilc])
        lc.mask = (lc < 1.0)

        if smooth:
            for i in (0,1,2):
                mask[:,i] = gaussian_filter(1.0*lc.mask[:,i],smooth)

            for i in (0,1,2):
                lc[:,i] = gaussian_filter(lc[:,i],smooth)

            lc.mask = mask > 0.01

        # -- estimate the noise
        dlc = (np.roll(lc,1,0)-lc)[1:-1].T#/pixfac
        avg = dlc.mean(1)
        sig = dlc.std(1)

        for _ in (0,1,2):
            thresh = (avg+5*sig)#.clip(1.0,1e6)
            w = np.where((dlc[0]<thresh[0]) &
                         (dlc[1]<thresh[1]) &
                         (dlc[2]<thresh[2]) 
                         )[0]
            avg = np.array([dlc[i][w].mean() for i in (0,1,2)])
            sig = np.array([dlc[i][w].std() for i in (0,1,2)])

        noise = sig/np.sqrt(2.0)

        # -- get a slice of the light curve and calculate chisq
        for iband in (0,1,2):
            for ii in range(imax):

                dvals[:] = lc[ii:ii+npix,iband]

                if True in dvals.mask:
                    continue

                avec            = np.dot(np.dot(tmpl_1,dvals),ptpinv_1)
                onoff[iband,ii] = 1 if avec[2]>avec[1] else -1

                chisq_1[iband,ii] = ((dvals - np.dot(avec,tmpl_1))**2
                                     ).sum()/(noise[iband]**2)/(float(npix)-3)

                chisq_2[iband,ii] = ((dvals - np.dot(np.dot(
                                np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2))**2
                                     ).sum()/(noise[iband]**2)/(float(npix)-2)

        # -- commpute Delta chi^2 and set thresholds
        if chism!=None:
            chisq_1 = gaussian_filter(chisq_1,chism)
            chisq_2 = gaussian_filter(chisq_2,chism)

        dif = chisq_2 - chisq_1
        avg = dif.mean(1)
        sig = dif.std(1)

        # -- threshold on Delta chi^2
#        if dif.max()<chisq_thr:
#            ind_onoff.append(np.array([],dtype=np.int))
#
#            print "bailing out, delta chi^2 is too small..."
#            continue

        # -- outlier rejection
        for _ in range(20):
            thresh = (avg+5*sig)#.clip(1.0,1e6)
            w = np.where((dif[0]<thresh[0]) &
                         (dif[1]<thresh[1]) &
                         (dif[2]<thresh[2])
                         )[0]
            if w.size>0:
                avg = np.array([dif[i][w].mean() for i in (0,1,2)])
                sig = np.array([dif[i][w].std() for i in (0,1,2)])

        thresh = (avg+20*sig)#.clip(1.0,1e6)

        # -- add outliers to list
        w = np.where(
            (dif[0] > thresh[0]) &
            (dif[1] > thresh[1]) &
#            (dif[2] > thresh[2]) &
#            (dif[0] > chisq_thr) &
#            (dif[1] > chisq_thr) &
#            (dif[2] > chisq_thr)
            (dif[2] > thresh[2])
            )[0]

        # -- check for transitions
        if len(w)>0:

            # -- find the peaks
            bound = [0] + list(np.where((w- np.roll(w,1))>10)[0]) + [w.size]
            peaks = [w[bound[i]] + npix/2 + \
                         np.argmax(dif[0,w[bound[i]:bound[i+1]]]) \
                         for i in range(len(bound)-1)]

            # -- check left and right
            if xcheck:
                for ip in peaks:
                    mn_l  = lc[ip-10:ip].mean(1).mean()
                    err_l = lc[ip-10:ip].mean(1).std()/np.sqrt(10.)
                    mn_r  = lc[ip:ip+10].mean(1).mean()

                    if abs(mn_r-mn_l)<2.0*err_l:
                        peaks.remove(ip)

            # -- add to list
            peaks = np.array(peaks)
            ind_onoff.append(peaks*onoff[0,peaks-npix/2])
        else:
            ind_onoff.append(np.array([],dtype=np.int))

    return ind_onoff


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def fit_step_plot():

    # -- utilities
    linec  = ['#990000','#006600', '#0000FF']
    fillc  = ['#FF6600','#99C299', '#0099FF']



    mx = lc.max()
    mn = lc.min()

    htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]

    figure(1, figsize=[10.0,10.])
    clf()

    off = np.array([30,70,110]) - lc.mean(0)

    subplot(221)
    plot(lc[:,0]+off[0],linec[0])
    plot(lc[:,1]+off[1],fillc[1])
    plot(lc[:,2]+off[2],fillc[2])
    xlim([0,3600])

    xticks([360.*j for j in range(10+1)], htimes, rotation=30.)
    ylabel('intensity [arb. units]')
    figtext(0.3,0.86,'window #'+str(ilc),fontsize=15,backgroundcolor='w')
    figtext(0.3,0.86,'window #'+str(ilc),fontsize=15)

    subplot(222)
    fill_between(np.arange(dif.shape[1])+npix/2,dif[0],facecolor=linec[0],
                 edgecolor=linec[0])
    xlim([0,3600])

    ymax = np.max([2*thresh[0],1.2*dif[0].max()])
    ylim([0.0,ymax])
    plot([0,3600],[thresh[0],thresh[0]],color='#EE4400')
    plot([0,3600],[avg[0]+5*sig[0],avg[0]+5*sig[0]],color='#EE4400')
    plot(w+npix/2,(dif[0])[w],'k+',ms=20)
    xticks([360.*j for j in range(10+1)], htimes, rotation=30.)
    text(2750,0.87*ymax,r'$\Delta \chi^2_{R}$',fontsize=20)
    text(3250,1.05*(avg[0]+5*sig[0]),r'$5\sigma$',fontsize=15)
    text(3250,1.05*(avg[0]+10*sig[0]),r'$10\sigma$',fontsize=15)

    subplot(223)
    fill_between(np.arange(dif.shape[1])+npix/2,dif[1],facecolor=fillc[1],
                 edgecolor=fillc[1])
    xlim([0,3600])

    ymax = np.max([2*thresh[1],1.2*dif[1].max()])
    ylim([0.0,ymax])
    plot([0,3600],[thresh[1],thresh[1]],color='#EE4400')
    plot([0,3600],[avg[1]+5*sig[1],avg[1]+5*sig[1]],color='#EE4400')
    plot(w+npix/2,(dif[1])[w],'k+',ms=20)
    xticks([360.*j for j in range(10+1)], htimes, rotation=30.)
    text(2750,0.87*ymax,r'$\Delta \chi^2_{G}$',fontsize=20)
    text(3250,1.05*(avg[1]+5*sig[1]),r'$5\sigma$',fontsize=15)
    text(3250,1.05*(avg[1]+10*sig[1]),r'$10\sigma$',fontsize=15)


    subplot(224)
    fill_between(np.arange(dif.shape[1])+npix/2,dif[2],facecolor=fillc[2],
                 edgecolor=fillc[2])
    xlim([0,3600])

    ymax = np.max([2*thresh[2],1.2*dif[2].max()])
    ylim([0.0,ymax])
    plot([0,3600],[thresh[2],thresh[2]],color='#EE4400')
    plot([0,3600],[avg[2]+5*sig[2],avg[2]+5*sig[2]],color='#EE4400')
    plot(w+npix/2,(dif[2])[w],'k+',ms=20)
    xticks([360.*j for j in range(10+1)], htimes, rotation=30.)
    text(2750,0.87*ymax,r'$\Delta \chi^2_{B}$',fontsize=20)
    text(3250,1.05*(avg[2]+5*sig[2]),r'$5\sigma$',fontsize=15)
    text(3250,1.05*(avg[2]+10*sig[2]),r'$10\sigma$',fontsize=15)


    draw()

#print("DST_FIT_STEP: writing window # {0} to png".format(ilc))
#savefig('../output/fitstep/fitstep_night_'+str(night).zfill(2)+'_'+
#        str(ilc).zfill(4)+'.png', clobber=True)

