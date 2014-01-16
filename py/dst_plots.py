import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import median_filter as mf
from scipy.ndimage.filters import gaussian_filter as gf
from .dst_io import *
from .dst_light_curves import *
from .dst_time_ticks import *
from .dst_kmeans import *
from .dst_night_times import *

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
    plt.figure(figsize=[7.5,7.5])
    plt.subplots_adjust(0.07,0.07,0.93,0.93)


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
    plt.xlim([0,3600])
    plt.clim([16,255])

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

        return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def drift_plot():

    # -- utilities
    wpath      = os.environ['DST_WRITE']
    regfile    = 'registration_dictionary.pkl'
    start, end = night_times()
    nnight     = len(start)
    dr, dc, tr = [], [], []


    # -- get file list
    print("DST_DRIFT: reading file list...")
    fopen = open(os.path.join(wpath,'filelist.pkl'),'rb')
    fl    = pkl.load(fopen)
    fopen.close()


    # -- read in the registration dictionary
    print("DST_DRIFT: reading registration dictionary...")
    fopen  = open(os.path.join(wpath,regfile),'rb')
    cc_dic = pkl.load(fopen)
    fopen.close()


    # -- loop through nights
    for inight in range(nnight):

        print("DST_DRIFT: night number {0}...".format(inight))

        paths, files, times = fl.time_slice(start[inight],end[inight])

        tr.append(len(times))
        dr.append([])
        dc.append([])

        for itime, (p,f,t) in enumerate(zip(paths,files,times)):
            dri, dci = cc_dic[f]

            dr[inight].append(dri)
            dc[inight].append(dci)


    # -- plot
    plt.figure(figsize=[10,6])
    plt.ylim([-40,40])
    plt.plot([item for sublist in dr for item in sublist])
    plt.plot([item for sublist in dc for item in sublist])
    for itr in np.cumsum(tr):
        plt.plot([itr,itr],[-40,40],'--',color='#0099FF')
    plt.xlim([0,itr])
    plt.grid(b=1)
    plt.xlabel('time step [10s]', size=15)
    plt.ylabel(r'$\Delta$ pixel', size=15)
    plt.text(100, 41.5, 'Required registration shift: ' +
             'red=vertical, yellow=horizontal', ha='left', size=17)
    plt.show()
    plt.savefig(os.path.join(os.environ['DST_WRITE'],
                             'registration_v_time.jpg'), clobber=True)
    plt.close()

    return


# -------- # -------- # -------- # -------- # -------- # -------- # --------


def one_off_dist(night):

    """ Plot of time of off transitions in which off is the first
    transition"""

    # -- utilities
    aoff  = []
    ooff  = []
    ooffp = []
    sind  = []
    tcks, htimes = time_ticks()


    # -- get the on/off times
    ind_onoff = []

    for ii in range(1,10):
        fopen = open(os.path.join(os.environ['DST_WRITE'],
                                  'ind_onoff_night_'+str(night).zfill(2)+
                                  '_'+str(ii)+'.pkl'),'rb')
        ind_onoff += pkl.load(fopen)
        fopen.close()


    # -- pull out useful subsets
    for index,ind in enumerate(ind_onoff):
        if len(ind)==1:
            if ind[0]<0:
                ooff.append(ind)
                sind.append(index)
        elif len(ind)>0:
            if ind[0]<0:
                ooffp.append([ind[0]])
            elif min(ind)<0:
                aoff.append([j for j in ind if j<0])


    # -- make the plot
    plt.figure(0,figsize=[7.5,7.5])
    plt.hist([abs(item) for sublist in aoff for item in sublist], 
             40,range=[0,3600],facecolor=fillc[2])
    plt.hist([abs(item) for sublist in ooffp for item in sublist], 
             40,range=[0,3600],facecolor=fillc[1])
    plt.hist([abs(item) for sublist in ooff for item in sublist], 
             40,range=[0,3600],facecolor=fillc[0])
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.ylim([0,200])
    plt.grid(b=1)
    plt.ylabel('# of OFF transitions', size=15)
    plt.figtext(0.13,0.91,'night '+str(night),size=17)
    plt.savefig(os.path.join(os.environ['DST_WRITE'],'one_off_night_'+
                             str(night).zfill(2)+'.png'),clobber=True)
    plt.close()

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 

def on_vs_time(lcs_dis, color=None, outfile='on_vs_time.png'):

    # -- check for list
    if not isinstance(lcs_dis,list):
        lcs_dis = [lcs_dis]
        color   = [color]


    # -- utilities
    tcks, htimes = time_ticks()
    nlcs = len(lcs_dis)


    # -- make a plot of number of windows "on" vs time
#    plt.figure(0, figsize=[7.5,7.5])
#    plt.xlim([0,3600])
#    plt.ylabel('# of windows on',fontsize=15)
#    plt.grid(b=1)
#    plt.xticks(tcks,htimes,rotation=30)
#
#    for ii in range(len(lcs_dis)):
#        plt.plot((lcs_dis[ii]>0).sum(0),color=color[ii])
#
#    plt.savefig(os.path.join(os.environ['DST_WRITE'],outfile),clobber=True)

    plt.figure(0, figsize=[15.0,7.5])
    plt.subplot(1,2,1)
    plt.xlim([0,3600])
    plt.ylabel('# of windows on',fontsize=15)
    plt.grid(b=1)
    plt.xticks(tcks,htimes,rotation=30)

    for ii in range(len(lcs_dis)):
        plt.plot((lcs_dis[ii]>0).sum(0),color=color[ii])

    plt.subplot(1,2,2)
    plt.xlim([0,3600])
    plt.ylabel('normalized # of windows on',fontsize=15)
    plt.grid(b=1)
    plt.xticks(tcks,htimes,rotation=30)

    for ii in range(len(lcs_dis)):
        lc = (lcs_dis[ii]>0).sum(0)
        plt.plot(lc/float(lc[0]),color=color[ii])

    plt.savefig(os.path.join(os.environ['DST_WRITE'],outfile),clobber=True)

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def res_lc_plot():

    # -- read in the residential data
    lcs = []

    for night in range(22):
        fopen = open(os.path.join(os.environ['DST_WRITE'],'res_lc_night_' + 
                                  str(night).zfill(2) + '_2.pkl'),'rb')
        lcs.append(pkl.load(fopen))
        fopen.close()


    # -- utilities
    tcks, htimes = time_ticks()


    # -- define fri/sat and all other days
#    we = [0,6,7,13,14,20,21]
#    wd = [1,2,3,4,5,8,9,10,11,12,15,16,17,18,19]
    we = [0,6,7,13,14,20,21]
    wd = [1,2,3,4,5,8,9,10,11,12,15,16,17,18,19]


    # -- get mean averaged lc
    tmax = min([len(i.mean(0)) for i in lcs if len(i.mean(0))>3000])
    mnwd = np.zeros(tmax)
    mnwe = np.zeros(tmax)

    cnt = 0.0
    for idy in wd:
        if idy==8: continue
        if len(lcs[idy].mean(0))>3000:
            mnwd += lcs[idy].mean(0)[:tmax]
            cnt += 1.0
    mnwd /= cnt

    cnt = 0.0
    for idy in we:
        if idy==21: continue
        if len(lcs[idy].mean(0))>3000:
            mnwe += lcs[idy].mean(0)[:tmax]
            cnt += 1.0
    mnwe /= cnt

    plt.figure(1,figsize=[5,5])
    plt.xticks(tcks,htimes,rotation=30)
    plt.ylabel('intensity [arb units]',size=15)
    plt.xlim([0,3600])
    plt.grid(b=1)
    plt.plot(mnwd,'k')
    plt.plot(mnwe,'r')
    plt.savefig('../output/res_lc_mean.png',clobber=True)
    plt.close()


    # -- open the figure
    plt.figure(0,figsize=[15,5])
    plt.subplots_adjust(0.05,0.1,0.95,0.95)

    plt.subplot(131)
    for idy in wd: 
#        plt.plot(mf(lcs[idy][0]/lcs[idy][0,0],6))
#        plt.plot(mf(lcs[idy].mean(0)/lcs[idy].mean(0)[0],6))
#        cdf = np.cumsum(lcs[idy].mean(0))
#        plt.plot(cdf/cdf[360]/(np.arange(cdf.size)/360.))
        lc_sm = gf(lcs[idy].mean(0)/lcs[idy].mean(0)[0],6)
        norm = (lc_sm-lc_sm[-1])
#        plt.plot(lc_sm/lc_sm[0])
        plt.plot(norm/norm[0])
#        plt.plot(gf((lc_sm-np.roll(lc_sm,1))[1:-1],360))
#    plt.ylim([0.8,1.2])
#    plt.ylim([0.6,1.4])
    plt.ylim([-0.1,1.4])
    plt.grid(b=1)
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.ylabel('intensity [arb units]',size=15)

    plt.subplot(132)
    for idy in we: 
#        plt.plot(mf(lcs[idy][0]/lcs[idy][0,0],6))
#        plt.plot(mf(lcs[idy].mean(0)/lcs[idy].mean(0)[0],6))
#        cdf = np.cumsum(lcs[idy].mean(0))
#        plt.plot(cdf/cdf[360]/(np.arange(cdf.size)/360.))
        lc_sm = gf(lcs[idy].mean(0)/lcs[idy].mean(0)[0],6)
        norm = (lc_sm-lc_sm[-1])
#        plt.plot(lc_sm/lc_sm[0])
        plt.plot(norm/norm[0])
#        plt.plot(gf((lc_sm-np.roll(lc_sm,1))[1:-1],360))
#    plt.ylim([0.6,1.4])
#    plt.ylim([0.8,1.2])
    plt.ylim([-0.1,1.4])
    plt.grid(b=1)
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.ylabel('intensity [arb units]',size=15)

    plt.subplot(133)
    for idy in wd: 
#        plt.plot(mf(lcs[idy][0]/lcs[idy][0,0],6),'k')
        plt.plot(mf(lcs[idy].mean(0)/lcs[idy].mean(0)[0],6),'k')
    for idy in we: 
#        plt.plot(mf(lcs[idy][0]/lcs[idy][0,0],6),'r')
        plt.plot(mf(lcs[idy].mean(0)/lcs[idy].mean(0)[0],6),'r')
    plt.ylim([0.6,1.4])
    plt.grid(b=1)
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.ylabel('intensity [arb units]',size=15)

    plt.savefig('../output/res_lc_all.png',clobber=True)
    plt.close()

    return

# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def aps_backpage():

    # -- utilities
    pix0 = 235
    ratn = 0.8

    # -- define the paths and file names
    day_path = '11/28/14.28.45'
    day_file = 'oct08_2013-10-25-175504-292724.raw'
    ngt_path = '11/28/19.23.54'
    ngt_file = 'oct08_2013-10-25-175504-294492.raw'


    # -- read in the images
    img_day = read_raw(day_file,os.path.join(os.environ['DST_DATA'],
                                             day_path))[:,pix0:,:]
    img_ngt = read_raw(ngt_file,os.path.join(os.environ['DST_DATA'],
                                             ngt_path))[:,pix0:,:]

    img_bnd = (ratn*img_ngt.astype(np.float) + 
               (1.0-ratn)*img_day.astype(np.float)).astype(np.uint8)


    # -- make plots
    nrow, ncol = img_ngt.shape[0:2]

    plt.figure(figsize=[26.25,26.25*float(nrow)/float(ncol)])
    plt.imshow(img_day)
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(os.path.join(os.environ['DST_WRITE'],'aps_backpage_day.png'), 
                clobber=True)
    plt.close()

    plt.figure(figsize=[26.25,26.25*float(nrow)/float(ncol)])
    plt.imshow(img_ngt)
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(os.path.join(os.environ['DST_WRITE'],'aps_backpage_ngt.png'), 
                clobber=True)
    plt.close()

    plt.figure(figsize=[26.25,26.25*float(nrow)/float(ncol)])
    plt.imshow(img_bnd)
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(os.path.join(os.environ['DST_WRITE'],'aps_backpage_bnd.png'), 
                clobber=True)
    plt.close()

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def plateau_plot(night, maxstep=True, norm=1, plot_on=True, lw=0.0, 
                 residential=False, band=0, diffsort=False):

    # -- utilities
    if night==1:
        plind = [1,4,6,7,10]
    elif night==9:
        plind = [0,2,6,7,10]
    elif night<0:
        night = -night
        plind = [0,1,2,3,4,5,6,7,8,9,10,11]
    else:
        print("DST_PLOTS: plateaus only chosen for nights 1 and 9!!!")
        return

    dayst = [i for _ in [0,1,2,3] for i in ['Saturday', 'Sunday', 'Monday', 
                                            'Tuesday', 'Wednesday', 
                                            'Thursday', 'Friday']]
    ngtst = str(night).zfill(2)


    # -- read in the light curves
    lcs = LightCurves('','',infile='lcs_night_'+ngtst,noerr=True)
    km  = pkl.load(open('../output/kmeans_night_'+ngtst+'_0.pkl'))


    # -- sort according to cluster labels
    sindex = np.argsort(km.labels_)


    # -- get window labels
    if residential:
        print("DST_PLOTS: reading window labels...")

        fopen = open(os.path.join(os.environ['DST_WRITE'],'window_labels.pkl'),'rb')
        labs  = pkl.load(fopen)
        fopen.close()


    # -- get the on/off transitions and convert to array
    ind_onoff = []
    for i in range(1,10):
        fopen = open('../output/ind_onoff_night_'+ngtst+'_'+str(i)+'.pkl','rb')
        ind_onoff += pkl.load(fopen)
        fopen.close()

    ind_arr = np.array([i for i in ind_onoff])


    # -- get lightcurves from select clusters
    if residential:
        index = np.array([((i in plind) and (j>1000) and (len(k[k<0])>0)) 
                          for (k,j,i) in 
                          zip(ind_arr[sindex],labs.rvec[sindex],km.labels_[sindex])])
    else:
        index = np.array([(i in plind) for i in km.labels_[sindex]])    

    sub   = (lcs.lcs[sindex,:,band])[index]


    # -- pull out the sublist of off transitions
    ind_sub = (ind_arr[sindex])[index]
    all_off = [item for sublist in ind_sub for item in sublist if item<0]
    all_on  = [item for sublist in ind_sub for item in sublist if item>0]


    # -- normalize according to maximum value of the lightcurve
    if norm==1:
        snorm = ((sub.T/sub.max(1)).T)
    elif norm==2:
        snorm = ((sub.T-sub.min(1)).T)
        snorm = ((snorm.T/snorm.max(1)).T)
    else:
        snorm = sub


    # -- identify largest transition and sort
    if maxstep:
        xx_off, yy_off = [], []
        xx_on, yy_on   = [], []

        for i, sublist in enumerate(ind_sub):
            bigoff, left = 0, 0.0
            bigon, right = 0, sub[i].mean()
            for j in sublist:
                if j<0:
                    tleft = sub[i,:-j].mean()
                    if tleft>left:
                        bigoff = j
                        left   = tleft
                elif j>0:
                    tright = sub[i,j:].mean()
                    if tright>right:
                        bigon = j
                        right = tright

            xx_off.append(i)
            yy_off.append(bigoff)
            xx_on.append(i)
            yy_on.append(bigon)

        # sort according to big off
        if diffsort:
            aindex = np.argsort(np.abs(yy_off)-np.abs(yy_on))
        else:
            aindex = np.argsort(np.abs(yy_off))

        yy_off = np.array([i for i in yy_off])[aindex[::-1]]
        yy_on = np.array([i for i in yy_on])[aindex[::-1]]

    else:
        # -- sort according to integral
        aindex = np.argsort(snorm.sum(1))

        # -- get the row/col positions of the off transisitions
        xx_off, yy_off = [], []
        xx_on, yy_on   = [], []
        for i,sublist in enumerate(ind_arr[sindex][index][aindex[::-1]]):
            for j in sublist:
                if j<0:
                    xx_off.append(i)
                    yy_off.append(j)
                elif j>0:
                    xx_on.append(i)
                    yy_on.append(j)


    # -- make the figure
    plt.figure(0, figsize=[15,15])
    plt.subplot(2,2,3)
    n_on, bins, patches = plt.hist(np.abs(all_on),bins=40,facecolor=fillc[1],
                                   edgecolor=linec[1])
    tcks,htimes = time_ticks()
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.grid(b=1)
    plt.ylabel('# of ON transitions', size=15)

    plt.subplot(2,2,4)
    n_off, bins, patches = plt.hist(np.abs(all_off),bins=40,facecolor=fillc[0], 
                                    edgecolor=linec[0])
    tcks,htimes = time_ticks()
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.ylim([0,1.2*max(max(n_on),max(n_off))])
    plt.grid(b=1)
    plt.ylabel('# of OFF transitions', size=15)

    plt.subplot(2,2,3)
    plt.ylim([0,1.2*max(max(n_on),max(n_off))])

    plt.subplot(2,1,1)
    plt.scatter(np.abs(yy_off),xx_off,c=fillc[0],marker='o',lw=lw)
    if plot_on:
        plt.scatter(np.abs(yy_on),xx_on,c=fillc[1],marker='o',lw=lw)
    plt.xticks(tcks,htimes,rotation=30)
    plt.yticks([0],'')
    plt.xlim([0,3600])
    plt.imshow(snorm[aindex[::-1]], aspect=0.5*float(snorm.shape[1])/
                     float(snorm.shape[0]))
    plt.text(0,-20,dayst[night], size=20)

#    plt.savefig(os.path.join(os.environ['DST_WRITE'],
#                             'plateau_onoff_'+ngtst+'.png'), 
#                clobber=True)
#    plt.close()

    return


# -------- # -------- # -------- # -------- # -------- # -------- # -------- 


def ordered_unclustered_plot(night, index=[], lc_ind=[], maxstep=True, norm=1, 
                             plot_on=True, lw=0.0, residential=False, band=0, 
                             write=False):

    # -- utilities
    ngtst = str(night).zfill(2)

    dayst = [i for _ in [0,1,2,3] for i in ['Saturday', 'Sunday', 'Monday', 
                                            'Tuesday', 'Wednesday', 
                                            'Thursday', 'Friday']]

    # -- read in the light curves
    lcs = LightCurves('','',infile='lcs_night_'+ngtst,noerr=True)


    # -- get window labels
    if residential:
        print("DST_PLOTS: reading window labels...")

        fopen = open(os.path.join(os.environ['DST_WRITE'],'window_labels.pkl'),'rb')
        labs  = pkl.load(fopen)
        fopen.close()


    # -- get the on/off transitions and convert to array
    ind_onoff = []
    for i in range(1,10):
        fopen = open('../output/ind_onoff_night_'+ngtst+'_'+str(i)+'.pkl','rb')
        ind_onoff += pkl.load(fopen)
        fopen.close()

    ind_arr = np.array([i for i in ind_onoff])


    # -- get lightcurves from select clusters
    if len(lc_ind)==0:
        if residential:
            for il in np.array([((j>1000) and (len(k[k<0])>0)) for (k,j) in 
                                zip(ind_arr,labs.rvec)]):
                lc_ind.append(il)

        else:
            for il in np.array([len(k[k<0])>0 for k in ind_arr]):
                lc_ind.append(il)
    else:
        print("DST_PLOTS: using input light curve indices...")

    lc_vind = np.array([i for i in lc_ind])
    sub     = lcs.lcs[lc_vind,:,band]


    # -- pull out the sublist of off transitions
    ind_sub = ind_arr[lc_vind]
    all_off = [item for sublist in ind_sub for item in sublist if item<0]
    all_on  = [item for sublist in ind_sub for item in sublist if item>0]


    # -- normalize according to maximum value of the lightcurve
    if norm==1:
        snorm = ((sub.T-sub.min(1)).T)
        snorm = ((snorm.T/snorm.max(1)).T)
    elif norm==2:
        snorm = ((sub.T/sub.max(1)).T)
    else:
        snorm = sub


    # -- identify largest transition and sort
    xx_off, yy_off = [], []
    xx_on, yy_on   = [], []

    for i, sublist in enumerate(ind_sub):
        bigoff, left = 0, 0.0
        bigon, right = 0, sub[i].mean()
        for j in sublist:
            if j<0:
                tleft = sub[i,:-j].mean()
                if tleft>left:
                    bigoff = j
                    left   = tleft
            elif j>0:
                tright = sub[i,j:].mean()
                if tright>right:
                    bigon = j
                    right = tright

        xx_off.append(i)
        yy_off.append(bigoff)
        xx_on.append(i)
        yy_on.append(bigon)


    # -- sort according to big off
    if len(index)==0:
        for iy in np.argsort(np.abs(yy_off)):
            index.append(iy)
    else:
        print("DST_PLOTS: using input sorting indices...")
    vindex = np.array([i for i in index])
    yy_off = np.array([i for i in yy_off])[vindex[::-1]]
    yy_on  = np.array([i for i in yy_on])[vindex[::-1]]


    # -- make the figure
    plt.figure(figsize=[15,15])
    plt.subplot(2,2,3)
    n_on, bins, patches = plt.hist(np.abs(all_on),bins=40,facecolor=fillc[1],
                                   edgecolor=linec[1])
    tcks,htimes = time_ticks()
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.grid(b=1)
    plt.ylabel('# of ON transitions', size=15)

    plt.subplot(2,2,4)
    n_off, bins, patches = plt.hist(np.abs(all_off),bins=40,facecolor=fillc[0], 
                                    edgecolor=linec[0])
    tcks,htimes = time_ticks()
    plt.xticks(tcks,htimes,rotation=30)
    plt.xlim([0,3600])
    plt.ylim([0,1.2*max(max(n_on),max(n_off))])
    plt.grid(b=1)
    plt.ylabel('# of OFF transitions', size=15)

    plt.subplot(2,2,3)
    plt.ylim([0,1.2*max(max(n_on),max(n_off))])

    plt.subplot(2,1,1)
    plt.scatter(np.abs(yy_off),xx_off,c=fillc[0],marker='o',lw=lw)
    if plot_on:
        plt.scatter(np.abs(yy_on),xx_on,c=fillc[1],marker='o',lw=lw)
    plt.xticks(tcks,htimes,rotation=30)
    plt.yticks([0],'')
    plt.xlim([0,3600])
    plt.imshow(snorm[vindex[::-1]], aspect=0.5*float(snorm.shape[1])/
                     float(snorm.shape[0]))
    plt.text(0,-20,dayst[night], size=20)

    if write:
        plt.savefig(os.path.join(os.environ['DST_WRITE'], 
                                 'unclustered_sorted_'+write+'.png'), clobber=True)
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


    # -- K-means clusters
    for night in range(22):
        for band in range(4):
            kmeans_plot(night,band)


    return
