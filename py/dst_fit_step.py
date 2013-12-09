import pickle as pkl
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import *

# -- things to pass
npix  = 6*5
width = 0


# -- utilities                                                              
linec    = ['#990000','#006600', '#0000FF']
fillc    = ['#FF6600','#99C299', '#0099FF']



# -- open the test file
#infile = '../output/light_curve_ex.pkl'
#lc     = gaussian_filter(pkl.load(open(infile,'rb')),width)

lc = np.ma.array(lcs.lcs[ilc])
lc.mask = (lc < 1.0)
nband=3

# -- utilities
step_l = np.zeros(npix)
step_r = np.zeros(npix)
offset = np.ones(npix)
tvals  = np.arange(float(npix))
dvals  = np.ma.zeros(npix)


# -- set the step function
step_l[:npix/2] = 1.0
step_r[npix/2:] = 1.0


# -- smooth 
if width>0:
    lc     = gaussian_filter(lc,width)
    tvals  = gaussian_filter(tvals,width)
    step_l = gaussian_filter(step_l,width)
    step_r = gaussian_filter(step_r,width)


# -- estimate the noise
pixfac = 0.532/np.sqrt(float(width)) if width > 0 else 1.0 # correlated noise
noise = (np.roll(lc,1)-lc)[1:-1].std()/np.sqrt(2.0)/pixfac


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


# -- get a slice of the light curve and calculate chisq
for iband in (0,1,2):
    for ii in range(imax):

        dvals[:] = lc[ii:ii+npix,iband]

        if True in dvals.mask:
            continue

        mod1 = np.dot(np.dot(np.dot(tmpl_1,dvals),ptpinv_1),tmpl_1)
        mod2 = np.dot(np.dot(np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2)

        chisq_1[iband,ii] = ((
            dvals - 
            np.dot(np.dot(np.dot(tmpl_1,dvals),ptpinv_1),tmpl_1))**2
        ).sum()/(noise**2)/(float(npix)-3)

        chisq_2[iband,ii] = ((
            dvals - 
            np.dot(np.dot(np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2))**2
        ).sum()/(noise**2)/(float(npix)-2)


dif = chisq_2 - chisq_1
avg = dif.mean(1)
sig = dif.std(1)
thresh = (avg+10*sig)#.clip(1.0,1e6)

w = where(
    (dif[0] > thresh[0]) &
    (dif[1] > thresh[1]) &
    (dif[2] > thresh[2])
)[0]
mx = lc.max()
mn = lc.min()

htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]

figure(1, figsize=[10.0,10.])
clf()

#off = [-30,0,30]
off = np.array([30,70,110]) - lc.mean(0)

subplot(221)
plot(lc[:,0]+off[0],linec[0])
plot(lc[:,1]+off[1],fillc[1])
plot(lc[:,2]+off[2],fillc[2])
xlim([0,3600])
#ymax = 1.2*mx + off[2]
#ymin = 0.8*mn + off[0]
#ymax = 0
#ymin = -30
#ylim([ymin,ymax])
xticks([360.*j for j in range(10+1)], htimes, rotation=30.)
ylabel('intensity [arb. units]')
figtext(0.3,0.86,'window #'+str(ilc),fontsize=15,backgroundcolor='w')
figtext(0.3,0.86,'window #'+str(ilc),fontsize=15)

subplot(222)
fill_between(np.arange(dif.shape[1])+npix/2,dif[0],facecolor=linec[0],
             edgecolor=linec[0])
xlim([0,3600])
#ylim([-1.2*np.abs(dif.min()),np.max([2*thresh[0],1.2*dif[0].max()])])
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
#ylim([-1.2*np.abs(dif.min()),np.max([2*thresh[1],1.2*dif[1].max()])])
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
#ylim([-1.2*np.abs(dif.min()),np.max([2*thresh[2],1.2*dif[2].max()])])
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

print("DST_FIT_STEP: writing window # {0} to png".format(ilc))
savefig('../output/fitstep/fitstep_night_'+str(night).zfill(2)+'_'+
        str(ilc).zfill(4)+'.png', clobber=True)
