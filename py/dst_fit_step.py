import pickle as pkl
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# -- things to pass
npix  = 6*15
width = 0


# -- open the test file
#infile = '../output/light_curve_ex.pkl'
#lc     = gaussian_filter(pkl.load(open(infile,'rb')),width)

lc = np.ma.array(lcs.lcs[ilc,:,0])
lc.mask = (lc < 1.0)


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
ioff    = lc.size % npix
imax    = lc.size-ioff-npix
chisq_1 = np.zeros(imax)
chisq_2 = np.zeros(imax)


# -- get a slice of the light curve and calculate chisq
for ii in range(imax):

    dvals[:] = lc[ii:ii+npix]

    if True in dvals.mask:
        continue

    mod1 = np.dot(np.dot(np.dot(tmpl_1,dvals),ptpinv_1),tmpl_1)
    mod2 = np.dot(np.dot(np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2)

    chisq_1[ii] = ((
            dvals - 
            np.dot(np.dot(np.dot(tmpl_1,dvals),ptpinv_1),tmpl_1))**2
                   ).sum()/(noise**2)/(float(npix)-3)

    chisq_2[ii] = ((
            dvals - 
            np.dot(np.dot(np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2))**2
                   ).sum()/(noise**2)/(float(npix)-2)


dif = chisq_2 - chisq_1
big = where(dif > 0.8)[0]

figure(1, figsize=[5.0,10.])
md = np.median(lc)
mx = lc.max()
clf()
subplot(211)
#fill_between(np.arange(lc.size),lc,md,
#             facecolor='#FF6600',alpha=0.5)
plot(lc)
xlim([0,3600])
ylim([2*md-1.2*mx,1.2*mx])

subplot(212)
fill_between(np.arange(dif.size)+npix/2,gaussian_filter(dif,5),facecolor='#FF6600',alpha=0.5)
plot(np.arange(dif.size)+npix/2,gaussian_filter(dif,5))
xlim([0,3600])
plot([0,3600],[1.0,1.0],'g--')
#ylim([-0.1,max(1.0,dif.max()*1.2)])
ylim([-0.1,2.0])

figtext(0.15,0.93,'window ID:'+str(ilc),fontsize=15)

draw()

#if big.size>0:
#    plot(big,dif[big],'k+',ms=20)
