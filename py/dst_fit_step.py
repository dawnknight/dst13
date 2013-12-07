import pickle as pkl
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# -- things to pass
npix  = 100
width = 1


# -- open the test file
infile = '../output/light_curve_ex.pkl'
lc     = gaussian_filter(pkl.load(open(infile,'rb')),width)



# -- utilities
step_l = np.zeros(npix)
step_r = np.zeros(npix)
offset = np.ones(npix)
tvals  = np.arange(float(npix))
dvals  = np.zeros(npix)



# -- set the step function
step_l[:npix/2] = 1.0
step_r[npix/2:] = 1.0



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

    mod1 = np.dot(np.dot(np.dot(tmpl_1,dvals),ptpinv_1),tmpl_1)
    mod2 = np.dot(np.dot(np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2)

    chisq_1[ii] = ((
            dvals - 
            np.dot(np.dot(np.dot(tmpl_1,dvals),ptpinv_1),tmpl_1))**2
                   ).sum()

    chisq_2[ii] = ((
            dvals - 
            np.dot(np.dot(np.dot(tmpl_2,dvals),ptpinv_2),tmpl_2))**2
                   ).sum()
