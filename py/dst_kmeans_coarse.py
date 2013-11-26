import os,pickle
import numpy as np
from sklearn.cluster import KMeans
from .dst_imtools import *

# -------- 
# coarse K-Means clustering on whole image to reproduce Brumby's plots
# -------- 

#def kmeans_coarse(start,end):

""" K-Means clustering on a la Brumby 2013-04-10 (see cusp.pro) """

print(" USING DEFAULTS...")
start = '11/15/13 19:00:00'
end = '11/16/13 05:00:00'

# -- utilities
nrow = 2160
ncol = 4096
fac  = 4
nrow_bin = nrow/fac
ncol_bin = ncol/fac

# -- get the file list
fopen = open(os.path.join(os.environ['DST_WRITE'],'filelist.pkl'),'rb')
fl    = pickle.load(fopen)
fopen.close()


# -- pull out a time slice
paths, files, times = fl.time_slice(start,end)


# -- initialize features matrix (r-band only), img container, and ones
lcs = np.zeros([nrow_bin*ncol_bin,files[::25].size])
img = np.zeros([nrow_bin*ncol_bin])
ons = np.ones(img.shape)


# -- rebin (r-band only), divide by L2-norm, and insert
for i, (p,f,t) in enumerate(zip(paths[::25],files[::25],times[::25])):
    print("File number {0} of {1}".format(i+1,lcs.shape[1]))
    img[:]   = rebin(read_raw(f,p).astype(np.float)[:,:,0],fac).flatten()
    img[:]  /= np.sqrt(np.dot(img**2,ons))
    lcs[:,i] = img


# -------- kmeans
kmeans_coar = KMeans(init='random', n_clusters=12, n_init=10)
kmeans_coar.fit(lcs)
