import os,pickle,multiprocessing
import numpy as np
from scipy.signal import fftconvolve, correlate2d
from .dst_io import *
import pdb

# -------- 
#  Register a raw image
#
#  2013/11/23 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def register(inpath='', infile='', outpath='', outfile='', start='', 
                 end='', multi=False):
    """ Register a single image or all images between some start and
    end time.  Pixels shifts for registration are written to a pickled
    dictionary in which the keys are the filenames and the values are
    a list or [row,col] pixel shifts."""


    # utilities
    bord  = 20
    bvec  = np.arange(-bord,bord+1)
    blen  = bvec.size
    nside = 800 # npix/side of a postage stamp
    reg   = (900, 1600, 1700, 2400) # (ul row, ul col, ll row, ll col)
    regsm = (reg[0]+bord,reg[1]+bord,reg[2]-bord,reg[3]-bord)


    # set the reference frame (registering off of the green image)
    rpath = os.path.join(os.environ['DST_DATA'],'11/15/19.03.48')
    rfile = 'oct08_2013-10-25-175504-182135.raw'
    ref   = 1.0*read_raw(rfile,rpath)[regsm[0]:regsm[2],regsm[1]:regsm[3],1]
    ref  -= ref.mean()
    ref  /= ref.std()


    # get the nighttime slice
    paths = [rpath]
    files = ['oct08_2013-10-25-175504-182136.raw']


    # initialize the correlation matrix and dictionary
    cc_mat = np.zeros([len(files),ref.shape[0],ref.shape[1]])
    cc_dic = {}


    # loop through the files and calculate the (sub-)correlation matrix
    nproc  = 1 if not multi else multi
    nfiles = len(files)
    dind   = 1 if nproc==1 \
        else nfiles//nproc if nfiles%nproc==0 \
        else nfiles//(nproc-1)

    def reg_subset(conn,paths,files,verbose=False):

        # initialize the correlation sub-matrix dictionary
        cc_sub_mat = np.zeros([len(files),ref.shape[0],ref.shape[1]])
        cc_sub_dic = {}

        # loop through files
        for i,(p,f) in enumerate(zip(paths,files)):
            if verbose:
                print("DST_REGISTER: " + 
                      "calculating correlation coefficient for file " + 
                      "{0} of {1}".format(i+1,len(files)))

            img = 1.0*read_raw(f,p)[reg[0]:reg[2],reg[1]:reg[3],1]

            # shift and find correlation
            print("DST_REGISTER: convolving")

            stm  = img[bord:-bord,bord:-bord]
            stm -= stm.mean()
            stm /= stm.std()

            cc_sub_mat[i] = fftconvolve(ref, stm[::-1,::-1], 'same')

            # find the maximum correlation and add to the dictionary
            mind          = cc_sub_mat[i].argmax()
            cc_sub_dic[f] = mind # [mind / blen - bord, mind % blen - bord]

        # send sub-matrix back to parent
        if multi:
            conn.send([cc_sub_mat,cc_sub_dic])
            conn.close()
        else:
            return cc_sub_mat, cc_sub_dic


    # calculate the correlation matrix
    if multi:
        print("DST_REGISTER: running {0} processes...".format(nproc))

        # initialize the full correlation matrix and processes
        cc_mat = np.zeros([len(files),2*bord+1,2*bord+1])
        parents, childs, ps = [], [], []

        # initialize the pipes and processes, then start
        for ip in range(nproc):
            ptemp, ctemp = multiprocessing.Pipe()
            parents.append(ptemp)
            childs.append(ctemp)
            ps.append(multiprocessing.Process(target=reg_subset,
                                           args=(childs[ip],
                                                 paths[dind*ip:dind*(ip+1)], 
                                                 files[dind*ip:dind*(ip+1)]), 
                                              kwargs={'verbose':ip==0}))
            ps[ip].start()

        # collect the results, put into cc_mat, and rejoin
        for ip in range(nproc):
            cc_sub_mat, cc_sub_dic = parents[ip].recv()
            cc_mat[dind*ip:dind*(ip+1),:,:] = cc_sub_mat
            cc_dic.update(cc_sub_dic)
            ps[ip].join()
            print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
        cc_mat, cc_dic = reg_subset(-314,paths,files,verbose=True)


    return cc_mat, cc_dic
