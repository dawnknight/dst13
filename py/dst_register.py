import os,multiprocessing
import pickle as pkl
import numpy as np
from scipy.signal import fftconvolve
from .dst_io import *

# -------- 
#  Register a raw image
#
#  2013/11/23 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def register(inpath=None, infile=None, outpath=None, outfile=None, start=None, 
             end=None, multi=False, cc_mat=None, cc_dic=None):

    """ Register a single image or all images between some start and
    end time.  Pixels shifts for registration are written to a pickled
    dictionary in which the keys are the filenames and the values are
    a list or [row,col] pixel shifts."""


    # -- utilities
    bord  = 20
    nside = 801 # npix/side of a postage stamp
    reg   = (900, 1600, 1701, 2401) # (ul row, ul col, ll row, ll col)


    # -- set the reference frame (registering off of the green image)
    rpath = os.path.join(os.environ['DST_DATA'],'11/15/19.03.48')
    rfile = 'oct08_2013-10-25-175504-182135.raw'
    ref   = 1.0*read_raw(rfile,rpath)[reg[0]:reg[2],reg[1]:reg[3],1]
    ref  -= ref.mean()
    ref  /= ref.std()


    # -- get the time slice
    if start!=None:
        print("DST_REGISTER: registering all images from " + 
              "{0} to {1}".format(start,end))

        fopen = open(os.path.join(os.environ['DST_WRITE'],'filelist.pkl'))
        fl    = pkl.load(fopen)
        fopen.close()

        paths, files, times = fl.time_slice(start,end)
    else:
        if type(inpath) is not list:
            if type(inpath) is np.ndarray:
                paths = list(inpath)
            else:
                paths = [inpath]
        else:
            paths = inpath

        if type(infile) is not list:
            if type(infile) is np.ndarray:
                files = list(infile)
            else:
                files = [infile]
        else:
            files = infile


    # -- initialize the correlation dictionary and matrix
    if cc_dic==None:
        cc_dic = {}
    if cc_mat==None:
        cc_mat = np.zeros([len(files),2*bord+1,2*bord+1])


    # -- loop through the files and calculate the (sub-)correlation matrix
    nproc  = 1 if not multi else multi
    nfiles = len(files)
    dind   = 1 if nproc==1 \
        else nfiles//nproc if nfiles%nproc==0 \
        else nfiles//(nproc-1)

    def reg_subset(conn,paths,files,verbose=False):

        # -- initialize the postage stamp, correlation, & sub-matrix dictionary
        stm        = np.zeros([reg[2]-reg[0],reg[3]-reg[1]])
        conv_mat   = np.zeros([ref.shape[0],ref.shape[1]])
        cc_sub_mat = np.zeros([len(files),2*bord+1,2*bord+1])
        cc_sub_dic = {}

        # -- loop through files
        for i,(p,f) in enumerate(zip(paths,files)):
            if verbose:
                print("DST_REGISTER: " + 
                      "calculating correlation coefficient for file " + 
                      "{0} of {1}".format(i+1,len(files)))

            # -- shift and find correlation
            stm[:,:] = 1.0*read_raw(f,p)[reg[0]:reg[2],reg[1]:reg[3],1]
            stm     -= stm.mean()
            stm     /= stm.std()

            conv_mat      = fftconvolve(ref, stm[::-1,::-1], 'same')
            cc_sub_mat[i] = conv_mat[nside//2-20:nside//2+21,
                                     nside//2-20:nside//2+21]

            # -- find the maximum correlation and add to the dictionary
            mind = conv_mat.argmax()
            off  = [mind / nside - nside//2, mind % nside - nside//2]

            if max(np.abs(off))>(bord-1):
                print("DST_REGISTER: REGISTRATION FAILED FOR FILE " + 
                      "{0}!!!".format(os.path.join(p,f)))

                off = [314,314]

            cc_sub_dic[f] = off

        # -- send sub-matrix back to parent
        if multi:
            conn.send([cc_sub_mat,cc_sub_dic])
            conn.close()
        else:
            return cc_sub_mat, cc_sub_dic


    # -- calculate the correlation matrix
    if multi:
        print("DST_REGISTER: running {0} processes...".format(nproc))

        # -- initialize the full correlation matrix and processes
        parents, childs, ps = [], [], []

        # -- initialize the pipes and processes, then start
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

        # -- collect the results, put into cc_mat, and rejoin
        for ip in range(nproc):
            cc_sub_mat, cc_sub_dic = parents[ip].recv()
            cc_mat[dind*ip:dind*(ip+1),:,:] = cc_sub_mat
            cc_dic.update(cc_sub_dic)
            ps[ip].join()
            print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
        cc_mat[:,:,:], cc_sub_dic = reg_subset(-314,paths,files,verbose=True)
        cc_dic.update(cc_sub_dic)


    return cc_dic



# -------- 
#  Register night time images
#
#  2013/11/29 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def register_night(multi=False):

    """ Register night time images. """

    # -- alerts
    print("DST_REGISTER: registering night time images...")


    # -- set the parameters
    days  = ["10/26", "10/27", "10/28", "10/29", "10/30", "10/31", "11/01",
             "11/02", "11/03", "11/04", "11/05", "11/06", "11/07", "11/08",
             "11/09", "11/10", "11/11", "11/12", "11/13", "11/14", "11/15", 
             "11/16", "11/17"]
    hours = ["19:00:00", "05:00:00"]


    # -- initialize dictionary and register
    dname = os.path.join(os.environ['DST_WRITE'],'registration_dictionary.pkl')
    
    try:
        fopen  = open(dname,'rb')
        cc_dic = pkl.load(fopen)
        fopen.close()
    except:
        cc_dic = {}

    for i in range(len(days)):
        start = days[i]   + "/13 " + hours[0]
        end   = days[i+1] + "/13 " + hours[1]
        dum = register(start=start, end=end, cc_dic=cc_dic, multi=multi)

        # -- write dictionary to file
        print("DST_REGISTER: writing registration dictionary to file")
        print("DST_REGISTER:   {0}".format(dname))
        fopen = open(dname,'wb')
        pickle.dump(cc_dic)
        fopen.close()


    # -- write dictionary to file
    print("DST_REGISTER: writing registration dictionary to file")
    print("DST_REGISTER:   {0}".format(dname))
    fopen = open(dname,'wb')
    pickle.dump(cc_dic)
    fopen.close()

    return
