import os
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from .dst_window_labels import *
from .dst_io import *


# -------- 
#  Class holding (night time) light curves for a given start and end time
#
#  2013/11/22 - Writen by Greg Dobler (CUSP/NYU)
# -------- 
class LightCurves():


    # -------- initialize the class
    def __init__(cls, start, end, dpath=os.environ['DST_DATA'], 
                 wpath=os.environ['DST_WRITE'], infile=None, 
                 registered=True, sample=1):
        """ Make night time light curves for start and end times """

        # -- if desired, read in pre-computed light curves
        if infile:
            try:
                cls.read_files(infile)
            except:
                print("DST_LIGHT_CURVES: ERROR - INFILES NOT FOUND!!!")
                return

            return


        # -- utilities
        bord = 20
        nrow = 2160
        ncol = 4096


        # -- get the full file list
        print("DST_LIGHT_CURVES: reading the full file list...")

        fopen = open(os.path.join(wpath,'filelist.pkl'),'rb')
        fl    = pkl.load(fopen)
        fopen.close()


        # -- take a time slice
        cls.start = start
        cls.end   = end
        cls.paths, cls.files, cls.times = fl.time_slice(start, end)


        # -- sample
        cls.samp = sample
        if sample>1:
            cls.paths = cls.paths[::sample]
            cls.files = cls.files[::sample]
            cls.times = cls.times[::sample]


        # -- get the registration dictionary
        cls.reg = registered
        if registered:
            print("DST_LIGHT_CURVES: reading the registration dictionary...")
            try:
                fopen  = open(os.path.join(wpath,
                                           'registration_dictionary.pkl'),'rb')
                cc_dic = pkl.load(fopen)
                fopen.close()
            except:
                cc_dic = {}


        # -- get the window labels and extract parameters
        print("DST_LIGHT_CURVES: reading the window labels...")

        labs   = WindowLabels(hand=True)
        wpix   = labs.wpix
        labvec = labs.labvec
        sind   = labs.sind
        labsrt = labs.labsrt
        lbound = labs.lbound


        # -- initialize the light curves
        cls.ntime = len(cls.times)
        cls.nwin  = len(lbound) - 1
        cls.lcs   = np.zeros([cls.nwin,cls.ntime,3])
        cls.std   = np.zeros([cls.nwin,cls.ntime,3])
        cls.err   = np.zeros([cls.nwin,cls.ntime,3])
        cls.bse   = np.zeros([cls.nwin,cls.ntime,3])


        # -- initialize color vectors
        red = np.zeros(sind.size)
        grn = np.zeros(sind.size)
        blu = np.zeros(sind.size)
        img = np.zeros([nrow-2*bord,ncol-2*bord,3]).astype(np.uint8)


        # -- loop through images
        for itime, (p,f,t) in enumerate(zip(cls.paths,cls.files,cls.times)):
            print("DST_LIGHT_CURVES: extracting window brightnesses for " + 
                  "file {0} of {1}".format(itime+1,len(cls.times)))

            # check for registration; register if not yet registered
            if registered:
                try:
                    dr, dc = cc_dic[f]
                except:
                    print("DST_LIGHT_CURVES: registration missing for file")
                    print("DST_LIGHT_CURVES:   {0}".format(f))
                    dr, dc = dst_register(p,f,cc_dic=cc_dic)[f]

                if max(abs(dr),abs(dc))<20:
                    img[:,:,:] = np.roll(
                        np.roll(
                            read_raw(f,p),dr,0
                        ),dc,1
                    )[bord:-bord,bord:-bord,:]
                else:
                    img[:,:,:] = read_raw(f,p)[bord:-bord,bord:-bord,:]
            else:
                img[:,:,:] = read_raw(f,p)[bord:-bord,bord:-bord,:]

            red[:] = ((img[:,:,0])[wpix])[sind].astype(np.float)
            grn[:] = ((img[:,:,1])[wpix])[sind].astype(np.float)
            blu[:] = ((img[:,:,2])[wpix])[sind].astype(np.float)

            for iwin in range(cls.nwin):
                ilo = lbound[iwin]
                ihi = lbound[iwin+1]
                rtn = np.sqrt(float(ihi-ilo+1))

                cls.lcs[iwin,itime,0] = red[ilo:ihi].mean()
                cls.lcs[iwin,itime,1] = grn[ilo:ihi].mean()
                cls.lcs[iwin,itime,2] = blu[ilo:ihi].mean()

                cls.std[iwin,itime,0] = red[ilo:ihi].std()
                cls.std[iwin,itime,1] = grn[ilo:ihi].std()
                cls.std[iwin,itime,2] = blu[ilo:ihi].std()

                cls.err[iwin,itime,0] = cls.std[iwin,itime,0]/rtn
                cls.err[iwin,itime,1] = cls.std[iwin,itime,1]/rtn
                cls.err[iwin,itime,2] = cls.std[iwin,itime,2]/rtn

        return



    # -------- calculate the L2-norm
    def l2_norm(cls, norm='time'):

        """ Calculate the L2 norm of the light curves in either time
        (default) or space. """

        # -- calculate L2-norm
        ax = 1 if norm=='time' else 0


        # -- return transposed (for speed later)
        return np.sqrt((cls.lcs**2).sum(axis=ax)).T



    # -------- principle component analysis in time
    def pca_time(cls, ncomp, colors='rgb'):

        """ Run a basic principle component analysis in time """

        # -- utilities
        clrs = []
        if 'r' in colors:
            clrs.append(0)
        if 'g' in colors:
            clrs.append(1)
        if 'b' in colors:
            clrs.append(2)


        # -- initialize PCA and do the fit
        pcas = []
        for ic,c in zip(clrs,colors):
            print("DST_LIGHT_CURVES: Running PCA with " + 
                  "{0} components in {1}-band...".format(ncomp,c))

            pca_ = PCA(n_components=ncomp)
            pca_.fit(cls.lcs[:,:,ic])

            pcas.append(pca_)


        # -- extract the components and return as list
        return pcas



    # -------- principle component analysis in time
    def pca_space(cls, ncomp, colors='rgb'):

        """ Run a basic principle component analysis in space """

        # -- utilities
        clrs = []
        if 'r' in colors:
            clrs.append(0)
        if 'g' in colors:
            clrs.append(1)
        if 'b' in colors:
            clrs.append(2)


        # -- initialize PCA and do the fit
        pcas = []
        for ic,c in zip(clrs,colors):
            print("DST_LIGHT_CURVES: Running PCA with " + 
                  "{0} components in {1}-band...".format(ncomp,c))

            pca_ = PCA(n_components=ncomp)
            pca_.fit(cls.lcs[:,:,ic].T)

            pcas.append(pca_)


        # -- extract the components and return as list
        return pcas



    # -------- write out class contents to files
    def write_files(cls, outbase):

        """ Write out the contents of this class to a file """

        # -- utilities
        outpath = os.environ['DST_WRITE']
        lcsfile = outbase + '_lcs.pkl'
        stdfile = outbase + '_std.pkl'
        errfile = outbase + '_err.pkl'
        bsefile = outbase + '_bse.pkl'
        parfile = outbase + '_par.pkl'


        # -- write the light curves
        outfile = os.path.join(outpath,lcsfile)

        print("DST_LIGHT_CURVES: writing light curves to")
        print("DST_LIGHT_CURVES:   {0}".format(outfile))

        fopen = open(outfile,'wb')
        pkl.dump(cls.lcs,fopen)
        fopen.close()


        # -- write the standard deviation
        outfile = os.path.join(outpath,stdfile)

        print("DST_LIGHT_CURVES: writing standard deviation to")
        print("DST_LIGHT_CURVES:   {0}".format(outfile))

        fopen = open(outfile,'wb')
        pkl.dump(cls.std,fopen)
        fopen.close()


        # -- write the error on the mean
        outfile = os.path.join(outpath,errfile)

        print("DST_LIGHT_CURVES: writing error to")
        print("DST_LIGHT_CURVES:   {0}".format(outfile))

        fopen = open(outfile,'wb')
        pkl.dump(cls.err,fopen)
        fopen.close()


        # -- write the bootstrap errors (not implemented yet)
#        outfile = os.path.join(outpath,bsefile)
#
#        print("DST_LIGHT_CURVES: writing bootstrap errors to")
#        print("DST_LIGHT_CURVES:   {0}".format(outfile))
#
#        fopen = open(outfile,'wb')
#        pkl.dump(cls.bse,fopen)
#        fopen.close()


        # -- write the parameters
        outfile = os.path.join(outpath,parfile)

        print("DST_LIGHT_CURVES: writing parameters to")
        print("DST_LIGHT_CURVES:   {0}".format(outfile))

        fopen = open(outfile,'wb')
        pkl.dump(cls.start, fopen)
        pkl.dump(cls.end,   fopen)
        pkl.dump(cls.paths, fopen)
        pkl.dump(cls.files, fopen)
        pkl.dump(cls.times, fopen)
        pkl.dump(cls.samp,  fopen)
        pkl.dump(cls.reg,   fopen)
        pkl.dump(cls.ntime, fopen)
        pkl.dump(cls.nwin,  fopen)
        fopen.close()

        return



    # -------- read out class contents from files
    def read_files(cls, inbase):

        """ Write out the contents of this class to a file """

        # -- alert the user
        print("DST_LIGHT_CURVES: reading light curves from")
        print("DST_LIGHT_CURVES:   path = {0}".format(os.environ['DST_WRITE']))
        print("DST_LIGHT_CURVES:   base = {0}".format(inbase))


        # -- utilities
        inpath  = os.environ['DST_WRITE']
        lcsfile = inbase + '_lcs.pkl'
        stdfile = inbase + '_std.pkl'
        errfile = inbase + '_err.pkl'
        bsefile = inbase + '_bse.pkl'
        parfile = inbase + '_par.pkl'


        # -- read the light curves
        infile  = os.path.join(inpath,lcsfile)
        fopen   = open(infile,'rb')
        cls.lcs = pkl.load(fopen)
        fopen.close()


        # -- read the standard deviation
        infile  = os.path.join(inpath,stdfile)
        fopen   = open(infile,'rb')
        cls.std = pkl.load(fopen)
        fopen.close()


        # -- read the error on the mean
        infile  = os.path.join(inpath,errfile)
        fopen   = open(infile,'rb')
        cls.err = pkl.load(fopen)
        fopen.close()


        # -- write the bootstrap errors (not implemented yet)
#        infile  = os.path.join(inpath,bsefile)
#        fopen   = open(infile,'rb')
#        cls.bse = pkl.load(fopen)
#        fopen.close()


        # -- read the parameters
        infile    = os.path.join(inpath,parfile)
        fopen     = open(infile,'rb')
        cls.start = pkl.load(fopen)
        cls.end   = pkl.load(fopen)
        cls.paths = pkl.load(fopen)
        cls.files = pkl.load(fopen)
        cls.times = pkl.load(fopen)
        cls.samp  = pkl.load(fopen)
        cls.reg   = pkl.load(fopen)
        cls.ntime = pkl.load(fopen)
        cls.nwin  = pkl.load(fopen)
        fopen.close()

        return
