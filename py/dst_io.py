import sys, os, pickle, gim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -------- 
# Class holding the names and directories of all DST files
# -------- 
class FileList():

    def __init__(cls, path=os.environ['DST_DATA']):
        """ Get the full file list. """

        # initialize file list
        cls._path = path
        cls.paths = []
        cls.files = []
        cls.times = []

        # loop through months, days, grab and get filename and
        # timestamp
        for month in os.listdir(path):
            tpath = os.path.join(path,month)
            for day in os.listdir(tpath):

                print "DST_IO: reading date ", month+'/'+day

                tpath = os.path.join(path,month,day)
                for grab in os.listdir(tpath):
                    tpath = os.path.join(path,month,day,grab)
                    for ifile in os.listdir(tpath):
                        cls.paths.append(tpath)
                        cls.files.append(ifile)
                        cls.times.append(os.path.getmtime(tpath+'/'+ifile))

        # sort according to time
        ind = sorted(range(len(cls.times)), key=cls.times.__getitem__)
        cls.paths = [cls.paths[i] for i in ind]
        cls.files = [cls.files[i] for i in ind]
        cls.times = [cls.times[i] for i in ind]

        cls.paths = np.array(cls.paths)
        cls.files = np.array(cls.files)
        cls.times = np.array(cls.times)


    def time_slice(cls, start, end):
        """ Pull out file names between a certain time .  Input times
        must either be in unix or in the format 'MM/DD/YY HH:MM:SS'. """

        # check the input format
        if type(start) is str:
            start = float(datetime.strptime(start,'%m/%d/%y %H:%M:%S'
                                            ).strftime('%s'))
            end   = float(datetime.strptime(end, '%m/%d/%y %H:%M:%S'
                                            ).strftime('%s'))


        # check boundaries
        if (start<cls.times[0]) or (end>cls.times[-1]):
            print("DST_IO: ERROR - out of time bounds.")
            return


        # pull out the appropriate times
        w = np.where((cls.times>=start) & (cls.times<end))

        return cls.paths[w], cls.files[w], cls.times[w]


    def night_files(cls):
        """ Select all files that are between 7pm and 5am. """

        # initialize the full list
        paths, files, times = [], [], []

        #set the parameters
        days  = ["10/26", "10/27", "10/28", "10/29", "10/30", "10/31", "11/01",
                 "11/02", "11/03", "11/04", "11/05", "11/06", "11/07", "11/08",
                 "11/09", "11/10", "11/11", "11/12", "11/13", "11/14", "11/15"]
        hours = ["19:00:00", "05:00:00"]

        # get the file names
        for i in range(len(days)-1):

            start = str("{0}/13 {1}".format(days[i],hours[0]))
            end   = str("{0}/13 {1}".format(days[i+1],hours[1]))
            p,f,t = cls.time_slice(start,end)
            paths = np.append(paths,p)
            files = np.append(files,f)
            times = np.append(times,t)

        return paths, files, times


    def write_list(cls, path=os.environ['DST_WRITE'], outfile='filelist.pkl'):
        """ Pickle the paths files and times. """

        # open the file
        print("DST_IO: Writing pickled paths, file names, and times to ")
        print("         {0}".format(os.path.join(path,outfile)))

        fopen = open(os.path.join(path,outfile),'wb')

        # pickle and close
        pickle.dump(cls,fopen)
        fopen.close()

        return


    def plot_live(cls, fname='dst13_nobs.png', path=os.environ['DST_WRITE']):
        """ Plot the "live" time of the DST observations """

        # set the time for initializing
        start = float(datetime.strptime("10/26/13 00:00:00",'%m/%d/%y %H:%M:%S'
                                        ).strftime('%s'))

        # bin into a histogram
        hour_s = 60.*60.
        day_s  = 24.*hour_s
        bins   = np.arange(start,start+day_s*21.+hour_s,hour_s)

        cov, hbins, pat = plt.hist(cls.times,bins=bins)

        # reshape and view
        grid = cov.reshape(21,24).T

        plt.figure(figsize=[7.5,6.0])
        plt.imshow(grid,'gist_heat',interpolation='nearest')
        plt.ylabel('Hour', size=20)
        plt.xlabel('Day', size=20)
        plt.title('DST13 observational coverage', size=20)
        cbar = plt.colorbar(pad=0.05)
        plt.text(26,12,'# of observations', rotation=270., va='center', 
                 size=15)
        plt.savefig(os.path.join(path,fname))
        plt.close()

        return


    def view_slice(cls, start, end, outdir, fac=None, sample=1):
        """ Write the images of a time slice to a list of files. """

        # set the default output path
        DST_WRITE = os.environ['DST_WRITE']

        # get the paths and files
        paths, files, times = cls.time_slice(start,end)

        # handle the case of no files
        nfiles = len(files)
        if nfiles==0:
            print("DST_IO: no raw images found.")
            return
        nfiles = len(files[::sample])

        # create the directory if it doesn't exist
        totpath = os.path.join(DST_WRITE,outdir)
        if not os.path.isdir(totpath):
            try:
                print("DST_IO: creating {0}".format(totpath))
                os.makedirs(totpath)
            except OSError:
                raise

        # read in the image and write to a file
        for i,(p,f,t) in enumerate(zip(paths[::sample],files[::sample], 
                                       times[::sample])):
            print("DST_IO: writing png file {0} of {1}".format(i+1,nfiles))

            fout = 'img_'+f[-10:-4]+'.png'

            if fac==None:
                img = read_raw(f,p)
            else:
                try:
                    img = gim.rebin(read_raw(f,p),fac)
                except:
                    print("DST_IO: gim.rebin is only compatible " + 
                          "with 2D arrays")
                    img = read_raw(f,p)

            write_png(img, fout, path=totpath)

        return



# -------- 
# Read in the raw file format to numpy array
# -------- 
def read_raw(fname, path=os.environ['DST_WRITE'], nrow=2160, ncol=4096):

    """ Read in the raw file format to a numpy array """

    # set the file name, read, and return
    infile = os.path.join(path,fname)

    return np.fromfile(open(infile,'rb'), dtype=np.uint8, count=-1,
                       ).reshape(nrow,ncol,3)[:,:,::-1]



# -------- 
# Write the png
# -------- 
def write_png(img, fname, path=os.environ['DST_WRITE'], time_='', text_='', 
              figsize=[7.50,3.96], cmap=None, clim=None):

    """ Save an image as png """

    # make the plot
    plt.figure(figsize=figsize)
    plt.imshow(img, interpolation='nearest',cmap=cmap,clim=clim)
    plt.subplots_adjust(0,0,1,1)
    plt.axis('off')

    if time_!='':
        plt.figtext(0.01,0.95,'Time = '+str(time_),ha='left', color='w')
    elif text_!='':
        plt.figtext(0.01,0.95,text_,ha='left', color='w')

    plt.savefig(os.path.join(path,fname),clobber=True)
    plt.close()

    return



