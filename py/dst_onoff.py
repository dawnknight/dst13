import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# -------- 
#  Find the on/off transitions for windows
#
#  2013/12/04 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def dst_onoff(data):

    # -- set the data
    if type(data) is str:
        infile = os.path.join(os.environ['DST_WRITE'],data+'_lcs.pkl')

        print("DST_ONOFF: Reading light curves from {0}".format(infile))
        fopen  = open(infile, 'rb')
        lcs    = pkl.load(fopen)
        fopen.close()
    else:
        lcs = data

        plt.figure(3,figsize=[7,10])

        for iwin in range(100,200):

            clf()
            subplot(211)

            lcr = nd.filters.median_filter(lcs[iwin,:,0],20)
            diff = (np.roll(savg(lcr),1) - savg(lcr))

            sig = diff.std()
            avg = diff.mean()

            ind = where(abs(diff) > (avg+10.0*sig))[0]
            big = diff[ind]

            plt.plot(diff,'#990000')
            plt.plot(ind,big,'k+',ms=20)
            plt.grid(b=1)
            ylim([-20,20])

            subplot(212)

            plt.plot(np.arange(lcr.size)/1.,lcr,'#FF6600')
            #plt.plot(lcs[iwin+1,:,0],'#000099')
            #plt.plot(lcs[iwin+2,:,0],'#990000')
            #plt.plot(savg(lcs[iwin-1,:,0]),'#009900')
            #plt.plot(lcs[iwin-2,:,0],'#990099')
            for i in ind:
                plot([i,i],[20,110],'#990000')

            plt.grid(b=1)
            plt.draw()
            plt.show()

            time.sleep(1)
