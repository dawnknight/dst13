import os
import pickle as pkl
from .dst_light_curves import *
from .dst_canny1d import *

# -------- 
#  Find the on/off transitions for windows and write to file
#
#  2014/01/26 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def onoff(night=None):

    # -- defaults
    nights = [night] if night else range(22)


    # -- run on/off detector write to file
    for night in nights:
        print("DST_ONOFF: running on/off detector for night " + 
              "{0}...".format(night))

        infile    = 'lcs_night_' + str(night).zfill(2)
        lcs       = LightCurves('','',infile=infile,noerr=True)
        ind_onoff = canny1d(lcs)
        outfile   = os.path.join(os.environ['DST_WRITE'], 
                                 'ind_onoff_night_'+str(night).zfill(2)+'.pkl')

        print("DST_ONOFF: writing output to file")
        print("DST_ONOFF:   {0}".format(outfile))

        fopen = open(outfile,'wb')
        pkl.dump(ind_onoff,fopen)
        fopen.close()

    return
