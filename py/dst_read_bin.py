import os
import numpy as np

# -------- 
# read a binary file
# -------- 
def read_bin(fname, dtype, shape=None, path=os.environ['DST_WRITE']):
    """ Read an anonymous binary file """

    # -- open the file, read, and close
    fopen = open(os.path.join(path,fname))
    data  = np.fromfile(fopen, dtype=dtype, count=-1)
    fopen.close()

    # -- reshape the data if desired
    if shape!=None:
        return data.reshape(shape)
    else:
        return data
