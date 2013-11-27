import numpy as np

# -------- 
#  Rebin an image
#
#  2013/11/25 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def rebin(img, fac):
    """ Rebins a 2D numpy array by the specified factor """

    # -- test input
    err = "REBIN: Error - input is not a 2D (or 2D x 3) numpy array."
    try:
        dim = img.shape
    except:
        print(err)
        return -1

    if (len(dim)!=2) and (len(dim)!=3):
        print(err)

    if (dim[0]%fac!=0) or (dim[1]%fac!=0):
        print("REBIN: Error: final dimensions must be integer number of " + 
              "initial.")
        return


    # -- get the input type
    intype  = img.dtype


    # -- reshape
    if len(dim)==3:
       return np.dstack([rebin(img[:,:,0], fac), 
                         rebin(img[:,:,1], fac), 
                         rebin(img[:,:,2], fac)])
    else:
        outsh = [i/fac for i in dim]
        sh    = outsh[0],img.shape[0]//outsh[0],outsh[1],img.shape[1]//outsh[1]

    return ((img.astype(np.float)).reshape(sh).mean(-1).mean(1)).astype(intype)
