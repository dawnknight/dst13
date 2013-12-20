import dst13, os
import numpy as np
import pickle as pkl

# -- define the empire state building pixels
ul = [397+20,403+20]
lr = [571+20,595+20]


# -- get the file list
fopen = open('../output/filelist.pkl','rb')
fl    = pkl.load(fopen)
fopen.close()


# -- pull out the files for the first night
paths, files, times = fl.time_slice('11/13/13 19:00:00', '11/14/13 05:00:00')


# -- set up utilities
nrow   = lr[0]-ul[0]
ncol   = lr[1]-ul[1]
nband  = 3
ntimes = len(times)
lc_esb = np.zeros([nband,ntimes])
bord   = 20
img    = np.zeros([nrow,ncol,3])

# -- get the registration dictionary
fopen  = open(os.path.join('../output','registration_dictionary.pkl'),'rb')
cc_dic = pkl.load(fopen)
fopen.close()
   

# -- get the file
for itime, (p,f,t) in enumerate(zip(cls.paths,cls.files,cls.times)):

    print("working on image {0} of {1}...".format(itime+1,ntimes))

    # -- apply the registration
    dr, dc = cc_dic[f]

    if max(abs(dr),abs(dc))<20:
        img[:,:,:] = np.roll(
            np.roll(
                dst13.read_raw(f,p),dr,0
                ),dc,1
            )[ul[0]:lr[0],ul[1]:lr[1],:]
    else:
        img[:,:,:] = dst13.read_raw(f,p)[ul[0]:lr[0],ul[1]:lr[1],:]


    # -- get the light curve
    lc_esb[:,0] = img.transpose(2,0,1).reshape(nband,nrow*ncol).mean(1)


# -- write to file
fopen = open('../output/esb_lc_night_18.pkl')
pkl.dump(lc_esb,fopen)
fopen.close()
