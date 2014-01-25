import dst13
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import median_filter as mf
from scipy.ndimage.filters import gaussian_filter as gf

#lcs = dst13.LightCurves('','',infile='lcs_night_09',noerr=True).lcs

#index = 2463

sm = 30
drng = 2
lc = mf(lcs[index].mean(1),sm)
diff = roll(lc,-drng)-roll(lc,drng)


# canny
#lcg = gf(lcs[index].mean(1),sm)
lcg = gf(lcs[index,:,band],sm)
plt.figure(3,figsize=[5,10])
clf()
subplot(2,1,1)
plot(lcs[index,:,0])
plot(lcs[index,:,1])
plot(lcs[index,:,2])
subplot(2,1,2)
plot(roll(lcg,-drng)-roll(lcg,drng))
ylim([-1,1])

# pull out potential peaks
difgm = np.ma.array([i for i in roll(lcg,-drng)-roll(lcg,drng)])
difgm.mask = np.abs(difgm)>10
difgm.mask[:sm] = True
difgm.mask[-sm:] = True
figure(5)
clf()
plot(difgm,lw=2)
ylim([1.2*difgm.min(),1.2*difgm.max()])
for _ in range(10):
    sig = difgm.std()
    print "iter,sig, min, max = ",_,sig,difgm.min(),difgm.max()
    difgm.mask[np.abs(difgm) > 2*sig] = True
    plot(difgm,lw=2)

#on = np.where((np.abs(difgm) > 10*sig) & 
#              (difgm>roll(difgm,-1)) & 
#              (difgm>roll(difgm,1)))
#off = np.where((np.abs(difgm) > 10*sig) & 
#              (difgm<roll(difgm,-1)) & 
#              (difgm<roll(difgm,1)))
on = np.where((difgm > difgm.mean()+10*sig) & 
              (difgm>roll(difgm,-1)) & 
              (difgm>roll(difgm,1)))
off = np.where((difgm < difgm.mean()-10*sig) & 
              (difgm<roll(difgm,-1)) & 
              (difgm<roll(difgm,1)))
difgm.mask[on] = False
difgm.mask[off] = False
plot(on[0],difgm[on],'go',markersize=5)
plot(off[0],difgm[off],'ro',markersize=5)

#w = np.where(np.abs(difgm) > 10*sig)
#difgm.mask[w] = False
#plot(w[0],difgm[w],'k+',markersize=10)


