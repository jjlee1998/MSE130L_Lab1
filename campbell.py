import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from scipy import ndimage as ndi
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray, rgba2rgb, rgb2hsv, hsv2rgb
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.morphology import binary_opening, opening, closing, local_minima
from skimage.filters import threshold_multiotsu
from skimage.exposure import histogram
#from skimage.segmentation import mark_boundaries
#from skimage.feature import peak_local_max
#from skimage.filters import threshold_otsu, gaussian, threshold_multiotsu
#from skimage.exposure import equalize_adapthist

temp = 35 
img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_140.png'))
img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_140.png'))

grad_bf = sobel(rgb2gray(img_bf))

bord_bf = grad_bf > (np.mean(grad_bf) + 0*np.std(grad_bf))
bord_df = rgb2gray(img_df)
img_proc = (1 - grad_bf) * (1 - bord_df)
img_camp = closing(img_proc)
img_camp_overlay = img_camp < np.mean(img_camp)
#img_camp = opening(img_proc)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img_bf)
ax[0].set_title('Original Bright Field')
ax[1].imshow(img_df)
ax[1].set_title('Original Dark Field')
ax[2].imshow(img_proc, cmap=plt.cm.gray)
ax[2].set_title('Campbell Method Intermediate')
ax[3].imshow(img_camp, cmap=plt.cm.nipy_spectral)
#ax[3].imshow(img_camp_overlay, alpha=.5, cmap=plt.cm.gray)
ax[3].set_title('Campbell Method Final')

for a in ax[:3]:
    a.set_axis_off()

fig.tight_layout()
plt.show()





#img_proc = np.logical_or(bord_bf, bord_df)
#img_proc = 1 - np.logical_or(grad_bf > 0.1, bord_df > 0.1)
#img_proc = 1 - grad_bf
#img_proc = (grad_bf + bord_df)/2
#img_camp = opening(1 - closing(img_proc, selem=disk(3)), selem=disk(3))
#img_camp = 1 - img_proc
#img_camp = img_proc > np.mean(img_proc)
#img_camp = img_proc > np.mean(img_proc)
#print(np.mean(img_camp))
#print(np.median(img_camp))
#img_proc = (img_proc - np.min(img_proc)) / np.max(img_proc)
#hsv_df = rgb2hsv(img_df)
#hsv_df[:, :, 0] = np.where(hsv_df[:, :, 0] < 0.5, hsv_df[:, :, 0], 0)
#img_proc = hsv2rgb(hsv_df)
#for _ in range(20):
#    img_camp_overlay = binary_opening(img_camp_overlay)

#lmx, lmy = local_minima(img_camp, indices=True, connectivity=10)
#img_camp = opening(1 - img_proc, selem=disk(3))
#markers = np.zeros_like(img_proc)
#markers[img_proc < 0.1] = 1
#markers[img_proc > 0.2] = 2
#elevation_map = sobel(img_proc)
#segmentation = watershed(img_proc, markers)
#img_proc = segmentation

#hist, hist_centers = histogram(img_proc, nbins=256)

#print(img_proc.shape)

#thresholds = threshold_multiotsu(img_proc, classes=2)
#img_camp = np.digitize(img_proc, bins=thresholds)

#markers = rank.gradient(img_proc, disk(5)) < 10
#markers = ndi.label(markers)[0]
#gradient = rank.gradient(img_proc, disk(2))
#img_camp = watershed(img_proc, markers=minima)
#img_camp = rank.gradient(rgb2gray(img_bf), disk(2))
#img_camp = sobel(img_proc)

#img_camp = np.zeros_like(img_proc)
#img_camp[img_proc < 0.05] = 1
#img_camp[img_proc > 0.95] = 2
#print(img_camp)
#img_camp = sobel(rgb2gray(img_df))
#print(np.mean(img_camp))
#print(np.max(img_camp))
#img_camp = img_camp > (np.mean(img_camp))
#img_camp = binary_opening(img_camp, selem=disk(2))

#img_camp = ndi.generic_filter(img_df, np.var, size=3)


