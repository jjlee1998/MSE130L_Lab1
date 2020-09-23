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
from skimage.morphology import binary_opening, opening, closing

from sklearn.cluster import DBSCAN

temp = 15 
img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_140.png'))
img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_140.png'))

grad_bf = sobel(rgb2gray(img_bf))

bord_bf = grad_bf > (np.mean(grad_bf) + 0*np.std(grad_bf))
bord_df = rgb2gray(img_df)

img_proc = (grad_bf + bord_df)/2
img_proc = closing(grad_bf)
#img_camp = opening(1 - closing(img_proc, selem=disk(3)), selem=disk(3))
img_camp = 1 - img_proc
img_fin = opening(sobel(opening(img_camp)))

x = np.arange(img_proc.shape[0])
y = np.arange(img_proc.shape[1])
xx, yy = np.meshgrid(x, y)
cc = img_proc > np.mean(img_proc)
data = np.vstack((xx.reshape(-1), yy.reshape(-1), cc.reshape(-1)))
data = np.transpose(data)

dbscanner = DBSCAN(eps=1, min_samples=5)
#clustering = dbscanner.fit(data)

#clusters = np.reshape(clustering.labels_, img_proc.shape)
#clusters = clusters + 2
#print(f'{np.max(clusters)} Clusters Located')
#clusters = clusters / np.max(clusters)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img_bf)
ax[0].set_title('Original Bright Field')
ax[1].imshow(img_df)
ax[1].set_title('Original Dark Field')
ax[2].imshow(img_proc, cmap=plt.cm.gray)
ax[2].set_title('Processing Intermediate')
#ax[3].imshow(img_camp, cmap=plt.cm.gray)
#ax[3].imshow(clusters, cmap=plt.cm.nipy_spectral)
ax[3].imshow(img_fin, cmap=plt.cm.gray)
ax[3].set_title('DBSCAN Clusters')

for a in ax[:3]:
    a.set_axis_off()

fig.tight_layout()
plt.show()
