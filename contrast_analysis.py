import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.io import imread
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.feature import peak_local_max
from skimage.color import rgb2gray, rgba2rgb, rgb2hsv, hsv2rgb
from skimage.filters import threshold_otsu, gaussian, threshold_multiotsu
from skimage.exposure import equalize_adapthist

temp = 25 
img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_140.png'))
img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_140.png'))
#img_df_gray = rgb2gray(rgba2rgb(img_df)[:,:,0])
img_bf_proc = rgb2hsv(img_bf)
img_df_proc = rgb2hsv(img_df)
#img_df_proc[:, :, [1, 2]] = 0

img_bf_gray = rgb2gray(img_bf_proc)
#img_df_gray = rgb2gray(img_df_proc)
img_df_gray = np.logical_and(img_df_proc[:, :, 0] > 0.2, img_df_proc[:, :, 1] > 0.2)
#img_df_gray = img_df_proc
#df_mul = 1 / np.max(rgba2rgb(img_df), axis=2)
#img_df_proc = rgb2gray(rgba2rgb(img_df) * df_mul[:, :, np.newaxis])
#img_df_proc = rgba2rgb(img_df) * df_mul[:, :, np.newaxis]

#img_df_proc = threshold_local(img_df_gray, block_size=3, offset=10)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
#distance = ndi.distance_transform_edt(image)
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#                            labels=image)
#markers = ndi.label(local_maxi)[0]
#labels = watershed(-distance, markers, mask=image)

channel0 = img_df_proc[:, :, 0]
thresh0 = threshold_otsu(channel0)
channel0 = channel0 > thresh0

#channel1 = img_df_proc[:, :, 1]
channel1 = rgb2gray(img_df)
thresh1 = threshold_otsu(channel1)
thresholds = threshold_multiotsu(channel1, classes=2, nbins=1024)
#channel1 = channel1 > thresh1
channel1 = np.digitize(channel1, bins=thresholds)

#channel2 = img_df_proc[:, :, 2]
channel2 = rgb2gray(img_df)
channel2 = equalize_adapthist(channel2, clip_limit=0.03)

fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(rgb2gray(img_df), cmap=plt.cm.gray)
ax[0].set_title('Original Dark Field')
ax[1].imshow(channel0, cmap=plt.cm.gray)
ax[1].set_title('Dark Field Channel 1')
ax[2].imshow(channel1, cmap='Accent')
#ax[2].imshow(channel1, cmap=plt.cm.gray)
ax[2].set_title('Dark Field Channel 2')
ax[3].imshow(channel2, cmap=plt.cm.gray)
ax[3].set_title('Dark Field Channel 3')
#ax[3].imshow(mark_boundaries(img_df_gray, img_df_proc))
ax[4].imshow(img_bf)
ax[4].set_title('Original Bright Field')

print(np.unique(channel1))

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
