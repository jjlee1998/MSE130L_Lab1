import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb, label2rgb
from skimage.filters import sobel, gaussian, threshold_otsu, threshold_local,\
        unsharp_mask
from skimage.filters.rank import mean
from skimage.morphology import disk
from skimage.morphology import closing, opening, binary_closing,\
        binary_opening, reconstruction, area_opening, diameter_closing,\
        binary_erosion, erosion, diameter_opening
from skimage.segmentation import clear_border
from skimage.measure import label

# Import images in RGB space:
temp = 35 
img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_140.png'))
img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_140.png'))

# Isolate greyscale maps of the boundaries of the images:
bound_bf = sobel(rgb2gray(img_bf))
bound_df = rgb2gray(img_df)

# Local adaptive thresholding to binarize fields:
block_size = 35
bin_bf = bound_bf > threshold_local(bound_bf, block_size, offset=0)
bin_df = bound_df > threshold_local(bound_df, block_size, offset=0)

# Combine fields two different ways:
result_1 = 1 - (bound_bf+bound_df) / 2
result_2 = 1 - np.logical_or(bin_bf, bin_df)

# Remap to maximize contrast across middle 95% of data:

retain = 95
minval = np.percentile(result_1, (100-retain)/2)
maxval = np.percentile(result_1, (retain+100)/2)
result_1 = (result_1 - minval) / (maxval - minval)
result_1[result_1 < 0] = 0
result_1[result_1 > 1] = 1
result_2 = result_1

# Erode boundaries and binarize:
#result_1 = opening(result_1)
#result_2 = result_1 > threshold_local(result_1, block_size=5, offset=0)
#result_2 = binary_opening(result_2)

# Patch single holes?

# Reconstruct image using adaptive binarized mask:
#mask = result_1
#seed = np.copy(result_1)
#seed[1:-1, 1:-1] = result_1.min()
#result_1 = reconstruction(seed, mask, method='dilation')

# Count objects:
#cleared = clear_border(bw)
result_2 = result_2 > threshold_local(result_2, block_size=11, offset=0)
result_2 = binary_erosion(result_2, selem=disk(2))
result_2_labels = label(result_2, connectivity=1)
result_2 = label2rgb(result_2_labels, image=result_1, bg_label=0)

result_2 = unsharp_mask(result_1, radius=5, amount=1)

result_2_cores = np.zeros_like(result_2)
result_2_areas = np.zeros_like(result_2)

while np.any(result_2_areas != result_2)
for i in range(5):
    
    #result_2_areas = area_opening(result_2, area_threshold=10, connectivity=2)
    result_2_areas = diameter_opening(result_2, diameter_threshold=5, connectivity=2)
    result_2_cores = result_2_cores + (result_2 - result_2_areas)
    result_2 = erosion(result_2_areas, selem=disk(2))
    #result_2 = result_2 + result_2_cores

result_3 = np.zeros((result_2_cores.shape[0], result_2_cores.shape[1], 4))
result_3[:, :, 0] = result_2_cores
result_3[:, :, 3] = result_2_cores > 0

#result_2 = result_2 > threshold_local(result_2, block_size=11, offset=0)
#result_2 = opening(result_2)
#result_2 = opening(result_2)

# Plot results:

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img_bf)
ax[0].set_title('Original Bright Field')
ax[1].imshow(img_df)
ax[1].set_title('Original Dark Field')
ax[2].imshow(result_1, cmap=plt.cm.gray)
ax[2].set_title('Processed Field 1 (Continuum Route)')
ax[3].imshow(result_1, cmap=plt.cm.gray)
ax[3].imshow(result_3)
ax[3].set_title('Processed Field 2 (Binary Route)')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()



#img_proc = (grad_bf + bord_df)/2
#img_fin = opening(sobel(opening(img_proc)))

#x = np.arange(img_proc.shape[0])
#y = np.arange(img_proc.shape[1])
#xx, yy = np.meshgrid(x, y)
#cc = img_proc > np.mean(img_proc)
#data = np.vstack((xx.reshape(-1), yy.reshape(-1), cc.reshape(-1)))
#data = np.transpose(data)

#dbscanner = DBSCAN(eps=1, min_samples=5)
#clustering = dbscanner.fit(data)

#clusters = np.reshape(clustering.labels_, img_proc.shape)
#clusters = clusters + 2
#print(f'{np.max(clusters)} Clusters Located')
#clusters = clusters / np.max(clusters)
