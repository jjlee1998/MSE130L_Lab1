import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb, label2rgb
from skimage.filters import sobel, gaussian, threshold_otsu, threshold_local,\
        unsharp_mask, threshold_multiotsu
from skimage.filters.rank import mean
from skimage.morphology import disk
from skimage.morphology import closing, opening, binary_closing,\
        binary_opening, reconstruction, area_opening, diameter_closing,\
        binary_erosion, erosion, diameter_opening, flood, remove_small_holes,\
        binary_closing, remove_small_objects, binary_dilation, dilation
from skimage.segmentation import watershed, random_walker, mark_boundaries
from skimage.measure import label, regionprops
from skimage.future import graph
from skimage.util import random_noise, img_as_ubyte

# Import images in RGB space:
temp = 15 
img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_140.png'))
img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_140.png'))

# Isolate greyscale maps of grain boundaries:
gbs_bf = sobel(rgb2gray(img_bf))
gbs_df = rgb2gray(img_df)

# Rescale fields to maximize contrast across middle 95% of data:

minval_bf = np.percentile(gbs_bf, 2.5)
minval_df = np.percentile(gbs_df, 2.5)
maxval_bf = gbs_bf.max()
maxval_df = gbs_df.max()

gbs_bf_maxcon = (gbs_bf - minval_bf) / (maxval_bf - minval_bf)
gbs_df_maxcon = (gbs_df - minval_df) / (maxval_df - minval_df)

gbs_bf_maxcon[gbs_bf_maxcon < 0] = 0
gbs_df_maxcon[gbs_df_maxcon < 0] = 0

# Combine maps with equal weighting:
gbs_maxcon = 1 - (gbs_bf_maxcon+gbs_df_maxcon) / 2

# Use gaussian method to smooth noise and sharpen boundaries:
gbs_smoothed = mean(img_as_ubyte(gbs_maxcon), selem=disk(1))
#gbs_smoothed = gaussian(gbs_maxcon, sigma=0.5)
gbs_sharpened = unsharp_mask(gbs_smoothed, radius=5, amount=2)

# Use multiple otsu thresholding to identify candidate boundaries:
multi_thresholds = threshold_multiotsu(gbs_sharpened, classes=4, nbins=256)
gbs_regions = np.digitize(gbs_sharpened, bins=multi_thresholds)

# Extract last region as grain candidates:
grains = (gbs_regions == gbs_regions.max())

# Execute decimation procedure:
#   - remove and save grain cores with diameter < threshold
#   - then erode the remaining map
#   - loop until there's nothing left to erode

grains_eroding = grains
grain_cores = np.zeros_like(grains_eroding)
iterations = 0
grain_progress = np.zeros_like(grains_eroding)
radius_threshold = 20

while np.any(grains_eroding) and False:

    at = int(np.pi / 4 * radius_threshold**2)
    grains_stripped = area_opening(
            grains_eroding, area_threshold=at, connectivity=1)
    grain_cores = np.logical_or(grain_cores,
            np.logical_xor(grains_eroding, grains_stripped))

    grain_progress = np.where(grains_eroding != 0, grain_progress + 1,
            grain_progress)
    iterations = iterations + 1
    print(f'Decimation cycle: {iterations}')

    #grains_eroding = binary_erosion(grains_stripped, selem=disk(3))
    grains_eroding = binary_erosion(grains_stripped)

# identify grain cores and locate centroids:

grain_cores_labels = label(grain_cores > 0, connectivity=2)
grain_cores_props = regionprops(grain_cores_labels)
grain_centroids = np.asarray([prop.centroid for prop in grain_cores_props])
grain_centroid_coors = grain_centroids.astype(int)
markers = np.zeros_like(grain_cores_labels)

# Patch pixel-sized glitches stemming from middle regions:

# Locate small grains (bright regions of dark field):
#threshold_df = threshold_otsu(gbs_df_maxcon)
#gbs_regions_df = gbs_df_maxcon > threshold_df

gbs_opening = gbs_sharpened
opening_progress = np.zeros_like(gbs_opening)
erosion_radius = 3
dilation_radius = 1.5

while np.any(gbs_opening):

    opening_progress = np.where(gbs_opening != 0, opening_progress + 1,
            opening_progress)
    gbs_opening = erosion(gbs_opening, selem=disk(erosion_radius))
    gbs_opening = dilation(gbs_opening, selem=disk(dilation_radius))

opening_progress = opening_progress / opening_progress.max()
print(opening_progress)

#result = gbs_regions
result = opening_progress
#result = grain_progress

#gbs_eroding = gbs_cleaned
#grain_cores = np.zeros_like(gbs_eroding)
#iterations = 0

#for _ in range(10):
    #gbs_eroding = opening(gbs_eroding, selem=disk(1.5))
    #gbs_eroding = opening(gbs_eroding, selem=disk(3))
    #gbs_eroding = erosion(gbs_eroding, selem=disk(1))

#gbs_flooding = gbs_sharpened
#flood_mask = np.zeros_like(gbs_flooding)

#for _ in range(1000):
#    seed_index = np.argmin(gbs_flooding+flood_mask)
#    seed_point = np.unravel_index(seed_index, gbs_flooding.shape)
#    new_flood_mask = flood(gbs_sharpened, seed_point, tolerance=0.1)
#    flood_mask = flood_mask + new_flood_mask






# Use grain cores as markers for watershed algorithm:

#grain_cores_labels = label(grain_cores > 0, connectivity=2)
#grain_cores_props = regionprops(grain_cores_labels)
#grain_centroids = np.asarray([prop.centroid for prop in grain_cores_props])
#grain_centroid_coors = grain_centroids.astype(int)
#markers = np.zeros_like(grain_cores_labels)
#
#for marker in range(grain_centroid_coors.shape[0]):
#    px_x = grain_centroid_coors[marker, 0]
#    px_y = grain_centroid_coors[marker, 1]
#    markers[px_x, px_y] = marker + 1

##markers = grain_cores_labels
#grain_labels = watershed(1 - gbs_sharpened, markers)
##grain_labels = watershed(1 - gbs_sharpened, markers, mask=gbs_sharpened)
#
##rag = graph.rag_mean_color(gbs_sharpened, grain_labels, mode='distance')
#
##grain_labels_merged = graph.merge_hierarchical(grain_labels, rag, thresh=0.1, rag_copy=True,
##                                     in_place_merge=True,
##                                     merge_func=merge_mean_color,
##                                     weight_func=_weight_mean_color)
#
#
#result = label2rgb(grain_labels, image=gbs_sharpened, bg_label=0)
#result = mark_boundaries(result, grain_labels)
##result = label2rgb(markers, image=gbs_sharpened, bg_label=0)
##print(np.unique(grain_cores_labels))

# Convert grain cores to RGBA image:
#result = np.zeros((grain_cores.shape[0], grain_cores.shape[1], 4))
#result[:, :, 0] = grain_cores > 0
#result[:, :, 3] = grain_cores > 0

# Plot results:

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img_bf)
ax[0].set_title('Original Bright Field')
ax[1].imshow(img_df)
ax[1].set_title('Original Dark Field')
ax[2].imshow(gbs_sharpened, cmap=plt.cm.gray)
ax[2].set_title('Sharpened Combined GBs')
#ax[3].imshow(gbs_sharpened, cmap=plt.cm.gray)
#ax[3].imshow(result, cmap=plt.cm.gray)
#ax[3].imshow(result)
ax[3].imshow(result, cmap=plt.cm.nipy_spectral)
#ax[3].scatter(grain_centroid_coors[:,1], grain_centroid_coors[:,0])
#ax[3].scatter(grain_centroids[:,1], grain_centroids[:,0])
ax[3].set_title('Decimated Grain Cores')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
