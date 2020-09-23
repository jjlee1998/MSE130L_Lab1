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
from skimage.segmentation import watershed, random_walker, mark_boundaries
from skimage.measure import label, regionprops
from skimage.future import graph


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])



# Import images in RGB space:
temp = 15 
img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_140.png'))
img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_140.png'))

# Isolate greyscale maps of grain boundaries, then combine:
gbs_bf = sobel(rgb2gray(img_bf))
gbs_df = rgb2gray(img_df)
gbs = 1 - (gbs_bf+gbs_df) / 2

# Remap to maximize contrast across middle 95% of data:
retain = 95
minval = np.percentile(gbs, (100-retain)/2)
maxval = np.percentile(gbs, (retain+100)/2)
gbs_maxcon = (gbs - minval) / (maxval - minval)
gbs_maxcon[gbs_maxcon < 0] = 0
gbs_maxcon[gbs_maxcon > 1] = 1

# Use gaussian method to sharpen boundaries:
gbs_sharpened = unsharp_mask(gbs_maxcon, radius=5, amount=1)

# Execute decimation procedure:
#   - remove and save grain cores with diameter < threshold
#   - then erode the remaining map
#   - loop until there's nothing left to erode
#   - 35C: d5, s2
#   - 15C: d20, s10

gbs_eroding = gbs_sharpened
grain_cores = np.zeros_like(gbs_eroding)
iterations = 0

while np.any(gbs_eroding):
    gbs_stripped = area_opening(
            gbs_eroding, area_threshold=10, connectivity=2)
    #gbs_stripped = diameter_opening(
    #        gbs_eroding, diameter_threshold=5, connectivity=2)
    grain_cores = grain_cores + (gbs_eroding - gbs_stripped)
    gbs_eroding = erosion(gbs_stripped, selem=disk(2))
    iterations = iterations + 1
    print(f'Decimation cycle: {iterations}')

# Use grain cores as markers for watershed algorithm:

grain_cores_labels = label(grain_cores > 0, connectivity=2)
grain_cores_props = regionprops(grain_cores_labels)
grain_centroids = np.asarray([prop.centroid for prop in grain_cores_props])
grain_centroid_coors = grain_centroids.astype(int)
markers = np.zeros_like(grain_cores_labels)

for marker in range(grain_centroid_coors.shape[0]):
    px_x = grain_centroid_coors[marker, 0]
    px_y = grain_centroid_coors[marker, 1]
    markers[px_x, px_y] = marker + 1

#markers = grain_cores_labels
grain_labels = watershed(1 - gbs_sharpened, markers)
#grain_labels = watershed(1 - gbs_sharpened, markers, mask=gbs_sharpened)

rag = graph.rag_mean_color(gbs_sharpened, grain_labels, mode='distance')

grain_labels_merged = graph.merge_hierarchical(grain_labels, rag, thresh=0.1, rag_copy=True,
                                     in_place_merge=True,
                                     merge_func=merge_mean_color,
                                     weight_func=_weight_mean_color)


result = label2rgb(grain_labels_merged, image=gbs_sharpened, bg_label=0)
result = mark_boundaries(result, grain_labels_merged)
#result = label2rgb(markers, image=gbs_sharpened, bg_label=0)
#print(np.unique(grain_cores_labels))

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
ax[3].imshow(gbs_sharpened, cmap=plt.cm.gray)
ax[3].imshow(result)
#ax[3].scatter(grain_centroids[:,1], grain_centroids[:,0])
ax[3].set_title('Decimated Grain Cores')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
