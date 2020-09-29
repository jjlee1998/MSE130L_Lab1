import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb, label2rgb
from skimage.filters import sobel, threshold_otsu,\
        unsharp_mask, threshold_multiotsu
from skimage.filters.rank import mean
from skimage.morphology import disk
from skimage.morphology import binary_erosion, binary_dilation, local_maxima
from skimage.segmentation import watershed, mark_boundaries
from skimage.measure import label
from skimage.util import img_as_ubyte

def locate_grains(temp, mag, res=2, offset=0, multi_otsu=True):
    
    # Import images in RGB space:
    img_bf = rgba2rgb(imread(f'./contrast_images/{temp}C_bf_{mag}.png'))
    img_df = rgba2rgb(imread(f'./contrast_images/{temp}C_df_{mag}.png'))

    # Vertical offset if so required:
    if offset > 0:
        img_bf = img_bf[:-offset,:]
        img_df = img_df[offset:,:]
    elif offset < 0:
        img_bf = img_bf[-offset:,:]
        img_df = img_df[:offset,:]

    # Horizontal crop if necessary:
    if img_bf.shape[1] > img_df.shape[1]:
        img_bf = img_bf[:, :img_df.shape[1]]
    elif img_bf.shape[1] < img_df.shape[1]:
        img_df = img_df[:, :img_bf.shape[1]]

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
    gbs_sharpened = unsharp_mask(gbs_smoothed, radius=5, amount=2)

    # Use thresholding to identify candidate boundaries:
    if multi_otsu:
        multi_thresholds = threshold_multiotsu(gbs_sharpened, classes=4, nbins=256)
        gbs_regions = np.digitize(gbs_sharpened, bins=multi_thresholds)
        grains = (gbs_regions == gbs_regions.max())
    else:
        threshold = threshold_otsu(gbs_sharpened)
        grains = gbs_sharpened > threshold

    # Execute weighted opening algorithm:
    #   - Erode grains using circular regions of erosion radius ER
    #   - (This means that ER ~ minimum detectable grain radius)
    #   - (i.e. decrease ER for smaller microstructures)
    #   - Then dilate by DR < ER (so as to have a net erode)
    #   - Continue until there's nothing left
    #   - Track progress in order to create a heatmap of survival time
    #   - Local maxima of this heatmap represents grain cores

    grains_opening = grains
    opening_progress = np.zeros_like(grains_opening)

    er = res        # erosion radius
    cf = 0.5        # coarseness factor
    dr = er * cf    # dilation radius

    while np.any(grains_opening):

        opening_progress = np.where(grains_opening != 0,
                opening_progress + 1, opening_progress)
        grains_opening = binary_erosion(grains_opening, selem=disk(er))
        grains_opening = binary_dilation(grains_opening, selem=disk(dr))

    opening_progress = opening_progress / opening_progress.max()

    # Generate map of grain cores
    # (connectivity defined so as to equal disk(er))
    grain_cores = local_maxima(opening_progress,
            connectivity=er**2,
            indices=False, allow_borders=True)
    cores_x, cores_y = local_maxima(opening_progress,
            connectivity=er**2,
            indices=True, allow_borders=True)

    # Use grain cores as markers for watershed algorithm

    markers = label(grain_cores, connectivity=2)

    #grain_labels = watershed(-1*grains, markers)
    #grain_labels = watershed(-gbs_sharpened, markers)
    grain_labels = watershed(-opening_progress, markers)

    #grain_map = label2rgb(grain_labels, image=grains, bg_label=0)
    grain_map = label2rgb(grain_labels, image=gbs_sharpened, bg_label=0)
    #grain_map = label2rgb(grain_labels, image=opening_progress, bg_label=0)

    marked_grain_map = mark_boundaries(grain_map, grain_labels)

    #print(np.unique(grain_labels))

    return grain_labels, img_bf, img_df, gbs_maxcon, gbs_sharpened, grains,\
            grain_cores, marked_grain_map, opening_progress

def grain_density(temp, mag, **kwargs):

    grain_labels, img_bf, img_df, gbs_maxcon, gbs_sharpened, grains, \
            grain_cores, marked_grain_map,\
            opening_progress = locate_grains(temp, mag, **kwargs)

    n_grains = grain_labels.max()
    hl_ratio = marked_grain_map.shape[0] / marked_grain_map.shape[1]

    length_um = mag
    height_um = length_um * hl_ratio
    area_um2 = length_um * height_um

    return n_grains/area_um2

def plot_grains(temp, mag, **kwargs):

    grain_labels, img_bf, img_df, gbs_maxcon, gbs_sharpened, grains, \
            grain_cores, marked_grain_map,\
            opening_progress = locate_grains(temp, mag, **kwargs)

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    ax = axes.ravel()

    label_size = 30

    ax[0].imshow(img_bf)
    ax[0].text(0.05, 0.9,'A', color='red', size=label_size, 
            transform = ax[0].transAxes)

    ax[1].imshow(img_df)
    ax[1].text(0.05, 0.9,'B', color='red', size=label_size, 
            transform = ax[1].transAxes)

    ax[2].imshow(gbs_maxcon, cmap=plt.cm.gray)
    ax[2].text(0.05, 0.9,'C', color='red', size=label_size, 
            transform = ax[2].transAxes)

    ax[3].imshow(grains, cmap=plt.cm.gray)
    ax[3].text(0.05, 0.9,'D', color='red', size=label_size, 
            transform = ax[3].transAxes)

    ax[4].imshow(opening_progress, cmap=plt.cm.plasma)
    ax[4].text(0.05, 0.9,'E', color='white', size=label_size, 
            transform = ax[4].transAxes)

    ax[5].imshow(marked_grain_map)
    ax[5].text(0.05, 0.9,'F', color='red', size=label_size, 
            transform = ax[5].transAxes)

    #ax[2].imshow(opening_progress, cmap=plt.cm.nipy_spectral)
    #ax[2].imshow(gbs_sharpened, cmap=plt.cm.gray)
    #ax[2].set_title('Intermediate')
    #ax[3].imshow(marked_grain_map, cmap=plt.cm.nipy_spectral)
    #ax[3].imshow(grain_cores, alpha=1.0*(grain_cores > 0), cmap='Reds')
    #ax[3].scatter(grain_centroid_coors[:,1], grain_centroid_coors[:,0])
    #ax[3].scatter(cores_y, cores_x, s=1, c='r', marker='x')
    #ax[3].set_title('Decimated Grain Cores')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    plot_grains(0, 140, res=3, offset=3, multi_otsu=False)
    plot_grains(10, 140, res=2, offset=18)
    plot_grains(15, 140, res=2)
    plot_grains(20, 140, res=2)
    plot_grains(25, 140, res=1.5)
    plot_grains(30, 140, res=1.5)
    plot_grains(35, 140, res=1.5)

    gd_0 = grain_density(0, 140, res=3, offset=3, multi_otsu=False)
    print(f'0C: {gd_0} gr/um^2')

    gd_10 = grain_density(10, 140, res=2, offset=18)
    print(f'10C: {gd_10} gr/um^2')

    gd_15 = grain_density(15, 140, res=2)
    print(f'15C: {gd_15} gr/um^2')

    gd_20 = grain_density(20, 140, res=2)
    print(f'20C: {gd_20} gr/um^2')

    gd_25 = grain_density(25, 140, res=1.5)
    print(f'25C: {gd_25} gr/um^2')

    gd_30 = grain_density(30, 140, res=1.5)
    print(f'30C: {gd_30} gr/um^2')

    gd_35 = grain_density(35, 140, res=1.5)
    print(f'35C: {gd_35} gr/um^2')

    temp = [0, 10, 15, 20, 25, 30, 35]
    gd = [gd_0, gd_10, gd_15, gd_20, gd_25, gd_30, gd_35]
    data = {'T': temp, 'grain_density_um^-2': gd}

    df = pd.DataFrame(data=data)
    df.set_index('T', drop=True, inplace=True)
    df.to_csv('./output/grain_densities.csv')
