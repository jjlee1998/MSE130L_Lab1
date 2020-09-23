import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.io.collection import ImageCollection
from joblib import Parallel, delayed

def integrated_intensity(img):

    img_gray = rgb2gray(img)
    dA = 1 / img_gray.size
    ii = np.sum(img_gray * dA)
    print(ii)
    return ii

def process_temperature(temp):

    folder = f'./MSE130_Lab_1_Data/{temp}C/*.png'
    ic = ImageCollection(folder, conserve_memory=True)
    filenames = ic.files
    #ii = [integrated_intensity(img) for img in ic]
    ii = Parallel(n_jobs=8)(delayed(integrated_intensity)(img) for img in ic)

if __name__ == '__main__':

    process_temperature(25)
