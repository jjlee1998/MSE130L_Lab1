import numpy as np
import pandas as pd
import ray
import psutil
import time
import os
from tqdm import tqdm
from math import floor
from skimage.color import rgb2gray
from skimage.io.collection import ImageCollection

def process_img(img):
    
    img_gray = rgb2gray(img)
    dA = 1 / img_gray.size
    ii = np.sum(img_gray * dA)
    return ii

@ray.remote
def process_ic(ic, i):

    return process_img(ic[i])
        
def process_temperature(temp):

    num_cpus = psutil.cpu_count(logical=True)
    ray.init(num_cpus=num_cpus)
    folder = f'./MSE130_Lab_1_Data/{temp}C/*.png'
    ic = ImageCollection(folder, conserve_memory=True)
    image_id = [int(os.path.basename(ipath)[3:8]) for ipath in ic.files]
    ic_id = ray.put(ic)
    ic_len = ic.__len__()

    def to_iterator(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    obj_ids = [process_ic.remote(ic_id, i) for i in range(ic_len)]
    ray_result = [x for x in tqdm(to_iterator(obj_ids), total=len(obj_ids))]
    ii = np.asarray(ray_result)

    data = {'image_id': image_id, 'integrated_intensity': ii}
    df = pd.DataFrame(data=data)
    df.set_index('image_id', inplace=True, drop=True)
    df.to_csv(f'./ii_{temp}C.csv')
    ray.shutdown()

if __name__ == '__main__':

    process_temperature(35)
    process_temperature(30)
    process_temperature(25)
    process_temperature(20)
    process_temperature(15)
    process_temperature(10)
