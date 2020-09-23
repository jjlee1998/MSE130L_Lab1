import numpy as np
import pandas as pd
import ray
import psutil
import time
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
def process_ic(ic):

    return [process_img(img) for img in ic]
        
def process_temperature(temp):

    num_cpus = psutil.cpu_count(logical=True)
    ray.init(num_cpus=num_cpus)
    folder = f'./MSE130_Lab_1_Data/{temp}C/*.png'
    ic = ImageCollection(folder, conserve_memory=True)
    total_jobsize = ic.__len__()
    cpu_jobsize = int(ic.__len__() / num_cpus)
    cpu_ic_ids = []
    for cpu in range(num_cpus):
        start = cpu*cpu_jobsize
        end = (cpu+1)*cpu_jobsize if cpu < num_cpus - 1 else ic.__len__()
        cpu_ic = ic[start:end]
        cpu_ic_ids.append(ray.put(cpu_ic))

    def to_iterator(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    obj_ids = [process_ic.remote(cpu_ic_ids[i]) for i in range(num_cpus)]
    for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
        pass

    #ray_result = ray.get([process_ic.remote(cpu_ic_ids[i]) for i in range(num_cpus)])
    #return np.concatenate(ray_result)

if __name__ == '__main__':

    process_temperature(35)
