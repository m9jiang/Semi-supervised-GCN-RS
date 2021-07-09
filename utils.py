import numpy as np
import scipy.io as scio
from skimage import io, measure



# MAGIC360: 
#   IRGS:
#       No boundaries between regons. land is 0. label starts from 0 to num_class-1. landmask is require to distinguish land and class 0
#   Local:
#   Global:
#       land and boundaries are 254. label starts from 0 to num_class-1.
def relabel_IRGS(irgs, land_val=254, landmask=None, b_save=True, b_return = True):

    relabeled = np.zeros((irgs.shape[0],irgs.shape[1]),dtype=np.float )
    new_label = 0

    num_class = len(np.unique(irgs))
    if np.unique(irgs)[-1] == 254 and land_val == 254 and landmask is None:
        num_class = num_class - 1
        relabeled[irgs==land_val] = 1e7

    if land_val == 0 and landmask is not None:
        relabeled[landmask==land_val] = 1e7


    for i in range(num_class):
        bw_img = np.zeros((irgs.shape[0],irgs.shape[1]),dtype=np.uint8)
        bw_img[irgs==i] = 1
        bw_label, bw_label_num = measure.label(bw_img,connectivity=2, return_num=True)  # bw_label_num is the maximum label index, not Number of labels (bw_label_num+1)

        print("Number of superpixels for Class {}: {}".format(i, bw_label_num))
        for j in range(1, bw_label_num + 1):
            relabeled[bw_label==j] = new_label  #relabeled = np.where(an_array > bw_label==j, new_label, relabeled)
            new_label += 1
    print("Number of superpixels for whole image: {}".format(new_label))
    if b_save:
        scio.savemat('glocal_to_slic_python.mat', {'irgs_to_slic':relabeled})
    if b_return:
        return relabeled
