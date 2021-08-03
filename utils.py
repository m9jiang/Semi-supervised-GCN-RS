import numpy as np
import scipy.io as scio
from skimage import io, measure
from skimage.util import dtype
import os

# MAGIC 5
#   AutoPolygonMask.bil
#       land: 0, boundaries: 254, autopolygon label starts from 1
#   IRGS: 
#       The segmentation label starts from 0 to num_class-1. land is 0. No boundaries. ONLY BMP RESULT
#   Local: 
#       land: -2, no boundaries, label starts from 1 to num_autopoly*num_class(defualt 4) FOR BIL RESULT
#   Global:
#       land: -2, no boundaries, label starts from 1 to num_class. However, the boundaries between land and non-land has value from num_class+1 to 400+. FOR BIL RESULT
#       land is 31. no boundaries, label starts from 32 to num_class+31. However, the boundaries between land and non-land has value except [31:num_class+31]. FOR BMP RESULT       

# MAGIC360: 
#   IRGS:
#       No boundaries between regons. land is 0. label starts from 0 to num_class-1. landmask is require to distinguish land and class 0
#   Local:
#       land: -2, boundaries: -2, label starts from 1 to num_autopoly*num_class(defualt 4) FOR BIL RESULT
#   Global:
#       land and boundaries are -2. label starts from 0 to num_class-1. FOR BIL RESULT
#       land and boundaries are 254. label starts from 0 to num_class-1. FOR BMP RESULT
def relabel_IRGS(irgs, land_val=254, landmask=None, b_save=True, b_return = True):

    relabeled = np.zeros((irgs.shape[0],irgs.shape[1]),dtype=np.float )
    new_label = 0
    irgs_label_list = np.unique(irgs)
    num_class = len(irgs_label_list)
    if np.unique(irgs)[-1] == 254 and land_val == 254 and landmask is None:
        num_class = num_class - 1
        relabeled[irgs==land_val] = 1e7
    if land_val == 0 and landmask is not None:  # IRGS
        relabeled[landmask==land_val] = 1e7
    if np.unique(irgs)[0] == land_val and land_val != 0 and landmask is not None:  # local, land and boundaries are -2
        relabeled[irgs==land_val] = 1e7
        num_class = num_class - 1
        irgs_label_list = irgs_label_list[1:]

    for i in range(num_class):
        bw_img = np.zeros((irgs.shape[0],irgs.shape[1]),dtype=np.uint8)
        bw_img[irgs==irgs_label_list[i]] = 1
        # bw_img[landmask==land_val] = 0
        bw_label, bw_label_num = measure.label(bw_img,connectivity=2, return_num=True)  # bw_label_num is the maximum label index, not Number of labels (bw_label_num+1)
        # print("Number of superpixels for Class {}: {}".format(i, bw_label_num))
        for j in range(1, bw_label_num + 1):
            relabeled[bw_label==j] = new_label  #relabeled = np.where(an_array > bw_label==j, new_label, relabeled)
            new_label += 1

    if land_val == 0 and landmask is not None:
        relabeled[landmask==land_val] = new_label
        print("Number of superpixels for whole image plus land: {}".format(new_label+1))
    elif land_val != 0 and landmask is not None:
        relabeled[irgs==land_val] = new_label
        # relabeled[relabeled==1e7] = new_label
        print("Number of superpixels for whole image plus land: {}".format(new_label+1))        
    else:
        print("Number of superpixels for whole image: {}".format(new_label))
    if b_save:
        scio.savemat('local_to_slic_python.mat', {'irgs_to_slic':relabeled})
    if b_return:
        return relabeled

def label_sp_from_GT(seg_label, gt, landmask = True):
    num_class = len(np.unique(seg_label))
    # if landmask:
    #     num_class = num_class - 1
    label_sp = np.zeros((num_class), dtype=np.int)
    for i in range(num_class):
        list_temp = np.unique(gt[seg_label==i])
        if len(list_temp) == 1 and list_temp != 0:
            label_sp[i] = list_temp
    num_labeled_sp = np.count_nonzero(label_sp)
    print("Number of labeled superpixels : {}".format(num_labeled_sp))
    if landmask:
        print("Number of superpixels : {}".format(num_class-1))
    else:
        print("Number of superpixels : {}".format(num_class))

    




def read_bil_hdr(bil_path):
    bil_path_no_ext = bil_path[:bil_path.rindex('.')]
    hdr_path = bil_path_no_ext + '.hdr'
    f = open(hdr_path,'r')
    lines = f.readlines()
    for lines in lines:
        if "lines = " in lines:
            row = int(lines[lines.rindex(' = ')+3:])
            print('row = {}'.format(row))
        elif "samples = " in lines:
            col = int(lines[lines.rindex(' = ')+3:])
            print('col = {}'.format(col))
    
    return row, col

def read_IRGS_bil(bil_path, bil_rows, bil_cols):
    '''This function is a simple way to read bil file from MAGIC
    Data type of IRGS results is int32, while autopolygon is int8'''
    # bil_nodata = -9999
    # hdr_path = bil_path[:-3] + '.hdr'
    bil_array = np.fromfile(bil_path, dtype=np.int32)


    bil_array = bil_array.reshape(bil_rows, bil_cols)
    # bil_array[bil_array == bil_nodata] = np.nan
    return bil_array

def write_hdr(save_dir, row, col):
    '''Write the corresponding.hdr file for .bil file
    this funtion is stupid but works'''
    h1 = 'ENVI'
    h2 = 'description = {'
    h3 = '  Image }'
    h4 = 'samples = '+str(col)    #col
    h5 = 'lines = '+str(row)    #row
    h6 = 'bands = '+str(1)     
    h7 = 'header offset = 0'
    h8 = 'file type = ENVI Standard'
    h9 = 'data type = 3'          #3: int32
    h10 = 'interleave = BIL'
    h11 = 'byte order = 0'
    h12 = 'pixel size = {1, 1, units=METERS}'
    h13 = 'data gains = {'
    h14 = '  1}'
    h15 = 'data offsets = {'
    h16 = '  0}'
    h17 = 'band names = {'
    h18 = '  SegmentResult}'
    h=[h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18]
    doc = open(save_dir + '.hdr', 'w')
    for i in range (0,len(h)):
        print(h[i], end='', file=doc)
        print('\n', end='', file=doc)
    doc.close()

def write_bil(img, save_dir):
    '''save_dir is the file path without extension'''
    dir = os.path.dirname(save_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    img = img.astype(np.int32)
    img.tofile(save_dir + '.bil')
    write_hdr(save_dir, img.shape[0], img.shape[1])