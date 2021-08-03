import scipy.io
import numpy as np
from skimage import data
from skimage import io
# from skimage.future import graph
import graph_IRGS as graph
from skimage.segmentation import slic
from matplotlib import pyplot as plt
import networkx as nx
from utils import *
from scipy import ndimage as ndi

from skimage import segmentation, filters, color


# idx = np.array([0,2,4,5,6])
# idx_map = {j: i for i, j in enumerate(idx)}


# img = data.astronaut()
# labels = slic(img, n_segments=100, compactness=10, start_label=0)
# edge_map = filters.sobel(color.rgb2gray(img))
# rag = graph.rag_IRGS_boundary(img, labels, edge_map, season=None)
# io.imshow(img)
# io.show()


# irgs = io.imread('Result_007.bmp')
# land = io.imread('landmask.bmp')
# relabel_IRGS(irgs, land_val=0, landmask=land)

# labels = scio.loadmat('IRGS_to_slic_python.mat')['irgs_to_slic']
# labels = labels.astype(np.int64)

# hh = io.imread('imagery_HH4_by_4average.tif')
# hv = io.imread('imagery_HV4_by_4average.tif')

# # Or using img for sobel. Does it need to be normalized?
# edge_map0 = filters.sobel(hh)
# edge_map1 = filters.sobel(hv)
# edge_map = edge_map0 + edge_map1

# img = np.zeros((hh.shape[0],hh.shape[1],2))
# img[:,:,0] = hh/255
# img[:,:,1] = hv/255

# rag = graph.rag_IRGS_boundary(img, labels, edge_map, season=None, sigma=1)

# For MAGIC360 local

# bil_path = 'D:\\Data\\21_scenes_GT_2021\\20100418_163315\\dualband_UW_4_by_4_average_BoundaryResult\\Result_001\\Result_001.bil'
# landmask = io.imread('D:\\Data\\21_scenes_GT_2021\\20100418_163315\\landmask.bmp')
# ground_truth = io.imread('D:\\Data\\21_scenes_GT_2021\\20100418_163315\\labeled_img_local_grey.png')
# row, col = read_bil_hdr(bil_path)
# seg = read_IRGS_bil(bil_path, row, col)
# # segmention_labels = relabel_IRGS(seg, land_val=-2, landmask=landmask, b_save = False)
# segmention_labels = scio.loadmat('local_to_slic_python.mat')['irgs_to_slic']

# label_sp_from_GT(segmention_labels, ground_truth)


matfile = scipy.io.loadmat('D:\\Data\\Semisupervised_graph\\Multi_folder\\20100418_163315\\local_to_slic.mat')

node_label = matfile['label_sp']


print('done')