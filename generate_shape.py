import scipy.io
import numpy as np
from skimage import io, filters
import graph_IRGS as graph
import time
from utils import *
from copy import deepcopy

matfile = scipy.io.loadmat('D:\\Data\\Semisupervised_graph\\Multi_folder\\20100605_163323\\local_to_slic.mat')
node_label = matfile['label_sp']
segmention_labels = matfile['irgs_to_slic']

hh = io.imread('D:\\Data\\Semisupervised_graph\\Multi_folder\\20100605_163323\\imagery_HH4_by_4average.tif')
hv = io.imread('D:\\Data\\Semisupervised_graph\\Multi_folder\\20100605_163323\\imagery_HV4_by_4average.tif')

edge_map0 = filters.sobel(hh)
edge_map1 = filters.sobel(hv)
edge_map = edge_map0 + edge_map1

img = np.zeros((hh.shape[0],hh.shape[1],2))
img[:,:,0] = hh/255
img[:,:,1] = hv/255


#CONSTRUCTING A RAG
start = time.time()

rag = graph.rag_IRGS_boundary_edge_stregth_shape(img,segmention_labels,edge_map)
end = time.time()
m, s = divmod(end - start, 60)
print("RAG constructed. Time elapsed: {:.0f}m:{:.0f}s".format(m,s))
num_node = len(rag)
shape_sp = np.zeros((num_node, 1))
for n in rag:
    shape_sp[n,0] = rag.nodes[n]['shape']

scio.savemat('D:\\Data\\Semisupervised_graph\\Multi_folder\\20100605_163323\\shape_sp.mat', {'shape_sp':shape_sp})

print('Done')