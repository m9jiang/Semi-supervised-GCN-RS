import numpy as np
from skimage import data
from skimage import io
from skimage.future import graph
from skimage.segmentation import slic
from matplotlib import pyplot as plt
import networkx as nx
from utils import *


# idx = np.array([0,2,4,5,6])

# idx_map = {j: i for i, j in enumerate(idx)}


# img = data.astronaut()
# labels = slic(img, n_segments=100, compactness=10, start_label=0)
# io.imshow(img)
# io.show()

# rag = graph.rag_mean_color(img, labels)
# plt.show()

# A = nx.adjacency_matrix(rag)
irgs = io.imread('Result_002.bmp')
land = io.imread('landmask.bmp')
# relabel_IRGS(irgs, land_val=0, landmask=land)
relabel_IRGS(irgs)

print('done')