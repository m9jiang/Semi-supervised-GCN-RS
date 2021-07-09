import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# from skimage.future import graph
import graph_IRGS as graph
import networkx as nx
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def euclidean_norm_distance_metric(m_sp_i, m_sp_j):
    dis = np.sum(np.power(m_sp_i - m_sp_j, 2))
    return dis

WEIGHT_SCALAR = 10  # The scalar in the wetghts in the calculation of S_i_w. The greater, the bigger the weights.
b_landmask = True
is_normalized = True
matfile = scipy.io.loadmat('irgs_to_slic.mat')
node_label = scipy.io.loadmat('label_mat.mat')['label_mat']

segmention_labels = matfile['irgs_to_slic']
del matfile

hh = io.imread('imagery_HH4_by_4average.tif')
hv = io.imread('imagery_HV4_by_4average.tif')

# skimage.future.graph does not support 2-channel image
# img = np.zeros((hh.shape[0],hh.shape[1],2))
# img[:,:,0] = hh
# img[:,:,1] = hv

img = hh

#CONSTRUCTING A RAG
rag = graph.rag_mean_color(img, segmention_labels)
print("RAG constructed. Time elapsed: {:.4f}s".format(time.time() - t_total))
# lc = graph.show_rag(segmention_labels, rag, img)
# io.show()

# TODO: replace there_is_land with b_landmask
there_is_land = np.any(segmention_labels == 10000000)
if there_is_land:
    rag.remove_node(10000000)

num_node = len(rag) # number of superpixels

feats = np.zeros((hh.shape[0],hh.shape[1],2))
feats[:,:,0] = hh
feats[:,:,1] = hv
num_feats = feats.shape[2]

if is_normalized:
    for feat in range(0, num_feats):
        min_f = np.min(feats[:, :, feat])
        max_f = np.max(feats[:, :, feat])
        # feats[:, :, feat] = (feats[:, :, feat] - min_f) / (max_f - min_f)
        feats[:, :, feat] = feats[:, :, feat]/255

# Thought: replace mean of the coordinatesby with the centre of Minimum Enclosing Rectangle (MER) or circumcircle
node_coord = np.zeros((num_node, 2), dtype=int)  # mean of the coordinates of each node
node_feat = np.zeros((num_node, num_feats))

for num_sp in range(0, num_node):  # the labels of superpixels are in range [0, num_node)
    idxs = np.where(segmention_labels == num_sp)  # gives us a tuple-typed output of rows and columns idxs[0]-->rows
    node_coord[num_sp, 0] = np.mean(idxs[0])  # mean of rows
    node_coord[num_sp, 1] = np.mean(idxs[1])  # mean of cols
    node_feat[num_sp, :] = np.mean(feats[idxs], axis=0)  # mean of feats



adj_mat = nx.adjacency_matrix(rag)
 # build symmetric adjacency matrix
adj_mat = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)

labels = encode_onehot(node_label)

features = normalize(node_feat)
adj_mat = normalize(adj_mat + sp.eye(adj_mat.shape[0]))

idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)

features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])
adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


# S_i_w = np.zeros((num_node, num_feats))
# for num_sp in rag:
#     num_of_neighbors = len(rag[num_sp])
#     ls_neighbours_idxs = sorted(rag.adj[num_sp].keys()) # the label of neighbours of the current node

#     w_i_zj = np.zeros((num_of_neighbors, 1))
#     sum_w = 0

#     for num_neighbors in range (0, num_of_neighbors):
#         w_i_zj[num_neighbors, 0] = np.exp(- euclidean_norm_distance_metric(node_feat[num_sp, :],
#                                             node_feat[ls_neighbours_idxs[num_neighbors], :]) / WEIGHT_SCALAR)
#         sum_w += np.exp(- euclidean_norm_distance_metric(node_feat[num_sp, :],
#                                                        node_feat[ls_neighbours_idxs[num_neighbors], :]) / WEIGHT_SCALAR)
#     w_i_zj = w_i_zj / sum_w

#     for num_neighbors in range(0, num_of_neighbors):
#         S_i_w[num_sp, :] += w_i_zj[num_neighbors, 0] * node_feat[ls_neighbours_idxs[num_neighbors], :]

# #CALCULATING W
# # W has a size of num_node*num_node and the element ij, for example, in the matrix represents the weight between the
# # regions (superpixels) that have labels i and j.
# W = np.zeros((num_node, num_node))


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
