#  Calculate rank-order distances.
#  Construct distance matrix using rank order distance measure [1].
#
#  $$d_m(a,b)=\sum_{i=0}^{min(O_a(b),k)} I_b(O_b(f_a(i)),k)$$
#
#  $$D(a,b)=\frac{d_m(a,b) + d_m(b,a)}{min(O_a(b),O_b(a))}$$
#  where $I_b$ is indicator fuction: 0 if NN is shared; else, 1.
#
#  @param nn_ids - Nxk matrix of the indices of k-NN for N samples
#
#  @return D     - RO distance matrix
#
#  @author Joseph P. Robinson
#  @date 2016 August 11
#
#  [1] Otto, Charles, Dayong Wang, and Anil K. Jain. "Clustering millions
#  of faces by identity." arXiv preprint arXiv:1604.00989 (2016).
#
from utils.io import is_file
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def build_nn_lists(feature_file, k=20, algorithm='kd_tree'):
    """Build knn lists for N samples. Return Nxk array with N being each sample index and k being knn indices"""

    print('##### Build Tree #####')

    feature_dict = pd.read_pickle(feature_file)


    #  Compute top-k NN for each face encoding
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=algorithm).fit(feature_dict['X'])
    distances, indices = nbrs.kneighbors(feature_dict['X'])

    return indices


def calculate_rank_order_distance(nn_ids):

    nsamples, k = nn_ids.shape
    # rank order distances between samples and k-NN
    D = np.full((nsamples,nsamples), np.inf)

    # track samples that have been compared
    touched = np.zeros(nsamples).astype(np.float)
    for x in tqdm(range(nsamples)):
        # for each sample

        # current face encoding (i.e., face a)
        a_id = x                               # id of sample face a
        a_nnlist = nn_ids[a_id,:] - 1            # NN list for face a

        for y, b_id in enumerate(a_nnlist):
            # for each of its k-NN, id of sample face b
            # kth NN of face a (i.e., face b)

            # if face_b has been as face_a

            if not np.isinf(D[b_id,a_id]):
                # if distance between pairs was already calculated
                continue

            b_nnlist = nn_ids[b_id,:] - 1        # NN list for face a


            # if face a is in face b's NN list, then determine its rank;
            rank_a = np.where(b_nnlist == a_id)[0]

            if len(rank_a)==0:
                rank_a = [k - 1]
                # if rank_a < k and b_id has been touched, then distance has
                # already been computed between faces (with a-b flipped)
            elif touched[b_id]==1:
                print("?")
            #  determine rank of face b in face a's NN list (just y)
            rank_b = y

            #  assymmetric rank order distance (0 for every shared NN; else, 1);
            d_ba = sum([np.any(a == b_nnlist) for a in a_nnlist[:rank_b]])
            d_ab = sum([np.any(b == a_nnlist) for b in b_nnlist[rank_a]])

            #  rank-order distance
            denom = rank_a[0] if rank_a[0] < rank_b else rank_b
            # if denom==0:
            #     print()
            D[a_id,b_id] = (d_ab + d_ba)/denom if denom > 0 else 0
            D[b_id,a_id] = D[a_id,b_id]
            if np.isnan(D[b_id,a_id]):
                print()
        #  mark current face for being compared to its entire NN list
        touched[a_id] = 1
    return D

if __name__ == '__main__':
    build_knn = False
    calculate_dmatrix = False
    k = 20
    file_knn_matrix = 'kd_tree_20.pkl'
    file_d_matrix = 'dmatrix_20.pkl'
    file_feature = '../pytorch-face/data/eval-features.pkl'
    print('Rank order')
    start = time.time()

    if not build_knn and is_file(file_knn_matrix):
        ids_mat = pd.read_pickle(file_knn_matrix)
        print('##### {} Closest Points for {} samples.s#####'.format(str(k), len(ids_mat)))
    elif not is_file(file_feature):
        print('No features ({}) or NN Lists ({}) files Exists.\nExit(0)'.format(file_feature, file_knn_matrix))
        exit(0)
    else:
        ids_mat = build_nn_lists(file_feature)
        pd.to_pickle(ids_mat, file_knn_matrix)
    if not calculate_dmatrix and is_file(file_d_matrix):
        dmatrix = pd.read_pickle(file_d_matrix)
    else:
        print('Calculating distance matrix')
        dmatrix =calculate_rank_order_distance(ids_mat)
        pd.to_pickle(dmatrix, file_d_matrix)


    print("Took {}".format(time.time() - start))

    # D.tofile('Dmatrix_new.csv', sep=',', format='%10.5f')


    # 44.91146397590637
    print()