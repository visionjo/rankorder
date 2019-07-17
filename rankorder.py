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

    feature_dict = pd.read_pickle(feature_file)

    #  Compute top-k NN for each face encoding
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(feature_dict['X'])
    distances, indices = nbrs.kneighbors(feature_dict['X'])

    return indices


def calculate_rank_order_distance(nn_ids):
    nsamples = len(nn_ids)
    # rank order distances between samples and k-NN
    D = np.full((nsamples, nsamples), np.inf)

    # track samples that have been compared
    touched = np.zeros(nsamples).astype(np.float)
    for i, (a_id, a_nnlist) in tqdm(enumerate(nn_ids.items())):
        # for each sample

        # current face encoding (i.e., face a)
        # a_nnlist = nn_ids[key]  # NN list for face a

        bdict = {a: nn_ids[a] for a in a_nnlist}

        for rank_b, (b_id, b_nnlist) in enumerate(bdict.items()):
            # for each of its k-NN, id of sample face b is also its rank on NN of face a.

            if not np.isinf(D[rank_b, a_id]):
                # if already calculated (i.e., if face_b has been as face_a)
                continue

            # b_nnlist = nn_ids[rank_b]  # NN list for face a

            # if face a is in face b's NN list, then determine its rank;
            rank_a = np.where(b_nnlist == a_id)[0]
            # min(O_a, k), where O_a is rank of NN list, thus, set as k if match is not in top k

            if np.size(rank_a) and touched[b_id] == 1:
                # print("Recalculated distances for face pairs in reverse order (i.e., b-a, flipped).")
                print(D[a_id, b_id])
                continue
            elif touched[b_id] == 1:
                # print("Recalculated distances for face pairs in reverse order (i.e., b-a, flipped).")
                continue
            rank_a = int(rank_a) + 1 if len(rank_a) > 0 else k

            rank_b += 1
            denom = rank_a if rank_a < rank_b else rank_b

            #  assymmetric rank order distance (0 for every shared NN; else, 1);
            d_ba = sum([not np.any(a == b_nnlist) for a in a_nnlist[:rank_b]])
            d_ab = sum([not np.any(b == a_nnlist) for b in b_nnlist[:rank_a]])

            D[a_id, b_id] = (d_ab + d_ba) / denom if denom > 0 else np.nan
            D[b_id, a_id] = D[a_id, b_id]
            if np.isnan(D[b_id, a_id]):
                print("NAN. Shouldn't be. Likely an attempt to divide by zero when normalizing above.")
        #  touch, as no need to do calculation between item a and all of its neighbors again (i.e., symmetric)
        # however, since only processing upper triangle part of D matrix, such redundancy should not occur
        touched[a_id] = 1
    return D


def transitively_merge_clusters(D, nn_dict, Eps=1.6, C=None):
    """
    Threshold rank-order distance matrix.

    Transitively step through matrix, merging each pairs w distances below threshold Eps into same cluster.

    Provided constraint matrix C, 'must-' and
    :param D:           Rank order distance matrix
    :param nn_dict:
    :param Eps:         Rank order distance threshold (i.e., epsilon) [default 1.6]
    :param C:           Constraint matrix [default zeros(size(D))]
    :return:            Cluster ID assignments (cluster_tags)

    SEE CALCULATE_RANK_ORDER_DISTANCE()
    """

    #
    #

    if C is None:
        C = np.zeros(D.shape)
    nsamples = D.shape[0]

    #  cluster ids -- initialize all samples in own cluster
    cluster_tags = np.arange(nsamples)

    #  Determine distances that follow below threshold AND are not constrained
    #  according to matrix C as 'cannot link' (i.e., c(i,j) ~= -1) OR matrix C
    #  forces a merge via a reference to 'must link' (i.e., c(i,j) == 1)
    to_merge = np.zeros_like(D)
    to_merge[np.eye(len(to_merge)) == 1] = 1
    to_merge[np.where(D < Eps)] = 1

    # apply constraints
    to_merge[np.where(C == 1)] = 1
    to_merge[np.where(C == -1)] = 0

    for x in range(nsamples - 1):
        #  for each element of upper triangle of symmetric distance matrix
        #      distances = D(x,(x+1):end);
        #      to_merge = distances < Eps;
        c_to_merge = to_merge[x, x:] == 1
        # print(c_to_merge)
        c_tag = np.min(cluster_tags[x:][c_to_merge])
        # c_tag = np.min([cluster_tags[x], cluster_tags[x+1:] & c_to_merge)]);
        cluster_tags[x:][c_to_merge] = c_tag

    return cluster_tags


def check_files_and_conditions(build_knn, calculate_dmatrix, file_knn_matrix, file_d_matrix, file_feature):
    if is_file(file_feature):
        return True
    elif is_file(file_knn_matrix) and not build_knn:
        return True
    elif is_file(file_d_matrix) and not calculate_dmatrix:
        return True
    print("Check features files and/or KNN list or D-Matrix exist.")
    return False


if __name__ == '__main__':
    build_knn = False
    calculate_dmatrix = True
    k = 20
    file_knn_matrix = 'kd_tree_20.pkl'
    file_d_matrix = 'dmatrix_20.pkl'
    file_feature = '../pytorch-face/data/eval-features.pkl'
    print('Rank order')
    if not check_files_and_conditions(build_knn, calculate_dmatrix, file_knn_matrix, file_d_matrix, file_feature):
        exit(0)
    start = time.time()

    ids_mat = pd.read_pickle(file_knn_matrix) if (not build_knn and is_file(file_knn_matrix)) else \
        build_nn_lists(file_feature)
    nn_lut = {id[0]: id[1:] for id in ids_mat}
    dmatrix = pd.read_pickle(file_d_matrix) if (not calculate_dmatrix and is_file(file_d_matrix)) else \
        calculate_rank_order_distance(nn_lut)

    print("Took {}".format(time.time() - start))

    eps_arr = np.arange(0.1, 3, 0.1)
    eps_arr = np.concatenate([list(eps_arr), list(np.arange(3, 20, 1))])
    results_lut = {}
    for eps in tqdm(eps_arr):
        results_lut[eps] = transitively_merge_clusters(dmatrix, nn_lut, eps)

    pd.to_pickle(results_lut, 'results_lut.pkl')
