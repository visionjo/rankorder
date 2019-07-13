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
from utils.io import load_mat
import numpy as np
from tqdm import tqdm
import time
def calculate_rank_order_distance(nn_ids):

    nsamples, k = nn_ids.shape
    # rank order distances between samples and k-NN
    D = np.ones((nsamples,nsamples))*-1
    # track samples that have been compared
    touched = np.zeros(nsamples).astype(np.float)
    for x in tqdm(range(nsamples)):
        # for each sample

        # current face encoding (i.e., face a)
        a_id = x                               # id of sample face a
        a_nnlist = nn_ids[a_id,:] - 1            # NN list for face a

        for y, b_id in enumerate(a_nnlist):
            # id of sample face b
            ## for each of its k-NN
            # kth NN of face a (i.e., face b)

            # if face_b has been as face_a
            # if distance between pairs was already claculated
            if not D[b_id,a_id]<0:
                continue

            b_nnlist = nn_ids[b_id,:]   - 1        # NN list for face a


            # if face a is in face b's NN list, then determine its rank;
            rank_a = np.where(b_nnlist == a_id)[0]

            # else, if face a is not in face b's NN list, or
            # if (isempty(rank_a) || touched(b_id)), continue; end

            if len(rank_a)==0:
                rank_a = [k - 1]
                # if rank_a < k and b_id has been touched, then distance has
                # already been computed between faces (with a-b flipped)
            elif touched[b_id]==1:
                print("?")


            #          if isempty(rank_a), rank_a = k; end
            #  determine rank of face b in face a's NN list (just y)
            rank_b = y

            #  get face b's NN list
            #         b_vec = nn_ids(b_id,:);

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
    print('Rank order')
    start = time.time()
    ids_mat = load_mat('kdtree_100.mat')
    D =calculate_rank_order_distance(ids_mat['ids_mat'])
    print("Took {}".format(time.time() - start))
    # np.fromfile('Dmatrix.csv', sep=',')
    D.tofile('Dmatrix_new.csv', sep=',', format='%10.5f')

    print()