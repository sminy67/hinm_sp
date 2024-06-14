import logging

import torch
import numpy as np
import time
from lap import lapjv
import pickle
import os

__all__ = [
    "apply_two_axis_perm"
]

_LOGGER = logging.getLogger(__name__)

def get_nnz_vec(vec_sparsity, in_channels, m):
    # how many input channels should be remained
    num_remained_channels = int(in_channels * (1 - vec_sparsity))
    if num_remained_channels % m == 0:
        return num_remained_channels
    else:
        num_nm_groups = int(num_remained_channels / m)
        num_remained_channels = num_nm_groups * m
        return num_remained_channels

def gen_vec_mask(weight: torch.Tensor,
                 vec_size: int,
                 nnz_vec: int,
                 reversed: bool = False):
    weight = weight.abs()
    out_channels, in_channels = weight.shape
    num_grps = out_channels // vec_size

    weight = weight.view(num_grps, vec_size, -1).sum(dim=1)
    topk, _ = torch.topk(weight, nnz_vec, dim=-1)
    thres = topk[:, -1].unsqueeze(-1)

    if not reversed:
        mask = (weight >= thres).float()
    else:
        mask = (weight < thres).float()

    # Expection Handling
    invalid_rows = torch.nonzero(mask.sum(dim=-1) != nnz_vec)[:, 0]
    for row in invalid_rows:
        target_row = weight[row]
        reorder_idx = torch.argsort(target_row)
        mask_row = mask[row]
        mask_row[:] = 0.0
        mask_row[reorder_idx[:nnz_vec]] = 1.0

    mask = mask.repeat_interleave(vec_size, dim=0)

    return mask.view(out_channels, in_channels)

def gen_nm_mask(weight, n=2, m=4, reversed=False):
    
    out_channels, in_channels = weight.shape
    
    weight = weight.view(-1, m)
    topk, _ = torch.topk(weight, n, dim=-1)
    thres = topk[:, -1].unsqueeze(-1)
    
    if not reversed:
        mask = (weight >= thres).float()
    else:
        mask = (weight < thres).float()
        
    equal_id = torch.where(mask.sum(dim=-1) != n)[0].tolist()
    num_equal = len(equal_id)
    if num_equal > 0:
        for eq in equal_id:
            reorder_id = torch.argsort(weight[eq]).tolist()
            if reversed:
                reorder_id = reorder_id[::-1]
            for i in range(n):
                mask[eq, reorder_id[i]] = 0.0
            for i in range(n, m):
                mask[eq, reorder_id[i]] = 1.0
                
    return mask.view(out_channels, in_channels)

def balanced_kmeans(data: np.ndarray,
                    k: int,
                    cluster_size: int,
                    max_iter: int = 20,
                    tolerance: float = 1e-4):

    assert data.shape[0] == (k * cluster_size), \
        "(number of clusters x cluster size) should be total number of data to be grouped"
    # Initialize centroid locations
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for _ in range(max_iter):
        # Assignment Step
        dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        dist = np.repeat(dist, cluster_size, axis=1)
        _, assign_indices, _ = lapjv(dist)
        assignments = np.array([(assign_idx // cluster_size)
                                for assign_idx in assign_indices])

        # Update Step
        new_centroids = np.empty((k, data.shape[1]))
        for i in range(k):
            new_centroids[i] = np.mean(data[assignments == i], axis=0)
        prev_centroids, centroids = centroids, new_centroids

        centroid_diff = np.linalg.norm(centroids - prev_centroids)
        if centroid_diff < tolerance:
            break

    return assignments, centroids

class OutPerm:
    def __init__(self,
                 weight: torch.Tensor,
                 vec_size: int,
                 nnz_vec: int,
                 importance: torch.Tensor = None):

        self.weight = weight.detach().cpu()
        if importance is not None:
            self.imp = importance
        else:
            self.imp = self.weight.abs()

        self.out_chan, self.in_chan = weight.shape

        assert (self.out_chan % vec_size) == 0, \
            "Number of output channels should be multiple of vector size"

        self.perm_idx = torch.arange(self.out_chan)
        self.block_size = vec_size

        self.num_blocks = self.out_chan // self.block_size
        self.nnz_vec = nnz_vec
        self.is_perm = False

    def __ext_multi_chan(self, imp: torch.Tensor, num_ext: int):

        # Random channel extraction
        ext_idx = np.array([], dtype=np.int32)
        rng = np.random.default_rng(77)
        for _ in range(self.num_blocks):
            ext_idx = np.append(
                ext_idx,
                np.sort(
                    rng.choice(self.block_size,
                               size=num_ext,
                               replace=False,
                               shuffle=False)))
        ext_idx = torch.tensor(ext_idx)
        block_offset = torch.repeat_interleave(torch.arange(
            0, self.out_chan, self.block_size),
                                               num_ext,
                                               dim=0)
        ext_idx += block_offset
        ext_imp = imp[ext_idx]

        # Remaining channels
        mask = torch.ones([self.out_chan], dtype=torch.bool)
        mask[ext_idx] = False
        rem_imp = imp[mask]

        return ext_idx, ext_imp, rem_imp

    def __ext_uni_chan(self, imp: torch.Tensor):

        # Random channel extraction
        ext_idx = torch.randint(self.block_size, (self.num_blocks, ))
        block_offset = torch.arange(0, self.out_chan, self.block_size)

        ext_idx += block_offset
        ext_imp = imp[ext_idx]

        # Remaining channels
        mask = torch.ones([self.out_chan], dtype=torch.bool)
        mask[ext_idx] = False
        rem_imp = imp[mask]

        return ext_idx, ext_imp, rem_imp

    def __assign_multi_chan(self, perm_idx: torch.Tensor,
                            ext_idx: torch.Tensor, cluster_idx: np.ndarray,
                            perm_cluster_idx: np.ndarray):
        ext_perm_idx = perm_idx[ext_idx][cluster_idx].view(self.num_blocks, -1)
        asgn_perm_idx = ext_perm_idx[perm_cluster_idx].view(-1)
        perm_idx[ext_idx] = asgn_perm_idx

        imp = self.imp[perm_idx]

        return perm_idx, imp

    def __assign_uni_chan(self, perm_idx: torch.Tensor, ext_idx: torch.Tensor,
                          perm_cluster_idx: np.ndarray):
        ext_perm_idx = perm_idx[ext_idx]
        asgn_perm_idx = ext_perm_idx[perm_cluster_idx]
        perm_idx[ext_idx] = asgn_perm_idx

        imp = self.imp[perm_idx]

        return perm_idx, imp

    def __gen_cost_mat(self,
                       ext_imp: torch.Tensor,
                       rem_imp: torch.Tensor,
                       num_ext: int = 1):

        rem_imp = rem_imp.view(self.num_blocks, self.block_size - num_ext,
                               -1).sum(dim=1)
        rem_imp = torch.repeat_interleave(rem_imp, self.num_blocks, dim=0)
        ext_imp = ext_imp.repeat(self.num_blocks, 1)

        total_imp = rem_imp + ext_imp
        topk, _ = torch.topk(total_imp, self.nnz_vec, dim=-1)
        thres = topk[:, -1].unsqueeze(-1)

        mask = (total_imp < thres).float()

        invalid_rows = torch.nonzero(mask.sum(dim=-1) != self.nnz_vec)[:, 0]

        # Exception Handling
        for row in invalid_rows:
            total_row = total_imp[row]
            reorder_id = torch.argsort(total_row)
            mask_row = mask[row]

            mask_row[:] = 0.0
            mask_row[reorder_id[:self.nnz_vec]] = 1.0

        cost_mat = (total_imp * mask).sum(dim=-1).view(self.num_blocks,
                                                       self.num_blocks)
        return cost_mat

    def get_cost(self, imp: torch.Tensor):
        mask = gen_vec_mask(imp,
                            vec_size=self.block_size,
                            nnz_vec=self.nnz_vec,
                            reversed=True)
        cost = (imp * mask).sum()

        return cost

    def permute_weight(self, weight, perm_idx):
        return weight[perm_idx]

    def fit(self,
            num_exts: list = None,
            use_kmeans: bool = True,
            verbose: int = 0,
            multi_cnts: int = 1,
            uni_cnts: int = 1):

        start = time.time()
        # Generating number of extraction list
        if num_exts is None:
            num_exts = []
            fac = self.block_size
            while (fac > 1):
                fac = fac // 2
                num_exts.append(fac)

        # Initializing the cost
        imp = self.imp
        perm_idx = self.perm_idx

        prev_cost = self.get_cost(imp)
        if (verbose > 0):
            print(f"[INIT] CURRENT COST : {prev_cost}")

        for num_ext in num_exts:
            if num_ext > 1:
                cnt = 0

                # Multi-channel Extraction
                while (1):
                    if cnt >= multi_cnts:
                        break

                    # Extraction Step
                    ext_idx, ext_imp, rem_imp = self.__ext_multi_chan(
                        imp, num_ext)

                    # Clustering Step
                    if use_kmeans:
                        cluster, _ = balanced_kmeans(
                            data=ext_imp.clone().detach().numpy(),
                            k=self.num_blocks,
                            cluster_size=num_ext)
                        cluster_idx = np.argsort(cluster)
                        ext_imp = ext_imp[cluster_idx]
                    else:
                        cluster_idx = np.arange(self.num_blocks * num_ext)

                    # Summation of each cluster of extracted channels
                    ext_imp = ext_imp.view(self.num_blocks, num_ext,
                                           -1).sum(dim=1)

                    # Assignment Step
                    cost_mat = self.__gen_cost_mat(ext_imp, rem_imp, num_ext)
                    cost, perm_cluster_idx, _ = lapjv(cost_mat.numpy())

                    if (cost < prev_cost):
                        if (verbose > 0):
                            print(
                                f"[{num_ext} MULTI-CHANNELS] CURRENT COST : {cost}"
                            )
                        prev_cost = cost
                        perm_idx, imp = self.__assign_multi_chan(
                            perm_idx, ext_idx, cluster_idx, perm_cluster_idx)
                    else:
                        cnt += 1
            else:
                cnt = 0

                # Uni-channel Extraction
                while (1):
                    if cnt >= uni_cnts:
                        break

                    # Extraction Step
                    ext_idx, ext_imp, rem_imp = self.__ext_uni_chan(imp)

                    # Assignment Step
                    cost_mat = self.__gen_cost_mat(ext_imp, rem_imp)
                    cost, perm_cluster_idx, _ = lapjv(cost_mat.numpy())

                    if (cost < prev_cost):
                        if (verbose > 0):
                            print(f"[UNI-CHANNEL] CURRENT COST : {cost}")
                        prev_cost = cost
                        perm_idx, imp = self.__assign_uni_chan(
                            perm_idx, ext_idx, perm_cluster_idx)
                    else:
                        cnt += 1
        self.perm_idx = perm_idx
        self.is_perm = True
        self.cost = cost
        end = time.time()
        self.time = (end - start)

        if verbose > 0:
            print(f"[FINAL] OPTIMIZED COST : {cost}")

        perm_weight = self.permute_weight(self.weight, self.perm_idx)
        perm_mask = gen_vec_mask(perm_weight, self.block_size, self.nnz_vec)

        return self.perm_idx, perm_weight, perm_mask

    def fit_no_perm(self):
        perm_weight = self.permute_weight(self.weight, self.out_perm)
        perm_mask = gen_vec_mask(perm_weight, self.block_size, self.nnz_vec)
        return self.out_perm, perm_weight, perm_mask


class InPerm:
    def __init__(self,
                 weight: torch.Tensor,
                 tile_size: int,
                 n: int = 2,
                 m: int = 4,
                 verbose: int = 0):
        
        self.imp = weight.detach().cpu()
        
        self.out_channels, self.in_channels = weight.shape
        self.n, self.m = n, m
        
        self.tile_size = tile_size
        self.num_tiles = self.out_channels // tile_size
        self.in_perm = torch.zeros([self.num_tiles, self.in_channels],
                                   dtype=torch.int64)
        self.num_parts = self.in_channels // self.m
        self.perm_imp = None
        
        self._is_perm = False
        self.verbose = verbose

    def __ext_vec(self, imp: torch.Tensor):
        imp = imp.view(-1, self.num_parts, self.m)
        
        ext_idx = torch.randint(self.m, (self.num_parts, ))
        ext_imp = imp[:, torch.arange(self.num_parts), ext_idx]

        mask = torch.ones([self.num_parts, self.m], dtype=torch.bool)
        mask[torch.arange(self.num_parts), ext_idx] = False
        rem_imp = imp[:, mask]
        
        return ext_idx, ext_imp, rem_imp

    def __gen_cost_mat(self, ext_imp: torch.Tensor, rem_imp: torch.Tensor):
        rem_imp = rem_imp.view(-1, self.num_parts, self.m -1)
        rem_imp = torch.repeat_interleave(rem_imp, self.num_parts, dim=1)
        
        ext_imp = ext_imp.repeat(1, self.num_parts).unsqueeze(-1)
        cost_mat = torch.cat((rem_imp, ext_imp), dim=-1)
        
        mask = gen_nm_mask(cost_mat.view(-1, self.num_parts * self.num_parts * self.m),
                           reversed=True,
                           n=self.n, m=self.m)
        mask = mask.view(-1, self.num_parts, self.num_parts, self.m)
        
        cost_mat = cost_mat.view(-1, self.num_parts, self.num_parts, self.m)
        cost_mat = (cost_mat * mask).sum(dim=-1).sum(dim=0)
        
        return cost_mat

    def __assign_vec(self, perm, imp, ext_idx, perm_idx):
        # Reorder the permutation by Hungarian algorithm
        ext_idx = torch.arange(0, self.in_channels, self.m) + ext_idx
        ext_perm_idx = perm[ext_idx]
        ins_perm_idx = ext_perm_idx[perm_idx]
        perm[ext_idx] = ins_perm_idx

        # Permute the weight matrix
        ext_imp = imp[:, ext_idx]
        ins_imp = ext_imp[:, perm_idx]
        imp[:, ext_idx] = ins_imp

        return perm, imp
 
    def get_cost(self, imp: torch.Tensor):
        mask = gen_nm_mask(imp,
                           self.n,
                           self.m,
                           reversed=True)
        cost = (imp * mask).sum()

        return cost

    def permute_imp(self):
        assert self.is_perm is True, "Input Channel Permutation Should Be Done First !"

        self.perm_imp = torch.zeros_like(self.imp).type(self.imp.type())
        for tile_id in range(self.num_tiles):
            tile_imp = self.imp[tile_id * self.tile_size: (tile_id + 1) * self.tile_size]
            self.perm_imp[tile_id * self.tile_size: (tile_id + 1) * self.tile_size] = tile_imp[:, self.in_perm[tile_id]]

    def get_inverted_nm_mask(self):
        assert self.is_perm is True, "Input Channel Permutation Should Be Done First !"

        if self.perm_imp is None:
            self.permute_imp()

        perm_mask = gen_nm_mask(self.perm_imp)
        inverted_mask = torch.zeros_like(perm_mask).type(perm_mask.type())
        for tile_id in range(self.num_tiles):
            tiled_mask = perm_mask[tile_id * self.tile_size: (tile_id + 1) * self.tile_size]
            tile_perm = self.in_perm[tile_id]
            inverted_tile_perm = torch.argsort(tile_perm)
            inverted_mask[tile_id * self.tile_size: (tile_id + 1) * self.tile_size] \
                = tiled_mask[:, inverted_tile_perm]

        return inverted_mask
 
    def fit(self, cnts):
        imp = self.imp
        cost = 0
        for tile_id in range(self.num_tiles):
            tiled_imp = imp[tile_id * self.tile_size: (tile_id + 1) * self.tile_size]
            tile_perm = torch.arange(self.in_channels)
            
            prev_tile_cost = self.get_cost(tiled_imp)
            if self.verbose > 0:
                print(
                    f"[INIT] TILE ID : {tile_id} CURRENT COST : {prev_tile_cost}"
                )
            
            cnt = 0
            while (1):
                if cnt > cnts:
                    self.in_perm[tile_id, :] = tile_perm
                    cost += prev_tile_cost
                    break
                
                # Extract an column-wise vector from each partition
                ext_idx, ext_imp, rem_imp = self.__ext_vec(tiled_imp)
                
                # Generate a cost matrix for Hungaian algorithm
                cost_mat = self.__gen_cost_mat(ext_imp, rem_imp)

                # Hungarian algorithm
                tile_cost, perm_idx, _ = lapjv(cost_mat.numpy())
                if self.verbose > 1:
                    print(f"[ITER] CURRENT COST FROM HUNGARIAN : {tile_cost}")

                if tile_cost >= prev_tile_cost:
                    cnt += 1
                else:
                    prev_tile_cost = tile_cost
                    tile_perm, tiled_imp = self.__assign_vec(tile_perm,
                                                             tiled_imp,
                                                             ext_idx,
                                                             perm_idx)
        self.is_perm = True
        if self.verbose > 0:
            print(f"[FINAL] TOTAL OPTIMIZED COST : {cost}")
            
def comp_vec_spar(weight, mask, vec_size, nnz_vec):
    out_channels = weight.size(0)  # the size of outer channels

    # loc matrix: non-zero
    row_idx = torch.nonzero(mask)[:, 0].view(out_channels, nnz_vec)
    col_idx = torch.nonzero(mask)[:, 1].view(out_channels, nnz_vec)
    # abstract the corresponding non-zero value
    comp_weight = weight[row_idx, col_idx]
    # empty for vector index
    vec_idx = torch.tensor([], dtype=torch.int32)

    for i in range(0, out_channels, vec_size):
        vec_idx = torch.cat((vec_idx, col_idx[i]), dim=0)

    return comp_weight, vec_idx.view(out_channels // vec_size, -1)

def get_comb_mask(nm_mask, vec_mask, vec_idx, vec_size):
    # get the number of output channels
    out_channels = nm_mask.shape[0]

    num_tiles = out_channels // vec_size
    vec_idx = vec_idx.view(num_tiles, -1)

    for i in range(num_tiles):
        vec_mask[i * vec_size: (i + 1) * vec_size, vec_idx[i]] \
            = nm_mask[i * vec_size: (i + 1) * vec_size]

    return vec_mask

def get_out_inverted_mask(mask, perm):
    inverted_perm = torch.argsort(perm)
    inverted_mask = mask[inverted_perm]
    
    return inverted_mask

def apply_out_perm(score,
                   vec_size,
                   vec_sparsity):
    
    out_channels, in_channels = score.shape
    nnz_vec = get_nnz_vec(vec_sparsity=vec_sparsity,
                          in_channels=in_channels,
                          m=4)
    
    ocp = OutPerm(weight=score,
                  vec_size=vec_size,
                  nnz_vec=nnz_vec,
                  importance=score)
    _LOGGER.info(f"Starting Outer Perm with vector size of {vec_size} and {vec_sparsity} sparsity")
    out_perm_idx, out_perm_saliency_mat, out_perm_mask = ocp.fit(num_exts=[8, 1, 4, 1, 2, 1, 1],
                                                                 use_kmeans=True,
                                                                 verbose=0,
                                                                 multi_cnts=3,
                                                                 uni_cnts=1)

    inverted_out_mask = get_out_inverted_mask(mask=out_perm_mask,
                                              perm=out_perm_idx)
    inverted_out_mask[inverted_out_mask==0] = -1
    
    return out_perm_idx, inverted_out_mask

def apply_in_perm(score,
                  vec_size,
                  vec_sparsity,
                  n, m):
    
    out_channels, in_channels = score.shape
    nnz_vec = get_nnz_vec(vec_sparsity=vec_sparsity,
                          in_channels=in_channels, m=m)
    
    vec_mask = gen_vec_mask(score, vec_size, nnz_vec)
    comp_score, vec_idx = comp_vec_spar(score, vec_mask, vec_size, nnz_vec)

    ivp = InPerm(weight=comp_score,
                 tile_size=vec_size,
                 n=n, m=m,
                 verbose=0)
    ivp.fit(cnts=2)
    inverted_nm_mask = ivp.get_inverted_nm_mask()
    comb_mask = get_comb_mask(nm_mask=inverted_nm_mask,
                              vec_mask=vec_mask,
                              vec_idx=vec_idx,
                              vec_size=vec_size)
    
    return inverted_nm_mask
    

def apply_gyro_perm(score,
                    vec_size,
                    vec_sparsity,
                    n, m):
    
    out_channels, in_channels = score.shape 
    nnz_vec = get_nnz_vec(vec_sparsity=vec_sparsity,
                          in_channels=in_channels, m=m)
    
    ocp = OutPerm(weight=score,
                  vec_size=vec_size,
                  nnz_vec=nnz_vec,
                  importance=score)
    _LOGGER.info(f"Starting Outer Perm with vector size of {vec_size} and {vec_sparsity} sparsity")
    out_perm_idx, out_perm_saliency_mat, out_perm_mask = ocp.fit(num_exts=[8, 1, 4, 1, 2, 1, 1],
                                                                 use_kmeans=True,
                                                                 verbose=0,
                                                                 multi_cnts=1,
                                                                 uni_cnts=1)
    num_tiles = (out_channels // vec_size)
    vec_comp_saliency_mat, vec_idx = comp_vec_spar(weight=out_perm_saliency_mat,
                                                   mask=out_perm_mask,
                                                   vec_size=vec_size,
                                                   nnz_vec=nnz_vec)
    ivp = InPerm(weight=vec_comp_saliency_mat,
                 tile_size=vec_size,
                 n=n, m=m,
                 verbose=0)
    _LOGGER.info(f"Starting Inner Perm with {num_tiles} number of blocks and 2:4 sparsity")
    ivp.fit(cnts=2)
    inverted_nm_mask = ivp.get_inverted_nm_mask()
    comb_mask = get_comb_mask(nm_mask=inverted_nm_mask,
                              vec_mask=out_perm_mask,
                              vec_idx=vec_idx,
                              vec_size=vec_size)
    inverted_hinm_mask = get_out_inverted_mask(mask=comb_mask,
                                               perm=out_perm_idx)
    
    return inverted_hinm_mask

def apply_two_axis_perm(score,
                    vec_size,
                    vec_sparsity,
                    n, m):
    
    out_channels, in_channels = score.shape 
    nnz_vec = get_nnz_vec(vec_sparsity=vec_sparsity,
                          in_channels=in_channels, m=m)
    
    ocp = OutPerm(weight=score,
                  vec_size=vec_size,
                  nnz_vec=nnz_vec,
                  importance=score)
    _LOGGER.info(f"Starting Outer Perm with vector size of {vec_size} and {vec_sparsity} sparsity")
    out_perm_idx, out_perm_saliency_mat, out_perm_mask = ocp.fit(num_exts=[8, 1, 4, 1, 2, 1, 1],
                                                                 use_kmeans=True,
                                                                 verbose=0,
                                                                 multi_cnts=1,
                                                                 uni_cnts=1)
    num_tiles = (out_channels // vec_size)
    vec_comp_saliency_mat, vec_idx = comp_vec_spar(weight=out_perm_saliency_mat,
                                                   mask=out_perm_mask,
                                                   vec_size=vec_size,
                                                   nnz_vec=nnz_vec)
    ivp = InPerm(weight=vec_comp_saliency_mat,
                 tile_size=vec_size,
                 n=n, m=m,
                 verbose=0)
    _LOGGER.info(f"Starting Inner Perm with {num_tiles} number of blocks and 2:4 sparsity")
    _ = ivp.fit(cnts=2)
    inverted_nm_mask = ivp.get_inverted_nm_mask()
    comb_mask = get_comb_mask(nm_mask=inverted_nm_mask,
                              vec_mask=out_perm_mask,
                              vec_idx=vec_idx,
                              vec_size=vec_size)
    inverted_hinm_mask = get_out_inverted_mask(mask=comb_mask,
                                               perm=out_perm_idx)
    
    inverted_hinm_mask[inverted_hinm_mask==0] = -1
    
    return inverted_hinm_mask