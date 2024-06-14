import numpy as np
import torch
import argparse
import random
import os

class NMData:
    def __init__(self,
                 weight: torch.Tensor = None,
                 input: torch.Tensor = None,
                 M: int = 16,
                 N: int = 8,
                 K: int = 32,
                 tile_M: int = 16,
                 tile_N: int = 8,
                 tile_K: int = 32,
                 dtype: str = "float16",
                 op_type: str = "nm",
                 vec_sparsity: float = None,
                 transpose_inp: bool = False):
        
        if (weight is None) or (input is None):
            self.gen_weight_input(M, N, K)
        else:
            self.weight = weight
            self.input = input
            
        self.transpose = transpose_inp
        self.op = op_type
        
        if op_type == "nm":
            self.gen_nm_data(self.weight, self.input, tile_M, tile_K, dtype)
        elif op_type == "hier_nm":
            assert vec_sparsity is not None, "Vector sparsity should have a value"
            self.gen_hier_nm_data(self.weight, self.input, tile_M, tile_K, vec_sparsity, dtype)
        else:
            self.gen_nm_data(self.weight, self.input, tile_M, tile_K, dtype)
        
    def gen_weight_input(self, M, N, K):
        self.weight = torch.rand([M, K], dtype=torch.float32)
        self.input = torch.rand([K, N], dtype=torch.float32)
    
    def mask_to_idx(self, vec_mask, M, K):
        row_idx, col_idx = torch.nonzero(vec_mask)[:, 0], torch.nonzero(vec_mask)[:, 1]
        return row_idx.view(M, K), col_idx.view(M, K)
        
    def mask_to_bin(self, tile_mask, numel_per_tile):
        even_threads = tile_mask[8:, :16].clone().detach()
        odd_threads = tile_mask[:8, 16:].clone().detach()
        
        tile_mask[:8, 16:] = even_threads
        tile_mask[8:, :16] = odd_threads
        
        tile_bin_meta = np.zeros((numel_per_tile,), dtype="uint32")
        bit_shift = 4 ** torch.arange(numel_per_tile)
        
        for i in range(numel_per_tile):
            u32_reg = tile_mask[i]
            meta_idx = torch.nonzero(u32_reg.view(-1, 4))[:, 1]
            
            if i < 8:
                tile_bin_meta[2 * i] = torch.sum(bit_shift * meta_idx, -1)
            else:
                k = (i % 8)
                tile_bin_meta[2 * k + 1] = torch.sum(bit_shift * meta_idx, -1)
        
        return tile_bin_meta
    
    def gen_vec_metadata(self, vec_mask, M, K, vec_size, num_vec_grps, nnz_vec): 
        _, col_idx = self.mask_to_idx(vec_mask, M, nnz_vec)
        
        total_num_vecs = num_vec_grps * nnz_vec
        vec_meta = np.zeros((total_num_vecs,), dtype="uint32")
        for i in range(num_vec_grps):
            vec_meta[i * nnz_vec:(i + 1) * nnz_vec] = col_idx[i * vec_size].numpy().astype("uint32")
        
        return vec_meta
    
    def gen_nm_metadata(self, mask, M, K, tile_M, tile_K):
        num_M_tiles = (M // tile_M)
        num_K_tiles = (K // tile_K)
        
        numel_per_tile = 16
        total_numel = numel_per_tile * num_M_tiles * num_K_tiles
        bin_meta = np.zeros((total_numel,), dtype="uint32")
        
        for i in range(num_M_tiles):
            for j in range(num_K_tiles):
                tile_mask = mask[(i * tile_M):((i + 1) * tile_M), (j * tile_K):((j + 1) * tile_K)]
                tile_bin_meta = self.mask_to_bin(tile_mask, numel_per_tile)
                
                bin_meta[(i * num_K_tiles + j) * numel_per_tile : (i * num_K_tiles + (j + 1)) * numel_per_tile] = tile_bin_meta
        
        return bin_meta
    
    def gen_nm_mask(self, weight, M, K):
        weight = weight.abs()
        norm_nm = weight.view(M, -1, 4)
        
        topk_values, _ = torch.topk(norm_nm, 2, dim=-1)
        thres = topk_values[:,:,-1].unsqueeze(-1)
        
        mask = (norm_nm >= thres).float().view(M, K)

        return mask
    
    def gen_vec_mask(self, weight, M, K, vec_size, num_vec_grps, nnz_vec):
        weight = weight.abs()
        
        norm_vec = weight.view(num_vec_grps, vec_size, -1).sum(dim=1)
        topk_values, _ = torch.topk(norm_vec, nnz_vec, dim=-1)
        thres = topk_values[:, -1].unsqueeze(-1)
        
        mask = (norm_vec >= thres)
        ignore = torch.where(mask.float().sum(dim=1) != nnz_vec)[0]
        num_ignore = len(ignore)
    
        if num_ignore > 0:
            ties = (norm_vec == thres).float()
            ignore_row_idx = torch.where(ties.sum(dim=1) > 1)[0].tolist()
            for row_idx in ignore_row_idx:
                ignore_col_idx = torch.where(ties[row_idx] == 1)[0].tolist()
                for i in range(len(ignore_col_idx) - 1):
                    col_idx = ignore_col_idx[i]
                    mask[row_idx, col_idx] = False
                    
        mask = mask.float().repeat_interleave(vec_size, dim=0)
        
        return mask.view(M, K)
    
    def comp_hier_weight(self, weight, vec_mask, nm_mask, M, K, nnz_vec, dtype):
        vec_row_idx, vec_col_idx = self.mask_to_idx(vec_mask, M, nnz_vec)
        nm_row_idx, nm_col_idx = self.mask_to_idx(nm_mask, M, nnz_vec//2)

        mask = np.zeros((M, K), dtype=dtype)
        np_row_idx, np_col_idx = nm_row_idx, vec_col_idx[nm_row_idx, nm_col_idx]
        mask[np_row_idx, np_col_idx] = 1
        
        weight = weight.numpy().astype(dtype)
        comp_weight = weight[np_row_idx, np_col_idx]
        
        pruned_weight = weight * mask
        
        return comp_weight, pruned_weight
    
    def comp_vec_weight(self, weight, vec_mask, M, K):
        row_idx, col_idx = self.mask_to_idx(vec_mask, M, K)
        return weight[row_idx, col_idx]
    
    def comp_nm_weight(self, weight, mask, M, K, dtype):
        row_idx, col_idx = self.mask_to_idx(mask, M, K//2)
        
        weight = weight.numpy().astype(dtype)
        comp_weight = weight[row_idx, col_idx]
        
        mask = mask.numpy().astype(dtype)
        pruned_weight = weight * mask
        
        return comp_weight, pruned_weight
    
    def gen_hier_nm_data(self, weight, input, tile_M, tile_K, vec_sparsity, dtype):
        M = weight.size(0)
        K = weight.size(1)
        N = input.size(1)
        
        vec_size = tile_M
        num_vec_grps = int(M // vec_size)
        nnz_vec = int((K // tile_K) * (1 - vec_sparsity) * tile_K)
        
        vec_mask = self.gen_vec_mask(weight, M, K, vec_size, num_vec_grps, nnz_vec)
        vec_meta = self.gen_vec_metadata(vec_mask, M, K, vec_size, num_vec_grps, nnz_vec)
        comp_vec_weight = self.comp_vec_weight(weight, vec_mask, M, nnz_vec)

        nm_mask = self.gen_nm_mask(comp_vec_weight, M, nnz_vec)
        bin_meta = self.gen_nm_metadata(nm_mask, M, nnz_vec, tile_M, tile_K)
        comp_weight, pruned_weight = self.comp_hier_weight(weight, vec_mask, nm_mask, M, K, nnz_vec, dtype)
        
        input = input.numpy().astype(dtype)
        accum = np.zeros((M, N), dtype)
        output = np.matmul(pruned_weight, input) + accum
        
        self.save_bin_files("bin_data/hier_nm_data/", comp_weight, pruned_weight, input, accum, output, bin_meta, vec_meta)
        
    def gen_nm_data(self, weight, input, tile_M, tile_K, dtype):
        M = weight.size(0)
        K = weight.size(1)
        N = input.size(1)
        
        mask = self.gen_nm_mask(weight, M, K)
        bin_meta = self.gen_nm_metadata(mask, M, K, tile_M, tile_K)
        comp_weight, pruned_weight = self.comp_nm_weight(weight, mask, M, K, dtype)
        
        input = input.numpy().astype(dtype)
        accum = np.zeros((M, N), dtype)
        output = np.matmul(pruned_weight, input) + accum
        
        self.save_bin_files("bin_data/nm_data/", comp_weight, pruned_weight, input, accum, output, bin_meta)
        
    def save_bin_files(self, dir, comp_weight, pruned_weight, input, accum, output, bin_meta, vec_meta = None):
        if self.transpose:
            input = input.transpose()

        if self.op == "cublas":
            input = input.transpose()
            pruned_weight = pruned_weight.transpose()
            
        comp_weight.tofile(dir + "a.bin")
        pruned_weight.tofile(dir + "den_a.bin")    
        input.tofile(dir + "b.bin")
        accum.tofile(dir + "c.bin")
        output.tofile(dir + "d.bin")
        bin_meta.tofile(dir + "metadata.bin")
        
        if vec_meta is not None:
            vec_meta.tofile(dir + "vec_metadata.bin")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=256)
    parser.add_argument("-n", type=int, default=256)
    parser.add_argument("-k", type=int, default=256)
    parser.add_argument("--dtype", default="float16", required=False)
    parser.add_argument("--op-type", type=str, default="hier_nm", choices=["nm", "hier_nm", "cublas", "dense"])
    parser.add_argument("--trans-input", action="store_true", default=False)
    parser.add_argument("--vec-sparsity", type=float, default=0.5)
    args = parser.parse_args()
    
    set_seed(7)
    sp_data = NMData(M=args.m, N=args.n, K=args.k, dtype=args.dtype, op_type=args.op_type, vec_sparsity=args.vec_sparsity, transpose_inp=args.trans_input)
