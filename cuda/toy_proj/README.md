# TOY Project of Hierarchical N:M sparsity

# How to run toy project
## Step 1: Generate Binary Data
```bash
mkdir bin_data && mkdir bin_data/hinm_data && mkdir bin_data/nm_data
python3 gen_data.py -m <spatial dim of weight> -n <spatial dim of input> -k <reduce dim> --op-type <N:M or HiNM>
```

## Step 2: Compile CUDA
```bash
/usr/local/cuda-<version>/bin/nvcc -arch <architecture> cuda_hinm_test.cu
```
