import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=1024)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("--dtype", default="float16", required=False)
    parser.add_argument("--sp-type", type=str, default="nm", choices=["nm", "hier_nm"])
    args = parser.parse_args()
    
    
    if args.sp_type == "nm":
        cpu_output = np.fromfile('bin_data/nm_data/d.bin', dtype=args.dtype)
    elif args.sp_type == "hier_nm":
        cpu_output = np.fromfile("bin_data/hier_nm_data/d.bin", dtype=args.dtype)
        
    gpu_output = np.fromfile('bin_data/device/d_gpu.bin', dtype=args.dtype)
    print("========== CPU output =========")
    print(cpu_output.reshape(args.m, args.n))
    print("========== GPU output =========")
    print(gpu_output.reshape(args.m, args.n))
    diff = np.abs(cpu_output - gpu_output).reshape(args.m, args.n)
    import pdb; pdb.set_trace()
    print('diff: {}'.format(diff))