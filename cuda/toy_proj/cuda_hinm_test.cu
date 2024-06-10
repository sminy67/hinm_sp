#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <vector>

#define gpuErrchk(ans)                                                         \
	{ gpuAssert((ans), __FILE__, __LINE__); }

#define CEIL(x, y) (((x) + (y) -1)/(y))
#define USE_F32

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

const static uint32_t num_warmup = 10;
const static uint32_t num_iters = 100;

const static uint32_t global_M = 768;
const static uint32_t global_N = 768;
const static uint32_t global_K = 4096;
const static uint32_t global_CompK = global_K / 2;

const static uint32_t BlockTile_M = 128;
const static uint32_t BlockTile_N = 64;
const static uint32_t BlockTile_K = 64;
const static uint32_t BlockTile_CompK = BlockTile_K / 2;
const static uint32_t BlockTile_Meta = BlockTile_M * BlockTile_K / 32;

const static uint32_t NumBlockTile_M = (global_M / BlockTile_M);
const static uint32_t NumBlockTile_N = (global_N / BlockTile_N);
const static uint32_t NumBlockTile_K = (global_K / BlockTile_K);

const static uint32_t WarpTile_M = 32;
const static uint32_t WarpTile_N = 32;

const static uint32_t MMATile_M = 16;
const static uint32_t MMATile_N = 8;
const static uint32_t MMATile_K = 32;
const static uint32_t MMATile_Meta = MMATile_M * MMATile_K / 32;

const static uint32_t NumWarpPerBlock_M = (BlockTile_M / WarpTile_M);
const static uint32_t NumWarpPerBlock_N = (BlockTile_N / WarpTile_N);
const static uint32_t NumWarpPerBlock = (NumWarpPerBlock_M * NumWarpPerBlock_N);
const static uint32_t NumThreadPerBlock = NumWarpPerBlock * 32;

const static uint32_t WarpTile_Load_Bytes = 32 * sizeof(int4);
const static uint32_t BlockTile_A_Line_Bytes = BlockTile_CompK * sizeof(__half);
const static uint32_t BlockTile_A_Lines_Per_Warp = (WarpTile_Load_Bytes / BlockTile_A_Line_Bytes);
const static uint32_t Lanes_Per_A_Line = (32 / BlockTile_A_Lines_Per_Warp);

const static uint32_t BlockTile_B_Line_Bytes = BlockTile_N * sizeof(__half);
const static uint32_t BlockTile_B_Lines_Per_Warp = (WarpTile_Load_Bytes / BlockTile_B_Line_Bytes);
const static uint32_t Lanes_Per_B_Line = (32 / BlockTile_B_Lines_Per_Warp);

const static size_t stages_count = 2;

using namespace nvcuda;
using namespace cooperative_groups;

__device__ __forceinline__ int swizzle(int offset) {
    return (offset ^ ((offset & (7<<6))>>3));
    //return (offset ^ (((offset >> 6) & 7) << 3));
}

__device__ __forceinline__ void lds_matrix_sync(uint* __restrict__ a, const __half *base, const int offset) {
    *( (float4*) a ) = *( (float4*) (base + swizzle(offset)) );
}

__global__ void sp_mmad_shared_16832_test(__half *a, __half *b, __half *d, uint32_t *metadata) {

    const uint32_t bx = blockIdx.x;
    const uint32_t by = blockIdx.y;
    const uint32_t bid = gridDim.x * by + bx;

    const uint32_t laneId = threadIdx.x;
    const uint32_t warpId = threadIdx.y;

    const uint32_t m_warpId = warpId / NumWarpPerBlock_N;
    const uint32_t n_warpId = warpId % NumWarpPerBlock_N;

    int tid = laneId + warpId * 32;

    __shared__ __half A_shared[stages_count][BlockTile_M * BlockTile_CompK];
    __shared__ __half B_shared[stages_count][BlockTile_K * BlockTile_N];

    __shared__ uint32_t meta_shared[stages_count][BlockTile_Meta];

    int A_warp_panel_offset = m_warpId * WarpTile_M * BlockTile_CompK;
    const int offset_A = A_warp_panel_offset + laneId * 8;


#ifdef USE_F32
    float fragmentD[(WarpTile_M / MMATile_M)][(WarpTile_N / MMATile_N)][4] = {};
#else
    __half fragmentD[(WarpTile_M / MMATile_M)][(WarpTile_N / MMATile_N)][4] = {};
#endif

    if (bx >= NumBlockTile_N && by >= NumBlockTile_M) {
        return;
    }
    const uint32_t num_iters_ld_A = CEIL(BlockTile_M * BlockTile_CompK / 8, NumThreadPerBlock);
    const uint32_t num_iters_ld_B = CEIL(BlockTile_N * BlockTile_K / 8, NumThreadPerBlock);

    {
        const __half *tile_A = a + by * BlockTile_M * global_CompK;
        #pragma unroll
        for (int i = 0; i < num_iters_ld_A; i++) {
            int idx = (tid + i * NumThreadPerBlock) * 8;
            if (idx < BlockTile_M * BlockTile_CompK) {
                int m = idx / BlockTile_CompK;
                int k = idx % BlockTile_CompK;
                const __half *src = tile_A + m * global_CompK + k;
                __half *dst = A_shared[0] + swizzle(idx);
                //*(int4*)dst = *(const int4*)src;
                __pipeline_memcpy_async(dst, src, sizeof(int4));
            }
        }

        const __half *tile_B = b + bx * BlockTile_N;
        #pragma unroll
        for (int i = 0; i < num_iters_ld_B; i++) {
            int idx = (tid + i * NumThreadPerBlock) * 8;
            if (idx < BlockTile_K * BlockTile_N) {
                int k = idx / BlockTile_N;
                int n = idx % BlockTile_N;
                const __half *src = tile_B + k * global_N + n;
                __half *dst = B_shared[0] + swizzle(idx);
                //*(int4*)dst = *(const int4*)src;
                __pipeline_memcpy_async(dst, src, sizeof(int4));
            }
        }

        for (int i = 0; i < CEIL(BlockTile_Meta, NumThreadPerBlock); i++) {
            int idx = tid + i * NumThreadPerBlock;
            if (idx < BlockTile_Meta)
                //meta_shared[idx] = metadata[((by * NumBlockTile_K + blk_k_idx) * BlockTile_Meta) + idx];
                __pipeline_memcpy_async(meta_shared[0] + idx, metadata + (by * NumBlockTile_K) * BlockTile_Meta + idx, sizeof(uint32_t));
        }
        __pipeline_commit();
    }

    for (int blk_k_idx = 0; blk_k_idx < NumBlockTile_K; blk_k_idx++) {
        
        if (blk_k_idx < NumBlockTile_K - 1) {
            const __half *tile_A = a + by * BlockTile_M * global_CompK + (blk_k_idx+1) * BlockTile_CompK;
            #pragma unroll
            for (int i = 0; i < num_iters_ld_A; i++) {
                int idx = (tid + i * NumThreadPerBlock) * 8;
                if (idx < BlockTile_M * BlockTile_CompK) {
                    int m = idx / BlockTile_CompK;
                    int k = idx % BlockTile_CompK;
                    const __half *src = tile_A + m * global_CompK + k;
                    __half *dst = A_shared[(blk_k_idx+1)%stages_count] + swizzle(idx);
                    //*(int4*)dst = *(const int4*)src;
                    __pipeline_memcpy_async(dst, src, sizeof(int4));
                }
            }

            const __half *tile_B = b + (blk_k_idx+1) * BlockTile_K * global_N + bx * BlockTile_N;
            #pragma unroll
            for (int i = 0; i < num_iters_ld_B; i++) {
                int idx = (tid + i * NumThreadPerBlock) * 8;
                if (idx < BlockTile_K * BlockTile_N) {
                    int k = idx / BlockTile_N;
                    int n = idx % BlockTile_N;
                    const __half *src = tile_B + k * global_N + n;
                    __half *dst = B_shared[(blk_k_idx+1)%stages_count] + swizzle(idx);
                    //*(int4*)dst = *(const int4*)src;
                    __pipeline_memcpy_async(dst, src, sizeof(int4));
                }
            }

            for (int i = 0; i < CEIL(BlockTile_Meta, NumThreadPerBlock); i++) {
                int idx = tid + i * NumThreadPerBlock;
                if (idx < BlockTile_Meta)
                    //meta_shared[idx] = metadata[((by * NumBlockTile_K + blk_k_idx) * BlockTile_Meta) + idx];
                    __pipeline_memcpy_async(meta_shared[(blk_k_idx+1)%stages_count] + idx, metadata + (by * NumBlockTile_K + (blk_k_idx+1)) * BlockTile_Meta + idx, sizeof(uint32_t));
            }
        }
        __pipeline_commit();

        __pipeline_wait_prior(stages_count-1);
        __syncthreads();

        // Shared Memory to Register
        for (int mma_k = 0; mma_k < (BlockTile_K/MMATile_K); mma_k++) {
            __half fragmentA[WarpTile_M/MMATile_M][8] = {};
            __half fragmentB[WarpTile_N/MMATile_N][8] = {};
            uint32_t fragmentMeta;

            #pragma unroll
            for (int mma_m = 0; mma_m < (WarpTile_M / MMATile_M); mma_m++) {
                uint32_t *A = reinterpret_cast<uint32_t *>(&fragmentA[mma_m][0]);
                size_t mma_m_A_lane = (m_warpId * WarpTile_M) + (mma_m * MMATile_M) + (laneId % 16);
                size_t mma_k_A_lane = (laneId / 16) * 8 + (mma_k * MMATile_K / 2);
                uint32_t tile_A_shared_ptr = static_cast<uint32_t>(__cvta_generic_to_shared((A_shared[blk_k_idx%stages_count] + swizzle(mma_m_A_lane * BlockTile_CompK + mma_k_A_lane))));
                //uint32_t tile_A_shared_ptr = static_cast<uint32_t>(__cvta_generic_to_shared((A_shared[blk_k_idx%stages_count] + swizzle(offset_A + (mma_m<<8)))));

                asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "r"(tile_A_shared_ptr));
                
                //lds_matrix_sync(A, A_shared[blk_k_idx%stages_count], offset_A+(mma_m<<8));

                fragmentMeta = meta_shared[blk_k_idx%stages_count][((m_warpId * (WarpTile_M/MMATile_M) + mma_m) * (BlockTile_K/MMATile_K) + mma_k) * MMATile_Meta + (laneId / 4) * 2 + (laneId % 4)];


                #pragma unroll
                for (int mma_n = 0; mma_n < (WarpTile_N / MMATile_N); mma_n++) {
                    uint32_t *B = reinterpret_cast<uint32_t *>(&fragmentB[mma_n][0]);
                    size_t mma_n_B_lane = (n_warpId * WarpTile_N) + (mma_n * MMATile_N);
                    size_t mma_k_B_lane = laneId + (mma_k * MMATile_K);
                    uint32_t tile_B_shared_ptr = static_cast<uint32_t>(__cvta_generic_to_shared((B_shared[blk_k_idx%stages_count] + swizzle(mma_k_B_lane * BlockTile_N + mma_n_B_lane))));

                    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(B[0]), "=r"(B[1]), "=r"(B[2]), "=r"(B[3]) : "r"(tile_B_shared_ptr));

#ifdef USE_F32
                    float *D = reinterpret_cast<float *>(fragmentD[mma_m][mma_n]);

					asm volatile (
                        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32\n\t"
                        "{%0, %1, %2, %3},\n"
                        "{%4, %5, %6, %7},\n"
                        "{%8, %9, %10, %11},\n"
                        "{%12, %13, %14, %15}, %16, 0x0;\n"
                        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]) "=f"(D[3])
                        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                        "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]), "r"(fragmentMeta)
                        );
#else
                    uint32_t *D = reinterpret_cast<uint32_t *>(fragmentD[mma_m][mma_n]);

					asm volatile (
                        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16\n\t"
                        "{%0, %1},\n"
                        "{%2, %3, %4, %5},\n"
                        "{%6, %7, %8, %9},\n"
                        "{%10, %11}, %12, 0x0;\n"
                        : "=r"(D[0]), "=r"(D[1])
                        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                        "r"(D[0]), "r"(D[1]), "r"(fragmentMeta)
                        );
#endif
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
	for (int mma_m = 0; mma_m < (WarpTile_M / MMATile_M); mma_m++) {
        #pragma unroll
	    for (int mma_n = 0; mma_n < (WarpTile_N / MMATile_N); mma_n++) {

		    size_t row_ptr = (BlockTile_M * by) + (m_warpId * WarpTile_M) + (MMATile_M * mma_m) + (laneId / 4);
			size_t col_ptr = (BlockTile_N * bx) + (n_warpId * WarpTile_N) + (MMATile_N * mma_n) + (laneId % 4) * 2;
#ifdef USE_F32
			float *D = reinterpret_cast<float *>(fragmentD[mma_m][mma_n]);

            d[row_ptr * global_N + col_ptr] = __float2half(D[0]);
            d[row_ptr * global_N + col_ptr + 1] = __float2half(D[1]);
            d[(row_ptr + 8) * global_N + col_ptr] = __float2half(D[2]);
            d[(row_ptr + 8) * global_N + col_ptr + 1] = __float2half(D[3]);
#else
			uint32_t *D = reinterpret_cast<uint32_t *>(fragmentD[mma_m][mma_n]);

			*((uint32_t *)&d[row_ptr * global_N + col_ptr]) = D[0];
			*((uint32_t *)&d[(row_ptr + 8) * global_N + col_ptr]) = D[1];
#endif
		}
	}
    
    return;
}

int main(int argc, char **argv) {
	size_t mat_a_size = global_M * global_CompK;
	size_t mat_b_size = global_N * global_K;
	size_t mat_d_size = global_M * global_N;
	size_t metadata_size_bytes = global_M * global_K / 8;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float t_ms;

	__half *mat_a_host = new __half[mat_a_size];
	__half *mat_b_host = new __half[mat_b_size];
	__half *mat_d_host = new __half[mat_d_size];

	uint32_t *metadata_host = new uint32_t[metadata_size_bytes / sizeof(uint32_t)];

	std::ifstream a_fs("./bin_data/nm_data/a.bin", std::ios_base::binary);
	a_fs.read((char *)mat_a_host, mat_a_size * sizeof(__half));
	std::ifstream b_fs("./bin_data/nm_data/b.bin", std::ios_base::binary);
	b_fs.read((char *)mat_b_host, mat_b_size * sizeof(__half));
	std::ifstream d_fs("./bin_data/nm_data/c.bin", std::ios_base::binary);
	d_fs.read((char *)mat_d_host, mat_d_size * sizeof(__half));
	std::ifstream m_fs("./bin_data/nm_data/metadata.bin", std::ios_base::binary);
	m_fs.read((char *)metadata_host, metadata_size_bytes);
  
	__half *mat_a_dev;
	__half *mat_b_dev;
	__half *mat_d_dev;

	 uint32_t *metadata_dev;

	gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__half)));
    gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__half)));
	gpuErrchk(cudaMalloc(&mat_d_dev, mat_d_size * sizeof(__half)));
	gpuErrchk(cudaMalloc(&metadata_dev, metadata_size_bytes));

	gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__half),
						 cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__half),
						 cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(mat_d_dev, mat_d_host, mat_d_size * sizeof(__half),
						 cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(metadata_dev, metadata_host, metadata_size_bytes,
						 cudaMemcpyHostToDevice));

	dim3 NUM_THREADS(32, NumWarpPerBlock);
	dim3 NUM_BLOCKS((global_N + BlockTile_N - 1) / BlockTile_N, (global_M + BlockTile_M - 1) / BlockTile_M);

	//const static size_t shared_bytes = ((BlockTile_M * BlockTile_K / 2) + (BlockTile_N * BlockTile_K)) * sizeof(__half) + BlockTile_Meta * sizeof(uint32_t);
	const static size_t shared_bytes = 0;


    sp_mmad_shared_16832_test<<<NUM_BLOCKS, NUM_THREADS, shared_bytes>>>(
        mat_a_dev, mat_b_dev, mat_d_dev, metadata_dev);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    for (int i = 0; i < num_warmup; i++) {
        sp_mmad_shared_16832_test<<<NUM_BLOCKS, NUM_THREADS, shared_bytes>>>(
            mat_a_dev, mat_b_dev, mat_d_dev, metadata_dev);
    }
	gpuErrchk(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        sp_mmad_shared_16832_test<<<NUM_BLOCKS, NUM_THREADS, shared_bytes>>>(
            mat_a_dev, mat_b_dev, mat_d_dev, metadata_dev);
    }
	gpuErrchk(cudaEventRecord(stop));
	gpuErrchk(cudaEventSynchronize(stop));

	gpuErrchk(cudaEventElapsedTime(&t_ms, start, stop));

	gpuErrchk(cudaMemcpy(mat_d_host, mat_d_dev, mat_d_size * sizeof(__half),
						 cudaMemcpyDeviceToHost));

	std::cout << "mma.sp.sync.aligned.m16n8k32 using shared mem latency: " << t_ms / num_iters * 1000 << " ns"  << std::endl;
	std::ofstream gpu_d_fs("./bin_data/device/d_gpu.bin", std::ios_base::binary);
	gpu_d_fs.write((char *)mat_d_host, mat_d_size * sizeof(__half));
  
	gpuErrchk(cudaFree(mat_a_dev));
	gpuErrchk(cudaFree(mat_b_dev));
	gpuErrchk(cudaFree(mat_d_dev));
  
	delete[] mat_a_host;
	delete[] mat_b_host;
	delete[] mat_d_host;
  
	return 0;	
	 
}
