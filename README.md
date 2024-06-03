# Toward Efficient Permutation for Hierarchical N:M sparsity on GPUs

## Introduction
This project introduces a novel permutation technique for hierarchical N:M(HiNM) sparsity on GPUs, termed "Gyro-Permutation." This technique addresses several key aspects critical for efficient sparsity implementation in deep learning models.

## Contributions
Our contributions with the "Gyro-Permutation" technique include solutions to the following key aspects:
1. **Hierarchical Pruning Awareness** - Ensuring the pruning process is aware of the hierarchical structure of neural networks.
2. **Consistency across Layers** - Maintaining consistency in sparsity patterns across different layers of the network.
3. **Escaping Local Minima** - We observed that existing permutation techniques can easily fall into local minima. Addressing this challenge is essential for enhancing the performance of HiNM pruning

## Getting Started

### Prerequisites
To use this project, you will need:
- Python 3.8 or newer
- PyTorch 1.7 or newer
- CUDA-compatible GPUs

### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/yourgithubusername/gyro-permutation.git
cd gyro-permutation
pip install -r requirements.txt
