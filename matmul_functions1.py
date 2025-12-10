import numpy as np
import time
from numba import njit
from numba.cuda import cuda
import numba.cuda as cuda_module


def matmul_transpose_trivial(X):
    """
    Calculates X · X^T using 3 nested for loops (trivial implementation).
    
    Args:
        X: numpy array of shape (m, n)
    
    Returns:
        result: numpy array of shape (m, m) representing X · X^T
    """
    m, n = X.shape
    result = np.zeros((m, m), dtype=X.dtype)
    
    for i in range(m):
        for j in range(m):
            for k in range(n):
                result[i, j] += X[i, k] * X[j, k]
    
    return result


@njit
def matmul_transpose_numba(X):
    """
    Calculates X · X^T using NJIT (Numba JIT compilation).
    This is the same algorithm as trivial but accelerated by Numba.
    
    Args:
        X: numpy array of shape (m, n)
    
    Returns:
        result: numpy array of shape (m, m) representing X · X^T
    """
    m, n = X.shape
    result = np.zeros((m, m), dtype=X.dtype)
    
    for i in range(m):
        for j in range(m):
            for k in range(n):
                result[i, j] += X[i, k] * X[j, k]
    
    return result


@cuda_module.jit
def matmul_kernel(X, X_T, result):
    """
    CUDA kernel to compute X · X^T on the GPU.
    
    Args:
        X: input matrix (m, n) on GPU
        X_T: transposed matrix (n, m) on GPU
        result: output matrix (m, m) on GPU
    
    Each thread computes one element of the result matrix.
    """
    idx = cuda_module.grid(1)  # Get thread index in 1D grid
    
    m, n = X.shape
    total_elements = m * m
    
    # Process one element per thread (grid-stride loop if needed)
    if idx < total_elements:
        i = idx // m
        j = idx % m
        
        # Compute result[i, j] = sum of X[i, k] * X_T[k, i] = X[i, k] * X[j, k]
        sum_val = 0.0
        for k in range(n):
            sum_val += X[i, k] * X_T[k, j]
        
        result[i, j] = sum_val


def matmul_transpose_gpu(X):
    """
    Calculates X · X^T on the GPU using CUDA.
    
    Args:
        X: numpy array of shape (m, n)
    
    Returns:
        result: numpy array of shape (m, m) representing X · X^T
    """
    m, n = X.shape
    
    # Copy data to GPU
    X_gpu = cuda_module.to_device(X.astype(np.float32))
    X_T_gpu = cuda_module.to_device(X.T.astype(np.float32))
    result_gpu = cuda_module.device_array((m, m), dtype=np.float32)
    
    # Launch kernel with 1 block and 1024 threads
    threads_per_block = 1024
    blocks = 1
    matmul_kernel[blocks, threads_per_block](X_gpu, X_T_gpu, result_gpu)
    
    # Copy result back to CPU
    result = result_gpu.copy_to_host()
    
    return result


def benchmark_functions():
    """
    Benchmark all three implementations and compare run-times.
    """
    # Test with different matrix sizes
    test_sizes = [
        (100, 50),
        (500, 250),
        (1000, 500),
    ]
    
    print("=" * 80)
    print("Matrix Multiplication Benchmark: X · X^T")
    print("=" * 80)
    
    for m, n in test_sizes:
        print(f"\nMatrix size: {m} × {n} (result: {m} × {m})")
        print("-" * 80)
        
        # Generate random test matrix
        X = np.random.randn(m, n).astype(np.float32)
        
        # Trivial implementation
        try:
            start = time.time()
            result_trivial = matmul_transpose_trivial(X)
            time_trivial = time.time() - start
            print(f"Trivial (3 nested loops):      {time_trivial:.4f} seconds")
        except Exception as e:
            print(f"Trivial (3 nested loops):      ERROR - {e}")
            time_trivial = float('inf')
        
        # Numba JIT implementation
        try:
            # Warm-up run for Numba
            _ = matmul_transpose_numba(X)
            
            start = time.time()
            result_numba = matmul_transpose_numba(X)
            time_numba = time.time() - start
            print(f"Numba JIT:                     {time_numba:.4f} seconds")
            speedup_numba = time_trivial / time_numba if time_numba > 0 else 0
            print(f"  → Speedup vs trivial:        {speedup_numba:.2f}x")
        except Exception as e:
            print(f"Numba JIT:                     ERROR - {e}")
            time_numba = float('inf')
        
        # GPU implementation (if CUDA available)
        try:
            start = time.time()
            result_gpu = matmul_transpose_gpu(X)
            time_gpu = time.time() - start
            print(f"GPU (CUDA):                    {time_gpu:.4f} seconds")
            speedup_gpu = time_trivial / time_gpu if time_gpu > 0 else 0
            print(f"  → Speedup vs trivial:        {speedup_gpu:.2f}x")
            
            # Verify GPU result matches trivial (within floating point tolerance)
            if np.allclose(result_trivial, result_gpu, rtol=1e-3):
                print(f"  ✓ GPU result validated")
            else:
                print(f"  ✗ GPU result differs from trivial")
        except Exception as e:
            print(f"GPU (CUDA):                    ERROR - {e}")
            print(f"  (Make sure CUDA is installed and configured)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_functions()
