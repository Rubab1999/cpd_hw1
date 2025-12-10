import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    n, m = X.shape
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(m):
                result[i, j] += X[i, k] * X[j, k]
    return result


@njit(parallel=True)
def matmul_transpose_numba(X):
    n, m = X.shape
    result = np.zeros((n, n))
    for i in prange(n):
        for j in range(n):
            for k in range(m):
                result[i, j] += X[i, k] * X[j, k]
    return result


def matmul_transpose_gpu(X):
    n, m = X.shape
    
    # Allocate device memory
    d_X = cuda.to_device(X)
    d_result = cuda.device_array((n, n))
    
    # Launch kernel with 1 block of 1024 threads
    threads_per_block = 1024
    blocks = 1
    
    matmul_kernel[blocks, threads_per_block](d_X, d_result)
    
    # Copy result back to host
    result = d_result.copy_to_host()
    return result


@cuda.jit
def matmul_kernel(A, C):
    n, m = A.shape
    total_elements = n * n
    
    # Global thread ID
    thread_id = cuda.threadIdx.x
    threads_per_block = cuda.blockDim.x
    
    # Each thread processes multiple elements
    for idx in range(thread_id, total_elements, threads_per_block):
        i = idx // n
        j = idx % n
        
        sum_val = 0.0
        for k in range(m):
            sum_val += A[i, k] * A[j, k]
        
        C[i, j] = sum_val


def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_trivial failed')
        exit(0)
    else:
        print('[+] matmul_transpose_trivial passed')

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_numba failed')
        exit(0)
    else:
        print('[+] matmul_transpose_numba passed')

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_gpu failed')
        exit(0)
    else:
        print('[+] matmul_transpose_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    verify_solution()
    matmul_comparison()
