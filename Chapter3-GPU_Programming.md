# GPU Programming Concepts

## Table of Contents
1. [Introduction to GPU Computing](#introduction-to-gpu-computing)
2. [GPU Architecture](#gpu-architecture)
3. [Programming Models](#programming-models)
4. [Memory Hierarchy](#memory-hierarchy)
5. [Thread Organization](#thread-organization)
6. [Execution Model](#execution-model)
7. [Performance Optimization](#performance-optimization)
8. [Common Patterns](#common-patterns)
9. [Debugging and Profiling](#debugging-and-profiling)
10. [Advanced Concepts](#advanced-concepts)

---

## Introduction to GPU Computing

### What is GPU Computing?
GPU (Graphics Processing Unit) computing leverages the parallel processing power of GPUs for general-purpose computing tasks beyond graphics rendering. Modern GPUs contain thousands of cores designed for executing many threads simultaneously.

### CPU vs GPU Architecture

| Aspect | CPU | GPU |
|--------|-----|-----|
| Design Philosophy | Few powerful cores | Many simpler cores |
| Thread Count | 4-32 threads | Thousands of threads |
| Memory | Large cache, complex hierarchy | High bandwidth, simpler hierarchy |
| Control Flow | Complex branch prediction | SIMT (Single Instruction, Multiple Thread) |
| Best For | Sequential tasks, complex logic | Parallel tasks, data processing |

### When to Use GPUs
**Ideal Workloads:**
- Highly parallel computations
- Data-intensive operations
- Mathematical computations (linear algebra, FFT)
- Image/signal processing
- Machine learning and AI
- Scientific simulations

**Not Ideal For:**
- Sequential algorithms
- Heavy branching logic
- Small datasets
- Frequent host-device communication

---

## GPU Architecture

### Modern GPU Structure
```
GPU Device
├── Streaming Multiprocessors (SMs)
│   ├── CUDA Cores / Stream Processors
│   ├── Shared Memory
│   ├── Register File
│   ├── L1 Cache
│   └── Special Function Units
├── L2 Cache
├── Global Memory (DRAM)
└── Memory Controllers
```

### Streaming Multiprocessor (SM)
The fundamental execution unit of a GPU:
- **CUDA Cores**: Basic arithmetic units
- **Warp Schedulers**: Manage thread execution
- **Shared Memory**: Fast, user-controlled cache
- **Register File**: Very fast thread-local storage

### Memory Types

| Memory Type | Size | Latency | Bandwidth | Scope |
|-------------|------|---------|-----------|-------|
| Registers | ~64KB per SM | 1 cycle | Highest | Thread |
| Shared Memory | 48-164KB per SM | ~20 cycles | Very High | Thread Block |
| L1 Cache | 32-128KB per SM | ~28 cycles | High | SM |
| L2 Cache | 1-6MB | ~200 cycles | Medium | Device |
| Global Memory | GBs | 400-800 cycles | Lower | Device |

---

## Programming Models

### CUDA Programming Model

#### Basic Concepts
- **Host**: CPU and its memory
- **Device**: GPU and its memory
- **Kernel**: Function that runs on GPU
- **Thread**: Basic execution unit
- **Block**: Group of threads
- **Grid**: Collection of blocks

#### Simple CUDA Program Structure
```cuda
// Host code
int main() {
    // 1. Allocate host memory
    float *h_data = (float*)malloc(size);
    
    // 2. Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // 3. Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 4. Launch kernel
    kernel<<<blocks, threads>>>(d_data);
    
    // 5. Copy results back
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // 6. Cleanup
    cudaFree(d_data);
    free(h_data);
    return 0;
}

// Device code
__global__ void kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Process data[idx]
}
```

### OpenCL Programming Model
Cross-platform parallel computing framework:
- **Platform**: Hardware + drivers
- **Context**: Environment for kernels
- **Command Queue**: Manages kernel execution
- **Work-item**: Individual thread
- **Work-group**: Group of work-items

### SYCL Programming Model
Modern C++ abstraction for heterogeneous computing:
```cpp
#include <sycl/sycl.hpp>

void vector_add(sycl::queue& q, float* a, float* b, float* c, size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    });
}
```

---

## Memory Hierarchy

### Global Memory
- **Largest but slowest** memory space
- **Accessible by all threads**
- **Coalesced access** crucial for performance
- **Typical size**: 4-80GB on modern GPUs

#### Memory Coalescing
```cuda
// Good: Coalesced access
__global__ void coalesced_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // Sequential access
}

// Bad: Strided access
__global__ void strided_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx * 32] = data[idx * 32] * 2.0f;  // Large stride
}
```

### Shared Memory
- **Fast, user-controlled cache**
- **Shared within thread block**
- **Enables cooperation between threads**

#### Usage Example
```cuda
__global__ void matrix_multiply_shared(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    float sum = 0.0f;
    
    for (int k = 0; k < N / TILE_SIZE; k++) {
        // Load tiles into shared memory
        As[ty][tx] = A[(by * TILE_SIZE + ty) * N + k * TILE_SIZE + tx];
        Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + bx * TILE_SIZE + tx];
        
        __syncthreads();
        
        // Compute using shared memory
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    C[(by * TILE_SIZE + ty) * N + bx * TILE_SIZE + tx] = sum;
}
```

### Constant Memory
- **Read-only memory**
- **Cached and broadcast**
- **Good for small, frequently accessed data**

```cuda
__constant__ float coefficients[256];

__global__ void use_constant_memory(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * coefficients[idx % 256];
    }
}
```

### Texture Memory
- **Optimized for 2D spatial locality**
- **Hardware interpolation**
- **Cached reads**

---

## Thread Organization

### Thread Hierarchy
```
Grid
├── Block (0,0)     ├── Block (1,0)     ├── Block (2,0)
│   ├── Thread(0,0) │   ├── Thread(0,0) │   ├── Thread(0,0)
│   ├── Thread(1,0) │   ├── Thread(1,0) │   ├── Thread(1,0)
│   ├── Thread(0,1) │   ├── Thread(0,1) │   ├── Thread(0,1)
│   └── Thread(1,1) │   └── Thread(1,1) │   └── Thread(1,1)
```

### Thread Indexing
```cuda
// 1D Grid and Block
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D Grid and Block
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int globalIdx = y * gridDim.x * blockDim.x + x;

// 3D indexing
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### Choosing Block Size
**Guidelines:**
- **Multiple of 32** (warp size)
- **Powers of 2** often work well
- **Common sizes**: 128, 256, 512, 1024
- **Consider register usage** and shared memory

```cuda
// Launch configuration examples
dim3 blockSize(256);
dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
kernel<<<gridSize, blockSize>>>(data, n);

// 2D configuration
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);
kernel2D<<<gridSize, blockSize>>>(data, width, height);
```

---

## Execution Model

### SIMT (Single Instruction, Multiple Thread)
- **Threads execute in groups of 32** (warps)
- **All threads in warp execute same instruction**
- **Branch divergence** reduces efficiency

### Warp Execution
```cuda
// Efficient - no divergence
__global__ void no_divergence(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2;  // All threads execute same operation
    }
}

// Inefficient - branch divergence
__global__ void with_divergence(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = data[idx] * 2;     // Half warp executes this
    } else {
        data[idx] = data[idx] + 1;     // Other half executes this
    }
}
```

### Synchronization
```cuda
// Block-level synchronization
__global__ void synchronize_example() {
    __shared__ float sdata[256];
    
    // Phase 1: Load data
    sdata[threadIdx.x] = /* load from global memory */;
    
    __syncthreads();  // Wait for all threads in block
    
    // Phase 2: Process shared data
    /* All threads can safely access sdata */
}

// Warp-level primitives
__global__ void warp_primitives() {
    int value = threadIdx.x;
    
    // Warp shuffle
    int neighbor = __shfl_down_sync(0xffffffff, value, 1);
    
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
}
```

---

## Performance Optimization

### Memory Optimization
1. **Coalesced Memory Access**
   - Consecutive threads access consecutive memory
   - Avoid strided access patterns

2. **Shared Memory Usage**
   - Reduce global memory accesses
   - Watch for bank conflicts

3. **Memory Bandwidth**
   - Achieve high arithmetic intensity
   - Use appropriate data types

### Compute Optimization
1. **Occupancy**
   - Balance threads per block vs. register usage
   - Use occupancy calculator

2. **Arithmetic Intensity**
   - Maximize computation per memory access
   - Use math libraries (cuBLAS, cuFFT)

3. **Branch Divergence**
   - Minimize conditional statements
   - Use predication when possible

### Occupancy Optimization
```cuda
// Check register usage
__global__ void __launch_bounds__(256, 4)  // Max 256 threads, min 4 blocks per SM
high_register_kernel() {
    // Kernel with high register usage
}

// Use shared memory efficiently
__global__ void optimize_shared_memory() {
    __shared__ float sdata[256];  // Static allocation
    
    // Avoid bank conflicts
    int tid = threadIdx.x;
    sdata[tid] = /* load data */;  // No conflicts
    
    // Padding to avoid conflicts in 2D access
    __shared__ float matrix[32][33];  // 33 instead of 32 to avoid conflicts
}
```

---

## Common Patterns

### Element-wise Operations
```cuda
template<typename T>
__global__ void elementwise_add(T* a, T* b, T* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Reduction Pattern
```cuda
__global__ void reduction_sum(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Perform reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Scan/Prefix Sum
```cuda
__global__ void scan_exclusive(int* input, int* output, int n) {
    __shared__ int temp[256];
    
    int tid = threadIdx.x;
    temp[tid] = (tid > 0) ? input[tid - 1] : 0;
    __syncthreads();
    
    // Up-sweep phase
    for (int d = 1; d < blockDim.x; d *= 2) {
        int index = (tid + 1) * 2 * d - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - d];
        }
        __syncthreads();
    }
    
    // Clear last element
    if (tid == 0) temp[blockDim.x - 1] = 0;
    __syncthreads();
    
    // Down-sweep phase
    for (int d = blockDim.x / 2; d > 0; d /= 2) {
        int index = (tid + 1) * 2 * d - 1;
        if (index < blockDim.x) {
            int t = temp[index - d];
            temp[index - d] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }
    
    output[tid] = temp[tid];
}
```

### Matrix Operations
```cuda
// Matrix transpose with shared memory
__global__ void transpose(float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Read from input
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Write to output (transposed)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

---

## Debugging and Profiling

### Debugging Tools
1. **CUDA-GDB**: Command-line debugger
2. **Nsight Compute**: Kernel profiler
3. **Nsight Systems**: System-wide profiler
4. **cuda-memcheck**: Memory error detector

### Common Debugging Techniques
```cuda
// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

// Debug prints in kernel
__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {  // Only first thread prints
        printf("Block %d, Thread %d: Processing %d elements\n", 
               blockIdx.x, threadIdx.x, n);
    }
}
```

### Performance Profiling
```bash
# Profile with Nsight Compute
ncu --set full --target-processes all ./my_cuda_program

# Profile with Nsight Systems
nsys profile --trace=cuda,nvtx ./my_cuda_program

# Basic timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<blocks, threads>>>(data);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

---

## Advanced Concepts

### Dynamic Parallelism
```cuda
__global__ void parent_kernel(int* data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (depth > 0 && idx < n) {
        // Launch child kernel from device
        child_kernel<<<1, 32>>>(data + idx, n - idx, depth - 1);
        cudaDeviceSynchronize();  // Wait for child completion
    }
}
```

### Unified Memory
```cuda
int *data;
// Allocate unified memory accessible from CPU and GPU
cudaMallocManaged(&data, size);

// Use on CPU
for (int i = 0; i < n; i++) {
    data[i] = i;
}

// Use on GPU
kernel<<<blocks, threads>>>(data, n);
cudaDeviceSynchronize();

// Use on CPU again
for (int i = 0; i < n; i++) {
    printf("%d ", data[i]);
}

cudaFree(data);
```

### Multi-GPU Programming
```cuda
void multi_gpu_compute(float* data, int n, int num_gpus) {
    int chunk_size = n / num_gpus;
    
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        
        float* d_data;
        int offset = gpu * chunk_size;
        int size = (gpu == num_gpus - 1) ? n - offset : chunk_size;
        
        cudaMalloc(&d_data, size * sizeof(float));
        cudaMemcpy(d_data, data + offset, size * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        // Launch kernel on each GPU
        kernel<<<(size + 255) / 256, 256>>>(d_data, size);
        
        cudaMemcpy(data + offset, d_data, size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}
```

### Cooperative Groups
```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void cooperative_kernel(int* data) {
    // Grid-wide cooperation
    grid_group grid = this_grid();
    
    // Block-wide cooperation  
    thread_block block = this_thread_block();
    
    // Warp-level cooperation
    coalesced_group warp = coalesced_threads();
    
    // Synchronize across grid
    grid.sync();
}
```

### Tensor Cores (Ampere/Hopper)
```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_gemm() {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Load matrices
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Perform mixed-precision matrix multiply-accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

---

## Best Practices Summary

### Memory Management
- Always check for CUDA errors
- Free allocated memory
- Use appropriate memory types
- Minimize host-device transfers

### Kernel Design
- Choose appropriate block sizes
- Minimize branch divergence
- Maximize occupancy
- Use shared memory effectively

### Performance
- Profile your code
- Optimize memory access patterns
- Use cuBLAS/cuFFT for standard operations
- Consider arithmetic intensity

### Code Organization
- Separate host and device code clearly
- Use templates for generic kernels
- Handle edge cases properly
- Document performance assumptions

This guide provides a comprehensive foundation for GPU programming concepts, from basic architecture understanding to advanced optimization techniques.
