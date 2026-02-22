# Mathematics Refresher for GPU Programming

## Table of Contents
1. [Linear Algebra Fundamentals](#linear-algebra-fundamentals)
2. [Matrix Operations](#matrix-operations)
3. [Memory Layout Mathematics](#memory-layout-mathematics)
4. [GEMM (General Matrix Multiply)](#gemm-general-matrix-multiply)
5. [Tensor Operations](#tensor-operations)
6. [Parallel Computing Math](#parallel-computing-math)
7. [GPU Architecture Mathematics](#gpu-architecture-mathematics)
8. [Optimization Mathematics](#optimization-mathematics)
9. [Common GPU Algorithms](#common-gpu-algorithms)
10. [Additional Advanced Concepts](#additional-advanced-concepts)
11. [Missing Areas for Complete Coverage](#missing-areas-for-complete-coverage)

---

## Linear Algebra Fundamentals

### Vectors
A vector is an ordered collection of numbers (scalars):

```math
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```

**Key Operations:**
- **Dot Product**: $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$
- **Vector Addition**: $\mathbf{c} = \mathbf{a} + \mathbf{b}$ where $c_i = a_i + b_i$
- **Scalar Multiplication**: $\mathbf{c} = k\mathbf{a}$ where $c_i = ka_i$

### Matrices
A matrix is a 2D array of numbers:

```math
\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
```

**Dimensions**: $m \times n$ (rows × columns)

---

## Matrix Operations

### Matrix Multiplication
For matrices 
```math
\mathbf{A}_{m \times k} \text{ and } \mathbf{B}_{k \times n}
```

```math
(\mathbf{AB})_{ij} = \sum_{l=1}^{k} a_{il} \cdot b_{lj}
```

**Requirements**: Inner dimensions must match ($k$)
**Result**: $\mathbf{C}_{m \times n}$

### Matrix Transpose

```math
(\mathbf{A}^T)_{ij} = a_{ji}
```

### Element-wise Operations
- **Hadamard Product**:
  ```math
  (\mathbf{A} \odot \mathbf{B})_{ij} = a_{ij} \cdot b_{ij}
  ```
- **Element-wise Addition**:
  ```math
  (\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}
  ```

---

## Memory Layout Mathematics

### Row-Major vs Column-Major
**Row-Major (C-style)**:
```
Index(i,j) = i × n + j
```
Where `n` is number of columns.

**Column-Major (Fortran-style)**:
```
Index(i,j) = j × m + i
```
Where `m` is number of rows.

### Memory Coalescing
For optimal memory access, consecutive threads should access consecutive memory locations:

```math
\text{Memory Address} = \text{Base} + \text{ThreadID} \times \text{ElementSize}
```

### Stride Calculations
For tensor with dimensions $(d_1, d_2, d_3, \ldots, d_n)$:

```math
\text{Stride}_i = \prod_{j=i+1}^{n} d_j
```

---

## GEMM (General Matrix Multiply)

### Standard GEMM Formula

```math
\mathbf{C} = \alpha \mathbf{AB} + \beta \mathbf{C}
```

Where:
- $\alpha$, $\beta$ are scalars

```math
  \mathbf{A}_{m \times k}, \mathbf{B}_{k \times n}, \mathbf{C}_{m \times n}
```
  are matrices

### Blocked GEMM
Divide matrices into blocks for cache efficiency:

```math
\mathbf{C}_{ij} = \sum_{k} \mathbf{A}_{ik} \mathbf{B}_{kj}
```

**Block size considerations**:
- L1 cache: typically 32KB-64KB
- Shared memory: 48KB-164KB per SM
- Register file: 64KB per SM

### GEMM Complexity
- **Time Complexity**: $O(mnk)$
- **Space Complexity**: $O(mn + mk + kn)$

---

## Tensor Operations

### Tensor Indexing
For a 4D tensor with dimensions $(N, C, H, W)$:

```math
\text{Index} = n \times (C \times H \times W) + c \times (H \times W) + h \times W + w
```

### Convolution Mathematics
2D Convolution:

```math
y[m,n] = \sum_{i=0}^{I-1} \sum_{j=0}^{J-1} x[m+i, n+j] \cdot h[i,j]
```

**Output size**: 

```math
\text{Output} = \left\lfloor \frac{\text{Input} + 2 \times \text{Padding} - \text{Kernel}}{\text{Stride}} \right\rfloor + 1
```

### Batch Operations
For batch size $B$ and operation $f$:

```math
\mathbf{Y} = \begin{bmatrix} f(\mathbf{X}_1) \\ f(\mathbf{X}_2) \\ \vdots \\ f(\mathbf{X}_B) \end{bmatrix}
```

---

## Parallel Computing Math

### Work Distribution
**Total Work**: $W$
**Number of Processors**: $P$
**Work per Processor**: $W_p = \lceil W/P \rceil$

### Thread Indexing
**1D Grid**:
```
globalID = blockIdx.x * blockDim.x + threadIdx.x
```

**2D Grid**:
```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

### Reduction Operations
**Parallel Sum Reduction**:
For array of size $n$, reduction takes $\log_2(n)$ steps.
Each step: $a[i] = a[i] + a[i + \text{stride}]$

### Memory Bandwidth Utilization

```math
\text{Efficiency} = \frac{\text{Useful Data Transferred}}{\text{Total Memory Bandwidth Used}}
```

---

## GPU Architecture Mathematics

### Occupancy Calculation

```math
\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}
```

**Limiting Factors**:
- Registers per thread: $\text{Max Threads} = \lfloor \frac{\text{Total Registers}}{\text{Registers per Thread}} \rfloor$
- Shared memory: $\text{Max Threads} = \lfloor \frac{\text{Total Shared Memory}}{\text{Shared Memory per Block}} \rfloor \times \text{Threads per Block}$

### Warp Efficiency

```math
\text{Warp Efficiency} = \frac{\text{Average Active Threads per Warp}}{32}
```

### Memory Throughput
**Theoretical Peak**:

```math
\text{Bandwidth} = \text{Memory Clock} \times \text{Bus Width} \times 2
```

**Achieved Bandwidth**:

```math
\text{Achieved} = \frac{\text{Bytes Transferred}}{\text{Time Elapsed}}
```

---

## Optimization Mathematics

### Arithmetic Intensity

```math
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}
```

### Roofline Model
Performance bound by:

```math
\text{Performance} = \min(\text{Peak FLOP/s}, \text{Arithmetic Intensity} \times \text{Memory Bandwidth})
```

### Amdahl's Law

```math
\text{Speedup} = \frac{1}{(1-P) + \frac{P}{N}}
```
Where:
- $P$ = fraction of code that can be parallelized
- $N$ = number of processors

### Cache Miss Penalty

```math
\text{Average Access Time} = \text{Hit Time} + \text{Miss Rate} \times \text{Miss Penalty}
```

### Compute Bound vs Memory Bound Analysis

**Compute Bound**: Performance limited by arithmetic operations

```math
\text{Compute Time} = \frac{\text{Total FLOPs}}{\text{Peak FLOP/s}}
```

**Memory Bound**: Performance limited by data movement

```math
\text{Memory Time} = \frac{\text{Bytes Transferred}}{\text{Memory Bandwidth}}
```

**Performance Bottleneck**:

```math
\text{Actual Time} = \max(\text{Compute Time}, \text{Memory Time})
```

**Operational Intensity Threshold**:

```math
\text{OI}_{\text{threshold}} = \frac{\text{Peak FLOP/s}}{\text{Peak Memory Bandwidth}}
```

- If $\text{OI} > \text{OI}_{\text{threshold}}$: **Compute Bound**
- If $\text{OI} < \text{OI}_{\text{threshold}}$: **Memory Bound**

**GEMM Example**:
For matrix sizes $M \times K \times N$:
- FLOPs: $2MKN$
- Memory: $M \times K + K \times N + M \times N$ elements
- Operational Intensity: $\frac{2MKN}{(MK + KN + MN) \times \text{sizeof(float)}}$

### Memory Hierarchy Latencies
**Typical GPU Memory Latencies**:
- L1 Cache: ~28 cycles
- L2 Cache: ~200 cycles  
- Global Memory: ~400-800 cycles
- Shared Memory: ~20 cycles
- Registers: 1 cycle

**Hide Latency Calculation**:

```math
\text{Threads Needed} = \frac{\text{Memory Latency}}{\text{Arithmetic Cycles per Thread}}
```

---

## Common GPU Algorithms

### Matrix Transpose
**Naive approach**: $O(mn)$ with poor coalescing
**Tiled approach**: Use shared memory tiles to improve coalescing

### Reduction
**Tree Reduction**: 
- Step $k$: $a[i] = a[i] + a[i + 2^k]$
- Total steps: $\log_2(n)$

### Prefix Sum (Scan)
**Up-sweep phase**:

```math
a[2^{k+1} \cdot i + 2^k - 1] += a[2^{k+1} \cdot i + 2^{k+1} - 1]
```

**Down-sweep phase**:

```math
a[2^{k+1} \cdot i + 2^{k+1} - 1] = a[2^{k+1} \cdot i + 2^k - 1] + a[2^{k+1} \cdot i + 2^{k+1} - 1]
```

### Softmax

```math
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}
```

**Numerically stable version**:
1. Find maximum: $m = \max(x)$
2. Compute: $e^{x_i - m}$
3. Normalize by sum

---

## Key Formulas Quick Reference

| Operation | Formula | Complexity |
|-----------|---------|------------|
| Matrix Multiply | $C_{ij} = \sum_k A_{ik} B_{kj}$ | $O(mnk)$ |
| Convolution | $y[m,n] = \sum_{i,j} x[m+i,n+j] h[i,j]$ | $O(NCHW \cdot KK)$ |
| Thread Index | `blockIdx.x * blockDim.x + threadIdx.x` | $O(1)$ |
| Reduction | Tree-based parallel sum | $O(\log n)$ |
| Transpose | $A^T[j][i] = A[i][j]$ | $O(mn)$ |

---

## Performance Tips

1. **Memory Coalescing**: Ensure consecutive threads access consecutive memory
2. **Shared Memory**: Use for data reuse within thread blocks
3. **Register Usage**: Minimize to maximize occupancy
4. **Warp Divergence**: Avoid conditional branches within warps
5. **Arithmetic Intensity**: Maximize compute per memory access

---

## Common Mathematical Patterns in GPU Kernels

### Element-wise Operations
```cuda
// Pattern: y[i] = f(x[i])
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    y[idx] = f(x[idx]);
}
```

### Reduction Pattern
```cuda
// Shared memory reduction
__shared__ float sdata[BLOCK_SIZE];
sdata[threadIdx.x] = input[idx];
__syncthreads();

for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
}
```

### Matrix Multiplication Tiled
```cuda
// Each thread computes one element of C
float sum = 0.0f;
for (int k = 0; k < K; k += TILE_SIZE) {
    // Load tiles into shared memory
    // Compute partial products
    sum += As[ty][k_local] * Bs[k_local][tx];
}
C[row][col] = sum;
```

This refresher covers the essential mathematical foundations needed for GPU programming, from basic linear algebra to advanced optimization techniques.

---

## Additional Advanced Concepts

### Numerical Precision Mathematics
**Mixed Precision Training**:
- Forward pass: FP16 ($\pm 6.55 \times 10^4$ range)
- Backward pass: FP32 ($\pm 3.4 \times 10^{38}$ range)
- Loss scaling: $\text{scaled\_loss} = \text{loss} \times 2^{scale}$

**Quantization**:

```math
\text{quantized} = \text{round}\left(\frac{\text{float\_value} - \text{zero\_point}}{\text{scale}}\right)
```

### Synchronization Mathematics
**Barrier Synchronization Cost**:

```math
\text{Sync Cost} = \text{Max Thread Time} - \text{Avg Thread Time}
```

**Load Balance Efficiency**:

```math
\text{Efficiency} = \frac{\text{Avg Work per Thread}}{\text{Max Work per Thread}}
```

### Tensor Broadcasting Rules
For tensors with shapes $(a_1, a_2, ..., a_n)$ and $(b_1, b_2, ..., b_m)$:
1. Align dimensions from right
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1

**Memory Expansion Factor**:

```math
\text{Expansion} = \prod_{i} \max(a_i, b_i) / \prod_{i} a_i
```

### Warp-Level Primitives
**Warp Shuffle**: Exchange data between threads in same warp
```
__shfl_down_sync(mask, value, delta)  // O(1) latency
```

**Warp Reduce Sum**:
```cpp
for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
}
// Total: log₂(32) = 5 shuffle operations
```

### Bank Conflict Mathematics
**Shared Memory Banks**: Typically 32 banks

```math
\text{Bank ID} = (\text{Address} / 4) \bmod 32
```

**Conflict Degree**:

```math
\text{Conflicts} = \max(\text{threads accessing same bank}) - 1
```

**Effective Bandwidth**:

```math
\text{Bandwidth}_{\text{effective}} = \frac{\text{Bandwidth}_{\text{peak}}}{\text{Conflict Degree} + 1}
```

---

## Missing Areas for Complete Coverage

The document covers core concepts but could be expanded with:

1. **Advanced Tensor Operations**
   - Tensor contractions and Einstein notation
   - Strided operations and memory layouts
   - Sparse tensor mathematics

2. **GPU-Specific Algorithms**
   - Warp-level matrix operations (Tensor Cores)
   - Cooperative groups mathematics
   - Multi-GPU communication patterns

3. **Performance Modeling**
   - Detailed roofline analysis
   - Memory coalescing efficiency models
   - Instruction-level parallelism

4. **Machine Learning Specific**
   - Attention mechanism mathematics
   - Transformer model computations
   - Gradient computation patterns

5. **Parallel Algorithms**
   - Sorting networks
   - Graph algorithms on GPU
   - Dynamic parallelism mathematics

6. **Error Analysis**
   - Floating-point error propagation
   - Numerical stability in iterative algorithms
   - Precision requirements analysis
