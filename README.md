# Backpropagation Is Just 3 GEMM Calls: The Math Nobody Shows You

> Every deep learning framework hides the same secret in plain sight: a fully-connected layer's entire training cycle — forward, weight gradient, input gradient — is three calls to `cublasSgemm`. No loops, no element-wise logic, no special cases. Just three matrix multiplies. This article derives **why** from first principles and shows **what happens on the GPU** when each one executes.

![Three cuBLAS GEMM Calls](img/three_cublas_gemm_calls.png)

---

## Table of Contents

1. [Why and Motivation](#1-why-and-motivation)
2. [Notation and Setup](#2-notation-and-setup)
3. [The Three GEMMs — Derivation from Matrix Calculus](#3-the-three-gemms--derivation-from-matrix-calculus)
   - [3.1 Forward Pass — GEMM #1](#31-forward-pass--gemm-1)
   - [3.2 Weight Gradient — GEMM #2](#32-weight-gradient--gemm-2)
   - [3.3 Input Gradient — GEMM #3](#33-input-gradient--gemm-3)
4. [The Matrix Calculus Bridge: Jacobians, Traces, and the Chain Rule](#4-the-matrix-calculus-bridge-jacobians-traces-and-the-chain-rule)
5. [Why This Is Not Obvious](#5-why-this-is-not-obvious)
6. [From Math to Metal: What the GPU Actually Executes](#6-from-math-to-metal-what-the-gpu-actually-executes)
   - [6.1 cuBLAS GEMM Signature](#61-cublas-gemm-signature)
   - [6.2 Mapping the 3 GEMMs to cublasSgemm](#62-mapping-the-3-gemms-to-cublassgemm)
   - [6.3 The Transpose Tax](#63-the-transpose-tax)
7. [Numerical Verification](#7-numerical-verification)
8. [Scaling to a Full Network](#8-scaling-to-a-full-network)
9. [Performance Implications](#9-performance-implications)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Why and motivation

When you call `loss.backward()` in PyTorch, the autograd engine dispatches a sequence of GPU kernel launches. For any `nn.Linear` layer, those launches reduce to three `cublasSgemm` calls — one for the forward pass, two for the backward pass.

This fact has profound consequences:

- **~80% of training FLOPs** in Transformers and MLPs come from these three GEMMs
- Optimizing GEMM **is** optimizing training — there is no "other" bottleneck to find
- Understanding the math behind each GEMM tells you exactly **which transpose costs you bandwidth** and **why the backward pass is 2× the cost** of the forward pass

Yet most ML courses teach backpropagation as element-wise partial derivatives and chain rule diagrams. The jump from $\frac{\partial L}{\partial w_{ij}}$ to `cublasSgemm(handle, CUBLAS_OP_T, ...)` is never shown.

This article fills that gap. The practical implementation and an optimization sequence for these GEMM calls can be found in the companion repository: [re-engineering-cublas](https://github.com/danepham2204/re-engineering-cublas).

---

## 2. Notation and Setup

Consider a single fully-connected (linear) layer with no bias:

$$Y = XW$$

| Symbol                          | Shape        | Description                                       |
| :------------------------------ | :----------- | :------------------------------------------------ |
| $X$                             | $M \times K$ | Input batch — $M$ samples, $K$ input features     |
| $W$                             | $K \times N$ | Weight matrix — $K$ inputs mapped to $N$ outputs  |
| $Y$                             | $M \times N$ | Output — $M$ samples, $N$ output features         |
| $L$                             | scalar       | Loss function value (e.g., cross-entropy)         |
| $\frac{\partial L}{\partial Y}$ | $M \times N$ | Upstream gradient — received from the layer above |

The training loop computes three things:

1. **Forward:** $Y = XW$
2. **Weight gradient:** $\frac{\partial L}{\partial W}$ to update $W$
3. **Input gradient:** $\frac{\partial L}{\partial X}$ to propagate backward to the previous layer

Each of these is a single GEMM.

---

## 3. The Three GEMMs — Derivation from Matrix Calculus

### 3.1 Forward Pass — GEMM #1

![Forward Pass — GEMM #1: Y = X · W](img/forward_pass_gemm_1.png)

$$\boxed{Y = X \cdot W}$$

This is the definition of the layer. Element-wise:

$$Y_{ij} = \sum_{k=1}^{K} X_{ik} \cdot W_{kj}$$

This is a textbook matrix multiplication:

- **Shape:** $(M \times K) \cdot (K \times N) \rightarrow (M \times N)$
- **FLOPs:** $2 \cdot M \cdot N \cdot K$ (one multiply + one add per term in the sum)
- **GPU call:** `cublasSgemm(..., X, W, Y)`

Nothing surprising here. The key insight comes in the backward pass.

---

### 3.2 Weight Gradient — GEMM #2

![Backward Pass — GEMM #2 & #3: Weight and Input Gradients](img/backward_pass_gemm_2_3.png)

We need $\frac{\partial L}{\partial W}$ — a $K \times N$ matrix that tells us how to update each weight.

**Starting point:** We have $\frac{\partial L}{\partial Y}$ from the layer above (or from the loss function directly).

**Derivation using the trace/differential technique:**

The loss $L$ depends on $W$ only through $Y = XW$. Consider a small perturbation $dW$:

$$dY = X \cdot dW$$

The scalar change in loss is:

$$dL = \text{tr}\!\left(\left(\frac{\partial L}{\partial Y}\right)^{\!T} dY\right) = \text{tr}\!\left(\left(\frac{\partial L}{\partial Y}\right)^{\!T} X \cdot dW\right)$$

Using the matrix transpose property $(AB)^T = B^T A^T$:

$$dL = \text{tr}\!\left(\left(\underbrace{X^T \frac{\partial L}{\partial Y}}_{\text{this is } \frac{\partial L}{\partial W}}\right)^{\!T} \cdot dW\right)$$

Wait — let's be careful. The definition of the matrix gradient is:

$$dL = \text{tr}\!\left(\left(\frac{\partial L}{\partial W}\right)^{\!T} dW\right)$$

Matching terms:

$$\boxed{\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}}$$

**Shape check:**

$$\underbrace{X^T}_{K \times M} \cdot \underbrace{\frac{\partial L}{\partial Y}}_{M \times N} = \underbrace{\frac{\partial L}{\partial W}}_{K \times N} \quad \checkmark \text{ (same shape as } W\text{)}$$

This is GEMM #2. The **transpose of the input** multiplied by the **upstream gradient**.

---

### 3.3 Input Gradient — GEMM #3

We need $\frac{\partial L}{\partial X}$ — a $M \times K$ matrix to pass backward to the previous layer.

**Derivation:** Now hold $W$ fixed and perturb $X$:

$$dY = dX \cdot W$$

$$dL = \text{tr}\!\left(\left(\frac{\partial L}{\partial Y}\right)^{\!T} dX \cdot W\right)$$

Using the cyclic property: $\text{tr}(A^T B C) = \text{tr}(C A^T B)$

$$dL = \text{tr}\!\left(W \left(\frac{\partial L}{\partial Y}\right)^{\!T} dX\right) = \text{tr}\!\left(\left(\frac{\partial L}{\partial Y} \cdot W^T\right)^{\!T} dX\right)$$

Matching with $dL = \text{tr}\!\left(\left(\frac{\partial L}{\partial X}\right)^T dX\right)$:

$$\boxed{\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T}$$

**Shape check:**

$$\underbrace{\frac{\partial L}{\partial Y}}_{M \times N} \cdot \underbrace{W^T}_{N \times K} = \underbrace{\frac{\partial L}{\partial X}}_{M \times K} \quad \checkmark \text{ (same shape as } X\text{)}$$

This is GEMM #3. The **upstream gradient** multiplied by the **transpose of the weights**.

---

### Summary: The Complete Training Cycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ONE LINEAR LAYER TRAINING CYCLE                  │
├──────────────┬──────────────────────────┬────────────────────────────┤
│ Phase        │ Math                     │ Shape                      │
├──────────────┼──────────────────────────┼────────────────────────────┤
│ GEMM #1      │  Y    = X  · W          │ (M×K)·(K×N) → (M×N)       │
│ Forward      │                          │                            │
├──────────────┼──────────────────────────┼────────────────────────────┤
│ GEMM #2      │ ∂L/∂W = Xᵀ · ∂L/∂Y     │ (K×M)·(M×N) → (K×N)       │
│ Weight Grad  │                          │                            │
├──────────────┼──────────────────────────┼────────────────────────────┤
│ GEMM #3      │ ∂L/∂X = ∂L/∂Y · Wᵀ     │ (M×N)·(N×K) → (M×K)       │
│ Input Grad   │                          │                            │
└──────────────┴──────────────────────────┴────────────────────────────┘
```

Three matrix multiplications. That's the entire training cycle for a linear layer.

---

## 4. The Matrix Calculus Bridge: Jacobians, Traces, and the Chain Rule

The derivations above used the **trace/differential** technique. Here's why it works and how it connects to the element-wise chain rule you already know.

### The Element-Wise View (What Textbooks Teach)

The chain rule for a single weight $w_{kn}$:

$$\frac{\partial L}{\partial w_{kn}} = \sum_{i=1}^{M} \sum_{j=1}^{N} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial w_{kn}}$$

Since $Y_{ij} = \sum_{\ell} X_{i\ell} W_{\ell j}$, the Jacobian term is:

$$\frac{\partial Y_{ij}}{\partial w_{kn}} = \begin{cases} X_{ik} & \text{if } j = n \\ 0 & \text{otherwise} \end{cases}$$

Substituting:

$$\frac{\partial L}{\partial w_{kn}} = \sum_{i=1}^{M} \frac{\partial L}{\partial Y_{in}} \cdot X_{ik} = \sum_{i=1}^{M} X_{ik} \cdot \frac{\partial L}{\partial Y_{in}}$$

This is the $(k, n)$ element of the matrix product $X^T \cdot \frac{\partial L}{\partial Y}$ — exactly what we derived above.

### The Trace/Differential View (What This Article Teaches)

Instead of working element-by-element, we express the total differential of the scalar $L$ as:

$$dL = \text{tr}\!\left(\left(\frac{\partial L}{\partial W}\right)^{\!T} dW\right)$$

This is the **Fréchet derivative** for matrix-valued arguments. The trace collapses the matrix inner product into a scalar. Both approaches yield the same gradient — the trace technique just does it in one line instead of $K \times N$ separate derivations.

### Why the Trace Technique Matters for GPU Programming

The element-wise derivation gives you $K \times N$ separate scalars. The trace derivation gives you a **single matrix equation** — which maps directly to a single GEMM call. The math and the GPU kernel share the same structure. This is not a coincidence: GEMM was designed to compute exactly these kinds of products.

---

## 5. Why This Is Not Obvious

**Objection 1: "But real layers have bias and activation functions."**

Bias is an addition ($Y = XW + b$), not a GEMM. Its gradient is a column-sum: $\frac{\partial L}{\partial b} = \mathbf{1}^T \frac{\partial L}{\partial Y}$ — a reduction, not a matrix multiply. It costs negligible time compared to the three GEMMs.

Activation functions (ReLU, GELU, etc.) are element-wise operations. Their backward pass is an element-wise multiply by the activation derivative mask. These are memory-bound kernels that take <1% of total training time on large models.

The three GEMMs dominate.

**Objection 2: "This only applies to fully-connected layers."**

Convolutions can be expressed as GEMM via `im2col`. Attention mechanisms are sequences of batched GEMMs. Embedding lookups are sparse GEMMs. The GEMM-centric view extends far beyond `nn.Linear`.

**Objection 3: "Backprop frameworks use autograd, not explicit GEMM calls."**

Correct at the API level. Under the hood, PyTorch's `torch.mm` dispatches to cuBLAS. When you trace `loss.backward()` with `torch.profiler`, you will see `cublas::gemm` in the kernel trace — three times per linear layer.

---

## 6. From Math to Metal: What the GPU Actually Executes

### 6.1 cuBLAS GEMM Signature

The standard SGEMM call computes:

$$C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C$$

where $\text{op}(X)$ is either $X$ or $X^T$, controlled by the `transa`/`transb` flags.

```c
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,    // CUBLAS_OP_N or CUBLAS_OP_T
    cublasOperation_t transb,
    int m, int n, int k,         // output dimensions and reduction dim
    const float *alpha,
    const float *A, int lda,     // matrix A and its leading dimension
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
);
```

### 6.2 Mapping the 3 GEMMs to cublasSgemm

**Critical note:** cuBLAS uses **column-major** storage (Fortran convention), while C/C++/Python use **row-major**. A row-major matrix $X$ of shape $(M, K)$ looks like a column-major matrix $X^T$ of shape $(K, M)$ to cuBLAS. This means every GEMM call requires careful transposition.

For row-major data passed to cuBLAS, the trick is:

$$C = A \cdot B \quad \Longleftrightarrow \quad C^T = B^T \cdot A^T$$

Since cuBLAS sees row-major data as already transposed, we compute $C^T$ and the output is correctly laid out in row-major.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ GEMM #1 — Forward: Y = X · W                                              │
│                                                                            │
│ cuBLAS sees:  Y^T = W^T · X^T                                             │
│                                                                            │
│ cublasSgemm(handle,                                                        │
│     CUBLAS_OP_N,              // W^T is already "W" in row-major → no-op   │
│     CUBLAS_OP_N,              // X^T is already "X" in row-major → no-op   │
│     N, M, K,                  // output (N×M in col-major = M×N in row)    │
│     &alpha,                                                                │
│     W, N,                     // "W^T" to cuBLAS, leading dim = N          │
│     X, K,                     // "X^T" to cuBLAS, leading dim = K          │
│     &beta,                                                                 │
│     Y, N);                    // "Y^T" to cuBLAS, leading dim = N          │
├────────────────────────────────────────────────────────────────────────────┤
│ GEMM #2 — Weight Grad: ∂L/∂W = X^T · ∂L/∂Y                               │
│                                                                            │
│ cuBLAS sees:  (∂L/∂W)^T = (∂L/∂Y)^T · X                                   │
│                                                                            │
│ cublasSgemm(handle,                                                        │
│     CUBLAS_OP_N,              // (∂L/∂Y)^T is "∂L/∂Y" in row-major        │
│     CUBLAS_OP_T,              // need X, but cuBLAS sees X^T → transpose   │
│     N, K, M,                                                               │
│     &alpha,                                                                │
│     dY, N,                                                                 │
│     X,  K,                                                                 │
│     &beta,                                                                 │
│     dW, N);                                                                │
├────────────────────────────────────────────────────────────────────────────┤
│ GEMM #3 — Input Grad: ∂L/∂X = ∂L/∂Y · W^T                                │
│                                                                            │
│ cuBLAS sees:  (∂L/∂X)^T = W · (∂L/∂Y)^T                                   │
│                                                                            │
│ cublasSgemm(handle,                                                        │
│     CUBLAS_OP_T,              // need W^T, cuBLAS sees W^T^T = W → transp  │
│     CUBLAS_OP_N,              // (∂L/∂Y)^T is "∂L/∂Y" in row-major        │
│     K, M, N,                                                               │
│     &alpha,                                                                │
│     W,  N,                                                                 │
│     dY, N,                                                                 │
│     &beta,                                                                 │
│     dX, K);                                                                │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 The Transpose Tax

Notice that **2 out of 3 GEMMs require a transposed operand.** On the GPU, transposition is not a separate operation — it changes the memory **access pattern**:

| Access Pattern                   | Stride Between Consecutive Elements | Effect                                  |
| :------------------------------- | :---------------------------------- | :-------------------------------------- |
| Normal (row of row-major)        | 1 element = 4 bytes                 | Coalesced — 1 cache line per warp       |
| Transposed (column of row-major) | $N$ elements = $4N$ bytes           | Strided — up to 32 cache lines per warp |

Reading a column of a row-major matrix means jumping $N$ elements between consecutive reads. For $N = 4096$, that's a stride of 16 KB between consecutive warp lanes — every thread hits a different cache line.

This is the **transpose tax**: the backward pass's two GEMMs inherently have worse memory access patterns than the forward pass. This is one reason why `loss.backward()` typically takes **2–3× longer** than the forward pass, even though it performs exactly 2× the arithmetic.

High-performance GEMM kernels handle this through **shared memory staging**: the transposed operand is loaded tile-by-tile into shared memory (which has no coalescing requirement), then read from shared memory in the optimal layout. This is exactly what Versions 02–10 in [re-engineering-cublas](https://github.com/danepham2204/re-engineering-cublas) implement.

---

## 7. Numerical Verification

```python
import numpy as np

np.random.seed(42)
M, K, N = 64, 128, 32

X = np.random.randn(M, K).astype(np.float32)
W = np.random.randn(K, N).astype(np.float32)

# ═══════════ GEMM #1: Forward ═══════════
Y = X @ W                          # (M, N)

# Simple loss: L = (1/2) * ||Y||^2
# so ∂L/∂Y = Y
dL_dY = Y.copy()

# ═══════════ GEMM #2: Weight gradient ═══════════
dL_dW_analytic = X.T @ dL_dY       # (K, N)

# ═══════════ GEMM #3: Input gradient ═══════════
dL_dX_analytic = dL_dY @ W.T       # (M, K)

# ═══════════ Verify with numerical gradient ═══════════
eps = 1e-4

dL_dW_numerical = np.zeros_like(W)
for i in range(K):
    for j in range(N):
        W_plus = W.copy();  W_plus[i, j] += eps
        W_minus = W.copy(); W_minus[i, j] -= eps
        L_plus  = 0.5 * np.sum((X @ W_plus) ** 2)
        L_minus = 0.5 * np.sum((X @ W_minus) ** 2)
        dL_dW_numerical[i, j] = (L_plus - L_minus) / (2 * eps)

dL_dX_numerical = np.zeros_like(X)
for i in range(M):
    for j in range(K):
        X_plus = X.copy();  X_plus[i, j] += eps
        X_minus = X.copy(); X_minus[i, j] -= eps
        L_plus  = 0.5 * np.sum((X_plus @ W) ** 2)
        L_minus = 0.5 * np.sum((X_minus @ W) ** 2)
        dL_dX_numerical[i, j] = (L_plus - L_minus) / (2 * eps)

print("=== GEMM #2: ∂L/∂W ===")
print(f"  Analytic shape:  {dL_dW_analytic.shape}  (should match W: {W.shape})")
print(f"  Max abs error:   {np.max(np.abs(dL_dW_analytic - dL_dW_numerical)):.2e}")

print("\n=== GEMM #3: ∂L/∂X ===")
print(f"  Analytic shape:  {dL_dX_analytic.shape}  (should match X: {X.shape})")
print(f"  Max abs error:   {np.max(np.abs(dL_dX_analytic - dL_dX_numerical)):.2e}")
```

Expected output:

```
=== GEMM #2: ∂L/∂W ===
  Analytic shape:  (128, 32)  (should match W: (128, 32))
  Max abs error:   ~1e-03

=== GEMM #3: ∂L/∂X ===
  Analytic shape:  (64, 128)  (should match X: (64, 128))
  Max abs error:   ~1e-03
```

The analytic gradients (from 2 matrix multiplications) match the brute-force numerical gradients to floating-point precision. No loops over individual weights needed.

---

## 8. Scaling to a Full Network

A network with $L$ linear layers performs:

| Phase     | GEMM Calls | Total    |
| :-------- | :--------- | :------- |
| Forward   | $L$        | $L$      |
| Backward  | $2L$       | $2L$     |
| **Total** |            | **$3L$** |

For a 12-layer Transformer encoder with 4 linear projections per layer (Q, K, V, output):

```
Linear layers:      12 × 4 = 48
GEMM calls/step:    48 × 3 = 144

For hidden_dim = 768, seq_len × batch = 512:
  FLOPs per GEMM:    2 × 512 × 768 × 768 ≈ 603M
  FLOPs per step:    144 × 603M ≈ 87 GFLOP
  A100 at 19.5 TFLOP/s: 87 / 19500 ≈ 4.5 ms per step
```

The entire training step is dominated by 144 GEMM calls. Everything else — LayerNorm, softmax, dropout, loss computation — accounts for the remaining ~20%.

---

## 9. Performance Implications

### 9.1 The Backward Pass Is Not 2× — It's Worse

Arithmetically, the backward pass performs exactly $2\times$ the FLOPs of the forward pass ($2L$ GEMMs vs $L$ GEMMs, each with the same $2MNK$ complexity).

In practice, `loss.backward()` takes 2.5–3× the wall time of the forward pass because:

1. **Transpose tax** — GEMM #2 reads $X^T$ and GEMM #3 reads $W^T$, both strided
2. **Memory pressure** — backward requires storing activations from the forward pass (activation memory)
3. **Kernel launch overhead** — 2× more kernels to launch, each with cold caches

### 9.2 Why Mixed Precision Matters

With FP16 Tensor Cores (WMMA), each GEMM call can use:

$$D_{16 \times 16} = A_{16 \times 16}^{\text{FP16}} \times B_{16 \times 16}^{\text{FP16}} + C_{16 \times 16}^{\text{FP32}}$$

This gives 8,192 FLOPs per warp per instruction vs 64 for scalar FP32 FMA — a **128× throughput increase**. On a T4 GPU:

```
FP32 peak:          8.1 TFLOP/s
FP16 Tensor Core:  65.0 TFLOP/s   ← 8× higher
```

Since each GEMM call is identical in structure, switching from FP32 to FP16 accelerates all three equally. This is why `torch.cuda.amp` (automatic mixed precision) speeds up training by 2–3× with virtually no code change — it's the same three GEMMs, just with wider hardware datapaths.

### 9.3 Connection to GEMM Kernel Optimization

If the three GEMMs are the bottleneck, then **optimizing GEMM is optimizing training**. The optimization trajectory in [re-engineering-cublas](https://github.com/danepham2204/re-engineering-cublas) directly applies:

```
Naive SGEMM (v01):        465 GFLOP/s  ← if your GEMM is this slow,
                                           training is 70× slower than it needs to be
Vectorized + Tiled (v05): 3304 GFLOP/s
Tensor Core SMEM (v08):   3230 GFLOP/s
cuBLAS reference:        ~40000 GFLOP/s ← this is what loss.backward() actually uses
```

Every kernel optimization from shared memory tiling (v02) to vectorized Tensor Core pipelines (v10) is directly accelerating one of these three GEMM calls.

---

## 10. Conclusion

The entire training cycle of a linear layer is three matrix multiplications:

$$Y = X \cdot W \qquad \frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} \qquad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$$

This is not an approximation. It is an exact consequence of the chain rule applied to matrix-valued functions. The trace/differential technique from matrix calculus gives us the result in closed form — one matrix equation per gradient, one GEMM call per equation.

On the GPU, these three equations become three calls to `cublasSgemm`, each with a different transpose configuration. The transpose tax (strided memory access on 2 of 3 GEMMs) and the 2:1 backward-to-forward ratio explain why training is always slower than inference, and why GEMM kernel optimization is the single highest-leverage activity in ML systems engineering.

Understanding this bridge — from the chain rule to `cublasSgemm` — transforms backpropagation from a black-box autograd call into something you can profile, optimize, and reason about at the hardware level.

---

## 11. References

- Deisenroth, Faisal & Ong — _Mathematics for Machine Learning_ (2020), Chapter 5: Vector Calculus
- Magnus & Neudecker — _Matrix Differential Calculus with Applications in Statistics and Econometrics_ (2019) — the definitive reference for the trace/differential technique
- Petersen & Pedersen — _The Matrix Cookbook_ (2012) — compact reference for matrix derivative identities
- NVIDIA — [cuBLAS Library Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- NVIDIA — [CUDA C++ Best Practices Guide: Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- **re-engineering-cublas** — [Re-engineering cuBLAS: From a Naive CUDA Kernel to a Tensor Core Pipeline](https://github.com/danepham2204/re-engineering-cublas) — 10-version optimization sequence demonstrating the kernel engineering behind each GEMM call
