# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-03-09

### Added

- `MLXContext` — wraps an MLX device + stream; factory constructors for GPU (Metal) and CPU.
- `MLXArray` — n-dimensional array backed by the MLX C runtime with full ownership semantics.
- `MLXDtype` — Dart enum covering all 14 MLX element types (`bool_`, `uint8`–`uint64`, `int8`–`int64`, `float16`, `bfloat16`, `float32`, `float64`, `complex64`).
- `MLXException` — typed exception carrying the failed operation name and error code.
- Array construction: `zeros`, `ones`, `arange`, `fromFloats`, `fromInts`, `bool_`, `int_`, `float_`.
- Array properties: `ndim`, `shape`, `size`, `dtype`, `nbytes`, `dim(axis)`.
- `eval()` / `evalAll()` for lazy evaluation.
- Data extraction: `toFloat32List`, `toInt32List`, `itemFloat`, `itemInt`.
- Arithmetic operators: `+`, `-`, `*`, `/`.
- Shape ops: `reshape`, `expandDims`, `squeeze`, `flatten`, `transpose`, `T`, `swapAxes`.
- Slicing & indexing: `slice`, `sliceUpdate`, `take`.
- Reductions: `sum`, `mean`, `max`, `argmax`, `softmax`, `cumsum`, `sort`, `argsort`.
- Element-wise math: `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `sigmoid`, `tanh`, `sin`, `cos`, `square`, `erf`, `floor`, `ceil`, `round`, `clip`, `pad`.
- Activation functions: `relu`, `gelu`, `silu`.
- Comparisons: `greaterEqual`, `less`, `logicalAnd`.
- Type casting: `astype`.
- Fast Metal-accelerated NN primitives: `rmsNorm`, `layerNorm`, `rope`, `scaledDotProductAttention`.
- Free functions: `conv2d`, `concatenate`, `stack`, `where`, `evalAll`, `createCausalMask`, `bilinearResize`.
- Sampling: `argmaxSample`, `categoricalSample`.
- `Module` base class and `ParametersMixin` for neural network parameter management.
- `WeightLoader` for loading checkpoints by dotted-path keys.
- Built-in layers: `Linear`, `Embedding`, `RMSNorm`, `LayerNorm`, `Conv2d`.
