# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-10

### Added

#### MLX Core (`MLXContext`, `MLXArray`)

- `MLXContext` — wraps an MLX device + stream; factory constructors for GPU (Metal) and CPU.
- `MLXArray` — n-dimensional array backed by the MLX C runtime with full ownership semantics.
- `MLXDtype` — Dart enum covering all 14 MLX element types (`bool_`, `uint8`–`uint64`, `int8`–`int64`, `float16`, `bfloat16`, `float32`, `float64`, `complex64`).
- `MLXException` — typed exception carrying the failed operation name and error code.
- Array construction: `zeros`, `ones`, `eye`, `full`, `linspace`, `arange`, `fromFloats`, `fromInts`, `bool_`, `int_`, `float_`.
- Array properties: `ndim`, `shape`, `size`, `dtype`, `nbytes`, `dim(axis)`.
- `eval()` / `evalAll()` for lazy evaluation.
- Data extraction: `toFloat32List`, `toInt32List`, `itemFloat`, `itemInt`.
- Arithmetic operators: `+`, `-`, `*`, `/`.
- Shape ops: `reshape`, `expandDims`, `squeeze`, `flatten`, `transpose`, `T`, `swapAxes`.
- Slicing & indexing: `slice`, `sliceUpdate`, `take`.
- Reductions: `sum`, `mean`, `max`, `min`, `prod`, `argmax`, `argmin`, `std`, `variance`, `softmax`, `cumsum`, `sort`, `argsort`.
- Element-wise math: `exp`, `expm1`, `log`, `log1p`, `log2`, `logSumExp`, `sqrt`, `rsqrt`, `abs`, `sigmoid`, `tanh`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `atan`, `acos`, `asin`, `square`, `erf`, `erfInverse`, `floor`, `ceil`, `round`, `clip`, `pad`, `sign`, `negative`, `degrees`, `radians`, `nanToNum`.
- Activation functions: `relu`, `relu6`, `reluSquared`, `gelu`, `silu`, `leakyRelu`, `elu`, `celu`, `logSoftmax`, `softPlus`, `softsign`, `softshrink`.
- Comparisons: `greaterEqual`, `less`, `logicalAnd`, `notEqual`, `isNaN`, `isInf`.
- Type casting: `astype`.
- Fast Metal-accelerated NN primitives: `rmsNorm`, `layerNorm`, `rope`, `scaledDotProductAttention`.
- Array ops: `tril`, `triu`, `trace`, `top`, `roll`, `einsum`, `inner`, `outer`, `kron`, `tensordot`, `tiled`, `repeat`, `split`.
- Free functions: `conv1d`, `conv2d`, `convTransposed1d`, `convTransposed2d`, `concatenate`, `stack`, `where`, `evalAll`, `createCausalMask`, `bilinearResize`.
- Sampling: `argmaxSample`, `categoricalSample`.
- Memory management on `MLXContext`: `activeMemory`, `cacheMemory`, `peakMemory`, `memoryLimit`, `clearCache()`, `resetPeakMemory()`, `setMemoryLimit()`, `setCacheLimit()`.

#### MLXRandom

- `MLXRandom` — static interface for random number generation.
- `seed`, `key`, `split`.
- `uniform`, `normal`, `multivariateNormal`.
- `randInt`, `bernoulli`, `truncatedNormal`, `gumbel`, `laplace`.

#### MLXNN — Layers

- `Module` base class and `ParametersMixin` for neural network parameter management.
- `WeightLoader` for loading checkpoints by dotted-path keys, `sanitize`.
- `UnaryLayer` interface for single-input / single-output layers.
- `Sequential` container.
- Linear layers: `Linear`, `Embedding`.
- Normalisation: `RMSNorm`, `LayerNorm`, `GroupNorm`, `BatchNorm`, `InstanceNorm`.
- Convolution: `Conv1d`, `Conv2d`, `ConvTransposed1d`, `ConvTransposed2d`.
- Pooling: `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`.
- Regularisation: `Dropout`, `Dropout2d`.
- Attention: `MultiHeadAttention`.
- Transformer: `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`, `TransformerDecoder`.
- Positional encoding: `SinusoidalPositionalEncoding`.
- Recurrent: `RNN`, `GRU`, `LSTM`.
- Upsampling: `Upsample` (bilinear resize).
- KV cache: `KVCache`, `RotatingKVCache`.
- Activation layer classes: `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `ELU`, `CELU`.

#### MLXNN — Losses

- `crossEntropy`, `binaryCrossEntropy`.
- `l1Loss`, `mseLoss`.
- `nllLoss`, `klDivLoss`.
- `smoothL1Loss`, `huberLoss`.
- `hingeLoss`, `tripletLoss`.
- `logCoshLoss`, `cosineSimilarityLoss`.

#### MLXOptimizers

- `SGD` (with momentum and weight decay).
- `Adam`, `AdamW`, `Adamax`.
- `RMSprop`, `AdaGrad`, `AdaDelta`.
- `Lion`.
- `Adafactor`.

#### MLXLinalg

- `norm`, `svd`, `qr`, `lu`.
- `inv`, `triInv`, `choleskyInv`, `cholesky`.
- `solve`, `solveTriangular`, `cross`.

#### MLXFFT

- `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`.
- `rfft`, `irfft`, `rfft2`, `irfft2`.

#### IO

- `MLXSafetensors` — load and save weights in the safetensors format.

#### Tests

- Pure-Dart unit tests (no native lib required) in `test/mlx_dart_test.dart`.
- Integration tests tagged `@Tags(['integration'])` in `test/integration/mlx_dart_integration_test.dart`, covering all modules above.
