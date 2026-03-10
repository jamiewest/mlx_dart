# mlx_dart

A Dart port of [mlx-swift](https://github.com/ml-explore/mlx-swift) — MLX tensor operations and neural network primitives for Dart on Apple silicon.

MLX is an array framework for machine learning on Apple silicon. `mlx_dart` brings the same API surface to Dart, wrapping the [MLX C bindings](https://github.com/jamiewest/mlx_c_ffi) via FFI.

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  mlx_dart: ^0.1.0
```

The package requires `libmlxc.dylib` (or `libmlx.dylib`) to be present on the library search path at runtime. Build it from [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c).

## Quick Start

```dart
import 'package:mlx_dart/mlx_dart.dart';

void main() {
  final ctx = MLXContext.gpu();

  final a = MLXArray.fromFloats(ctx, [1, 2, 3, 4], shape: [2, 2]);
  final b = MLXArray.ones(ctx, [2, 2]);
  final c = a.matmul(b);
  c.eval();

  print(c.toFloat32List()); // [3.0, 3.0, 7.0, 7.0]

  c.dispose();
  b.dispose();
  a.dispose();
  ctx.dispose();
}
```

## API Status

The table below tracks progress against mlx-swift modules.

### MLX Core (`Source/MLX` in mlx-swift)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `MLXContext` (device + stream) | `Device` / `Stream` | `MLXContext` |
| `MLXDtype` | `DType` | `MLXDtype` |
| Array construction | `MLXArray.init(...)` | `MLXArray.zeros/ones/arange/fromFloats/fromInts/bool_/int_/float_` |
| Properties (`ndim`, `shape`, `size`, `dtype`, `nbytes`) | yes | yes |
| `eval()` / `evalAll` | yes | yes |
| Data extraction (`toFloat32List`, `toInt32List`, `itemFloat`, `itemInt`) | yes | yes |
| `toString()` | yes | yes |
| Arithmetic operators (`+`, `-`, `*`, `/`) | yes | yes |
| Shape ops (`reshape`, `expandDims`, `squeeze`, `flatten`, `transpose`, `T`, `swapAxes`) | yes | yes |
| Slicing (`slice`, `sliceUpdate`, `take`) | yes | yes |
| Reductions (`sum`, `mean`, `max`, `argmax`, `softmax`, `cumsum`, `sort`, `argsort`) | yes | yes |
| Element-wise math (`exp`, `log`, `sqrt`, `rsqrt`, `abs`, `sigmoid`, `tanh`, `sin`, `cos`, `square`, `erf`, `floor`, `ceil`, `round`, `clip`, `pad`) | yes | yes |
| Comparisons (`greaterEqual`, `less`, `logicalAnd`) | yes | partial |
| Type casting (`astype`) | yes | yes |
| `matmul` | yes | yes |
| `concatenate`, `stack` | yes | yes |
| `where` | yes | yes |
| `conv2d` | yes | yes |
| `repeat`, `split` | yes | yes |
| Fast NN primitives (`rmsNorm`, `layerNorm`, `rope`, `scaledDotProductAttention`) | `MLXFast` | yes (on `MLXArray`) |
| Causal mask utility | `createCausalMask` | `createCausalMask` |
| Bilinear resize | custom | `bilinearResize` |
| Sampling (`argmaxSample`, `categoricalSample`) | various | yes |
| Reductions (`min`, `prod`, `std`, `var`) | yes | yes |
| Element-wise math (`log1p`, `log2`, `logSumExp`, `maximum`, `minimum`, `negative`, `notEqual`, `isNaN`, `isInf`, `sign`, `sinh`, `cosh`, `tan`, `atan`, `acos`, `asin`, `expm1`, `erfInverse`) | yes | yes |
| Array ops (`tril`, `triu`, `trace`, `top`, `roll`, `einsum`, `inner`, `outer`, `kron`, `tensordot`, `tiled`, `nanToNum`, `degrees`, `radians`) | yes | yes |
| `eye`, `full`, `linspace` factory ops | yes | yes |
| `conv1d` | yes | yes |
| `conv3d` | yes | not yet |
| `convTransposed2d` | yes | yes |
| `convTransposed1d` | yes | yes |
| `convTransposed3d` | yes | not yet |
| Grad transforms (`grad`, `vjp`, `jvp`, `valueAndGrad`) | yes | not yet |
| `vmap` | yes | not yet |
| `compile` | yes | not yet |
| `MLXCustomFunction` | yes | not yet |
| Memory / wired memory | yes | yes (`activeMemory`, `cacheMemory`, `peakMemory`, `clearCache`, etc.) |
| State / global stream management | yes | not yet |
| IO (safetensors) | yes | yes |
| IO (GGUF) | yes | not yet |

### MLXRandom (`Source/MLXRandom`)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `seed`, `key`, `split` | yes | yes |
| `uniform` | yes | yes |
| `normal` | yes | yes |
| `multivariateNormal` | yes | yes |
| `randInt` | yes | yes |
| `bernoulli` | yes | yes |
| `truncatedNormal` | yes | yes |
| `gumbel` | yes | yes |
| `categorical` | partial (on `MLXArray`) | `categoricalSample` |
| `laplace` | yes | yes |

### MLXNN (`Source/MLXNN`)

#### Infrastructure

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `Module` base class | yes | `Module` |
| Parameter tracking / `parameters()` | yes | `ParametersMixin` |
| `loadWeights` | yes | yes |
| `sanitize` | yes | yes |
| `WeightLoader` | — | `WeightLoader` |
| `Sequential` container | yes | yes |
| `UnaryLayer` protocol | yes | yes (`UnaryLayer` interface) |
| Value-and-grad helpers | `ValueAndGrad` | not yet |

#### Layers

| Layer | mlx-swift | mlx_dart |
|---|---|---|
| `Linear` | yes | yes |
| `Embedding` | yes | yes |
| `RMSNorm` | yes | yes |
| `LayerNorm` | yes | yes |
| `Conv2d` | yes | yes |
| `GroupNorm` | yes | yes |
| `BatchNorm` | yes | yes |
| `InstanceNorm` | yes | yes |
| `Conv1d` | yes | yes |
| `Conv3d` | yes | not yet |
| `ConvTransposed1d` | yes | yes |
| `ConvTransposed2d` | yes | yes |
| `ConvTransposed3d` | yes | not yet |
| `Dropout` | yes | yes |
| `Dropout2d` | yes | yes |
| `Dropout3d` | yes | not yet |
| `MaxPool2d` | yes | yes |
| `MaxPool1d` | yes | yes |
| `MaxPool3d` | yes | not yet |
| `AvgPool2d` | yes | yes |
| `AvgPool1d` | yes | yes |
| `AvgPool3d` | yes | not yet |
| `MultiHeadAttention` | yes | yes |
| `Transformer` (encoder + decoder) | yes | yes (`TransformerEncoder`, `TransformerDecoder`) |
| `SinusoidalPositionalEncoding` | yes | yes |
| `RNN`, `GRU`, `LSTM` | yes | yes |
| `Upsample` | yes | yes |
| Quantized layers | yes | not yet |
| KV cache (`Cache`, `RotatingKVCache`) | yes | yes (`KVCache`, `RotatingKVCache`) |

#### Activations

Functional activations available on `MLXArray`: `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`, `softmax()`, `leakyRelu()`, `elu()`, `celu()`.

| Activation | mlx-swift | mlx_dart |
|---|---|---|
| `relu`, `gelu`, `silu`, `sigmoid`, `tanh` | yes | yes (on `MLXArray`) |
| `leakyRelu` | yes | yes |
| `elu`, `celu` | yes | yes |
| `relu6`, `reluSquared` | yes | yes |
| `logSoftmax` | yes | yes |
| `softPlus` / `softsign` | yes | yes |
| `softshrink` | yes | yes |
| Activation layer classes (`ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `ELU`, `CELU`) | yes | yes |

#### Losses

| Loss | mlx-swift | mlx_dart |
|---|---|---|
| `crossEntropy` | yes | yes |
| `binaryCrossEntropy` | yes | yes |
| `l1Loss`, `mseLoss` | yes | yes |
| `nllLoss`, `klDivLoss` | yes | yes |
| `smoothL1Loss`, `huberLoss` | yes | yes |
| `hingeLoss`, `tripletLoss` | yes | yes |
| `logCoshLoss`, `cosineSimilarityLoss` | yes | yes |

### MLXOptimizers (`Source/MLXOptimizers`)

| Optimizer | mlx-swift | mlx_dart |
|---|---|---|
| `SGD` | yes | yes |
| `Adam`, `AdamW`, `Adamax` | yes | yes |
| `RMSprop`, `AdaGrad`, `AdaDelta` | yes | yes |
| `Lion` | yes | yes |
| `Adafactor` | yes | yes |

### MLXLinalg (`Source/MLXLinalg`)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `norm` | yes | yes |
| `svd`, `qr` | yes | yes |
| `lu` | yes | yes |
| `inv` | yes | yes |
| `triInv`, `choleskyInv` | yes | yes |
| `cholesky` | yes | yes |
| `solve`, `solveTriangular` | yes | yes |
| `cross` | yes | yes |

### MLXFFT (`Source/MLXFFT`)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn` | yes | yes |
| `rfft`, `irfft`, `rfft2`, `irfft2` | yes | yes |

## Architecture

```
mlx_dart
├── lib/src/
│   ├── context.dart   — MLXContext, MLXException
│   ├── array.dart     — MLXArray, MLXDtype, free ops
│   ├── module.dart    — Module, ParametersMixin, WeightLoader
│   ├── layers.dart    — Linear, Embedding, RMSNorm, LayerNorm, Conv2d, Conv1d, Sequential, Dropout, Dropout2d, GroupNorm, InstanceNorm, BatchNorm, MultiHeadAttention, TransformerEncoder, TransformerDecoder, SinusoidalPositionalEncoding, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, Upsample, RNN, GRU, LSTM, activation layer classes
│   ├── losses.dart    — MLX loss functions
│   ├── optimizers.dart — SGD, Adam, AdamW, Adamax, RMSprop, AdaGrad, AdaDelta, Lion, Adafactor
│   ├── kv_cache.dart  — KVCache, RotatingKVCache
│   ├── random.dart    — MLXRandom
│   ├── linalg.dart    — MLXLinalg
│   ├── fft.dart       — MLXFFT
│   └── io.dart        — MLXSafetensors
└── pubspec.yaml
```

Key design decisions mirroring mlx-swift:

- **Explicit context** — `MLXContext` wraps a device + stream. All arrays belong to a context (mlx-swift uses an implicit global default, Dart makes it explicit).
- **Manual disposal** — `MLXArray.dispose()` must be called when an array is no longer needed (no GC finaliser). Consider `try/finally` patterns.
- **Functional style** — operations return new arrays rather than mutating in place, matching mlx-swift.

## Contributing

Contributions expanding coverage of the mlx-swift API surface are welcome. Good first targets:

1. Additional `MLXArray` ops (missing element-wise math, reductions, factory ops)
2. `MLXRandom` module
3. `Sequential` container and remaining `MLXNN` layers
4. `MLXOptimizers`
5. Grad / vmap transforms

## License

Apache 2.0 — see [LICENSE](LICENSE).
