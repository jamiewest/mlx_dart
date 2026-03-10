# mlx_dart

A Dart port of [mlx-swift](https://github.com/ml-explore/mlx-swift) — MLX tensor operations and neural network primitives for Dart on Apple silicon.

MLX is an array framework for machine learning on Apple silicon. `mlx_dart` brings the same API surface to Dart, wrapping the [MLX C bindings](https://github.com/jamiewest/mlx_c_ffi) via FFI.

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  mlx_dart:
    git:
      url: https://github.com/jamiewest/mlx_dart.git
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
| **Missing ops** (`min`, `prod`, `std`, `var`, `log1p`, `log2`, `logSumExp`, `maximum`, `minimum`, `negative`, `notEqual`, `isNaN`, `isInf`, `sign`, `sinh`, `cosh`, `tan`, `atan`, `acos`, `asin`, `expm1`, `erfInverse`, `tril`, `triu`, `trace`, `top`, `roll`, `einsum`, `inner`, `outer`, `kron`, `tensordot`, `tiled`, `nanToNum`, `degrees`, `radians`) | yes | not yet |
| `eye`, `full`, `linspace` factory ops | yes | not yet |
| `conv1d`, `conv3d`, transposed convolutions | yes | not yet |
| Grad transforms (`grad`, `vjp`, `jvp`, `valueAndGrad`) | yes | not yet |
| `vmap` | yes | not yet |
| `compile` | yes | not yet |
| `MLXCustomFunction` | yes | not yet |
| Memory / wired memory | yes | not yet |
| State / global stream management | yes | not yet |
| IO (safetensors, GGUF) | yes | not yet |

### MLXRandom (`Source/MLXRandom`)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `seed`, `key`, `split` | yes | not yet |
| `uniform` | yes | not yet |
| `normal`, `multivariateNormal` | yes | not yet |
| `randInt` | yes | not yet |
| `bernoulli` | yes | not yet |
| `truncatedNormal` | yes | not yet |
| `gumbel` | yes | not yet |
| `categorical` | partial (on `MLXArray`) | `categoricalSample` |
| `laplace` | yes | not yet |

### MLXNN (`Source/MLXNN`)

#### Infrastructure

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `Module` base class | yes | `Module` |
| Parameter tracking / `parameters()` | yes | `ParametersMixin` |
| `loadWeights` | yes | yes |
| `sanitize` | yes | yes |
| `WeightLoader` | — | `WeightLoader` |
| `Sequential` container | yes | not yet |
| `UnaryLayer` protocol | yes | not yet |
| Value-and-grad helpers | `ValueAndGrad` | not yet |

#### Layers

| Layer | mlx-swift | mlx_dart |
|---|---|---|
| `Linear` | yes | yes |
| `Embedding` | yes | yes |
| `RMSNorm` | yes | yes |
| `LayerNorm` | yes | yes |
| `Conv2d` | yes | yes |
| `GroupNorm` | yes | not yet |
| `BatchNorm` | yes | not yet |
| `InstanceNorm` | yes | not yet |
| `Conv1d` | yes | not yet |
| `Conv3d` | yes | not yet |
| `ConvTransposed1d/2d/3d` | yes | not yet |
| `Dropout`, `Dropout2d`, `Dropout3d` | yes | not yet |
| `MaxPool1d/2d/3d` | yes | not yet |
| `AvgPool1d/2d/3d` | yes | not yet |
| `MultiHeadAttention` | yes | not yet |
| `Transformer` (encoder + decoder) | yes | not yet |
| `SinusoidalPositionalEncoding` | yes | not yet |
| `RNN`, `GRU`, `LSTM` | yes | not yet |
| `Upsample` | yes | not yet |
| Quantized layers | yes | not yet |
| KV cache (`Cache`, `RotatingKVCache`) | yes | not yet |

#### Activations

Functional activations available on `MLXArray`: `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`, `softmax()`.

| Activation | mlx-swift | mlx_dart |
|---|---|---|
| `relu`, `gelu`, `silu`, `sigmoid`, `tanh` | yes | yes (on `MLXArray`) |
| `leakyRelu` | yes | not yet |
| `elu`, `celu` | yes | not yet |
| `relu6`, `reluSquared` | yes | not yet |
| `logSoftmax` | yes | not yet |
| `softPlus` / `softsign` | yes | not yet |
| `softshrink` | yes | not yet |
| Activation layer classes (`ReLU`, `GELU`, etc.) | yes | not yet |

#### Losses

| Loss | mlx-swift | mlx_dart |
|---|---|---|
| `crossEntropy` | yes | not yet |
| `binaryCrossEntropy` | yes | not yet |
| `l1Loss`, `mseLoss` | yes | not yet |
| `nllLoss`, `klDivLoss` | yes | not yet |
| `smoothL1Loss`, `huberLoss` | yes | not yet |
| `hingeLoss`, `tripletLoss` | yes | not yet |
| `logCoshLoss`, `cosineSimilarityLoss` | yes | not yet |

### MLXOptimizers (`Source/MLXOptimizers`)

| Optimizer | mlx-swift | mlx_dart |
|---|---|---|
| `SGD` | yes | not yet |
| `Adam`, `AdamW`, `Adamax` | yes | not yet |
| `RMSprop`, `AdaGrad`, `AdaDelta` | yes | not yet |
| `Lion`, `Adafactor` | yes | not yet |

### MLXLinalg (`Source/MLXLinalg`)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `norm` | yes | not yet |
| `svd`, `qr`, `lu` | yes | not yet |
| `inv`, `triInv`, `cholesky`, `choleskyInv` | yes | not yet |
| `solve`, `solveTriangular` | yes | not yet |
| `cross` | yes | not yet |

### MLXFFT (`Source/MLXFFT`)

| Feature | mlx-swift | mlx_dart |
|---|---|---|
| `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn` | yes | not yet |
| `rfft`, `irfft`, `rfft2`, `irfft2` | yes | not yet |

## Architecture

```
mlx_dart
├── lib/src/
│   ├── context.dart   — MLXContext, MLXException
│   ├── array.dart     — MLXArray, MLXDtype, free ops
│   ├── module.dart    — Module, ParametersMixin, WeightLoader
│   └── layers.dart    — Linear, Embedding, RMSNorm, LayerNorm, Conv2d
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
