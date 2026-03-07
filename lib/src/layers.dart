import 'array.dart';
import 'context.dart';
import 'module.dart';

/// A fully-connected (dense) linear layer: `y = x @ weight.T + bias`.
///
/// Corresponds to `MLXNN.Linear` in mlx-swift.
final class Linear extends Module {
  Linear(this.ctx, {required int inFeatures, required int outFeatures, bool bias = true})
      : weight = MLXArray.zeros(ctx, [outFeatures, inFeatures]),
        bias = bias ? MLXArray.zeros(ctx, [outFeatures]) : null;

  final MLXContext ctx;
  MLXArray weight;
  MLXArray? bias;

  MLXArray call(MLXArray x) {
    final out = x.matmul(weight.T);
    if (bias case final b?) {
      final result = out + b;
      out.dispose();
      return result;
    }
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    return {
      'weight': weight,
      if (bias != null) 'bias': bias!,
    };
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
    if (weights['bias'] case final b?) bias = b;
  }

  @override
  void dispose() {
    weight.dispose();
    bias?.dispose();
  }
}

/// Token embedding table: maps integer token IDs to dense vectors.
///
/// Corresponds to `MLXNN.Embedding` in mlx-swift.
final class Embedding extends Module {
  Embedding(this.ctx, {required int vocabSize, required int dims})
      : weight = MLXArray.zeros(ctx, [vocabSize, dims]);

  final MLXContext ctx;
  MLXArray weight;

  MLXArray call(MLXArray indices) => weight.take(indices, axis: 0);

  @override
  Map<String, MLXArray> parameters() => {'weight': weight};

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
  }

  @override
  void dispose() => weight.dispose();
}

/// Root mean square layer normalisation.
///
/// Corresponds to `MLXNN.RMSNorm` (backed by `mlx_fast_rms_norm`) in mlx-swift.
final class RMSNorm extends Module {
  RMSNorm(this.ctx, {required int dims, this.eps = 1e-5})
      : weight = MLXArray.ones(ctx, [dims]);

  final MLXContext ctx;
  MLXArray weight;
  final double eps;

  MLXArray call(MLXArray x) => x.rmsNorm(weight, eps: eps);

  @override
  Map<String, MLXArray> parameters() => {'weight': weight};

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
  }

  @override
  void dispose() => weight.dispose();
}

/// 2-D convolutional layer.
///
/// Weight shape: `[outChannels, kernelH, kernelW, inChannels]` (MLX NHWC).
/// Input shape: `[batch, H, W, inChannels]`.
///
/// Corresponds to `MLXNN.Conv2d` in mlx-swift.
final class Conv2d extends Module {
  Conv2d(
    MLXContext ctx, {
    required int inChannels,
    required int outChannels,
    required int kernelSize,
    this.stride = 1,
    this.padding = 0,
    bool bias = true,
  })  : weight = MLXArray.zeros(
            ctx, [outChannels, kernelSize, kernelSize, inChannels]),
        bias = bias ? MLXArray.zeros(ctx, [outChannels]) : null,
        _ctx = ctx;

  final MLXContext _ctx;
  MLXArray weight; // [outC, kH, kW, inC]
  MLXArray? bias;
  final int stride;
  final int padding;

  MLXArray call(MLXArray x) {
    final out = conv2d(_ctx, x, weight, stride: stride, padding: padding);
    if (bias case final b?) {
      final result = out + b;
      out.dispose();
      return result;
    }
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        'weight': weight,
        if (bias != null) 'bias': bias!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
    if (weights['bias'] case final b?) bias = b;
  }

  @override
  void dispose() {
    weight.dispose();
    bias?.dispose();
  }
}

/// Standard layer normalisation.
///
/// Corresponds to `MLXNN.LayerNorm` in mlx-swift.
final class LayerNorm extends Module {
  LayerNorm(this.ctx, {required int dims, this.eps = 1e-5, bool bias = true})
      : weight = MLXArray.ones(ctx, [dims]),
        bias = bias ? MLXArray.zeros(ctx, [dims]) : null;

  final MLXContext ctx;
  MLXArray weight;
  MLXArray? bias;
  final double eps;

  MLXArray call(MLXArray x) => x.layerNorm(weight, bias, eps: eps);

  @override
  Map<String, MLXArray> parameters() => {
        'weight': weight,
        if (bias != null) 'bias': bias!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
    if (weights['bias'] case final b?) bias = b;
  }

  @override
  void dispose() {
    weight.dispose();
    bias?.dispose();
  }
}
