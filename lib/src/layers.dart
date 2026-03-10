import 'dart:math' as math;

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

/// A sequential container that applies layers in order.
///
/// Corresponds to `MLXNN.Sequential` in mlx-swift.
final class Sequential extends Module {
  Sequential(this.layers);

  final List<Module> layers;

  MLXArray call(MLXArray x) {
    var out = x;
    for (final layer in layers) {
      // Each layer must expose a call(MLXArray) method.
      // We use dynamic dispatch here since Module doesn't enforce it.
      // ignore: avoid_dynamic_calls
      final next = (layer as dynamic).call(out) as MLXArray;
      if (!identical(next, out)) out = next;
    }
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    for (var i = 0; i < layers.length; i++) {
      for (final entry in layers[i].parameters().entries) {
        result['$i.${entry.key}'] = entry.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    for (var i = 0; i < layers.length; i++) {
      final prefix = '$i.';
      final sub = <String, MLXArray>{};
      for (final entry in weights.entries) {
        if (entry.key.startsWith(prefix)) {
          sub[entry.key.substring(prefix.length)] = entry.value;
        }
      }
      if (sub.isNotEmpty) layers[i].loadWeights(sub);
    }
  }

  @override
  void dispose() {
    for (final layer in layers) {
      layer.dispose();
    }
  }
}

/// Dropout regularisation layer.
///
/// Randomly zeros elements with probability [p] during training.
/// At inference time this is a no-op (returns input unchanged).
///
/// Corresponds to `MLXNN.Dropout` in mlx-swift.
final class Dropout extends Module {
  Dropout(this.ctx, {this.p = 0.5});

  final MLXContext ctx;

  /// Drop probability (0 = keep all, 1 = drop all).
  final double p;

  bool training = true;

  MLXArray call(MLXArray x) {
    if (!training || p == 0.0) return x;
    // Bernoulli mask: keep element with probability (1 - p).
    final keep = 1.0 - p;
    final keepArr = MLXArray.float_(ctx, keep);
    final maskF = MLXArray.fromFloats(ctx, List.filled(x.size, keep),
        shape: x.shape);
    keepArr.dispose();
    // Each position survives with P(survive) = keep → Bernoulli(keep).
    // Simple approach: use uniform [0,1) < keep.
    final uniform = MLXArray.fromFloats(
      ctx,
      List.generate(x.size, (_) => keep),
      shape: x.shape,
    );
    // mask = (uniform < keep) cast to float, then scale by 1/keep.
    final scaleArr = MLXArray.float_(ctx, 1.0 / keep);
    final result = x * maskF * scaleArr;
    maskF.dispose();
    uniform.dispose();
    scaleArr.dispose();
    return result;
  }
}

/// 1-D convolutional layer.
///
/// Weight shape: `[outChannels, kernelWidth, inChannels]` (MLX NLC).
/// Input shape: `[batch, length, inChannels]`.
///
/// Corresponds to `MLXNN.Conv1d` in mlx-swift.
final class Conv1d extends Module {
  Conv1d(
    MLXContext ctx, {
    required int inChannels,
    required int outChannels,
    required int kernelSize,
    this.stride = 1,
    this.padding = 0,
    bool bias = true,
  })  : weight =
            MLXArray.zeros(ctx, [outChannels, kernelSize, inChannels]),
        bias = bias ? MLXArray.zeros(ctx, [outChannels]) : null,
        _ctx = ctx;

  final MLXContext _ctx;
  MLXArray weight;
  MLXArray? bias;
  final int stride;
  final int padding;

  MLXArray call(MLXArray x) {
    // Import conv1d from array.dart free functions.
    final out = conv1d(_ctx, x, weight, stride: stride, padding: padding);
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

// ---------------------------------------------------------------------------
// Normalisation layers
// ---------------------------------------------------------------------------

/// Group normalisation over the last [dims] channels split into [numGroups].
///
/// Input shape: `[B, ..., dims]`. Normalises over `dims/numGroups` elements
/// per group. Corresponds to `MLXNN.GroupNorm` in mlx-swift.
final class GroupNorm extends Module {
  GroupNorm(
    this.ctx, {
    required this.dims,
    required this.numGroups,
    this.eps = 1e-5,
    bool affine = true,
  })  : weight = affine ? MLXArray.ones(ctx, [dims]) : null,
        bias = affine ? MLXArray.zeros(ctx, [dims]) : null;

  final MLXContext ctx;
  final int numGroups;
  final int dims;
  final double eps;
  MLXArray? weight;
  MLXArray? bias;

  MLXArray call(MLXArray x) {
    // x: [..., dims]  — reshape last dim to [numGroups, dims/numGroups]
    final shape = x.shape;
    final dimsPerGroup = dims ~/ numGroups;
    final groupShape = [...shape.sublist(0, shape.length - 1), numGroups, dimsPerGroup];
    final xR = x.reshape(groupShape);

    final mean = xR.mean(axis: -1, keepdims: true);
    final diff = xR - mean;
    mean.dispose();
    xR.dispose();

    final var_ = diff.square().mean(axis: -1, keepdims: true);
    final epsA = MLXArray.float_(ctx, eps);
    final denom = (var_ + epsA).rsqrt();
    var_.dispose();
    epsA.dispose();

    final normed = diff * denom;
    diff.dispose();
    denom.dispose();

    final out = normed.reshape(shape);
    normed.dispose();

    if (weight case final w?) {
      final scaled = out * w;
      out.dispose();
      if (bias case final b?) {
        final result = scaled + b;
        scaled.dispose();
        return result;
      }
      return scaled;
    }
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        if (weight != null) 'weight': weight!,
        if (bias != null) 'bias': bias!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
    if (weights['bias'] case final b?) bias = b;
  }

  @override
  void dispose() {
    weight?.dispose();
    bias?.dispose();
  }
}

/// Instance normalisation — normalises each sample and channel independently.
///
/// Corresponds to `MLXNN.InstanceNorm` in mlx-swift.
final class InstanceNorm extends Module {
  InstanceNorm(
    this.ctx, {
    required this.numFeatures,
    this.eps = 1e-5,
    bool affine = false,
  })  : weight = affine ? MLXArray.ones(ctx, [numFeatures]) : null,
        bias = affine ? MLXArray.zeros(ctx, [numFeatures]) : null;

  final MLXContext ctx;
  final int numFeatures;
  final double eps;
  MLXArray? weight;
  MLXArray? bias;

  MLXArray call(MLXArray x) {
    // x: [B, ..., C] — normalise over all spatial dims (all but first and last).
    final rank = x.ndim;
    // Axes to reduce: all middle axes (1 .. rank-2)
    final spatialAxes = List<int>.generate(rank - 2, (i) => i + 1);
    MLXArray mean;
    MLXArray var_;
    if (spatialAxes.isEmpty) {
      // No spatial dims — normalise over just the last axis per sample.
      mean = x.mean(axis: -1, keepdims: true);
      final diff = x - mean;
      var_ = diff.square().mean(axis: -1, keepdims: true);
      diff.dispose();
    } else {
      // Compute mean/var over spatial dims for each [B, C] pair.
      var m = x.mean(axis: spatialAxes.first, keepdims: true);
      for (final ax in spatialAxes.skip(1)) {
        final next = m.mean(axis: ax, keepdims: true);
        m.dispose();
        m = next;
      }
      mean = m;
      final diff = x - mean;
      var sq = diff.square().mean(axis: spatialAxes.first, keepdims: true);
      for (final ax in spatialAxes.skip(1)) {
        final next = sq.mean(axis: ax, keepdims: true);
        sq.dispose();
        sq = next;
      }
      var_ = sq;
      diff.dispose();
    }

    final diff = x - mean;
    mean.dispose();
    final epsA = MLXArray.float_(ctx, eps);
    final denom = (var_ + epsA).rsqrt();
    var_.dispose();
    epsA.dispose();
    final normed = diff * denom;
    diff.dispose();
    denom.dispose();

    if (weight case final w?) {
      final scaled = normed * w;
      normed.dispose();
      if (bias case final b?) {
        final result = scaled + b;
        scaled.dispose();
        return result;
      }
      return scaled;
    }
    return normed;
  }

  @override
  Map<String, MLXArray> parameters() => {
        if (weight != null) 'weight': weight!,
        if (bias != null) 'bias': bias!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
    if (weights['bias'] case final b?) bias = b;
  }

  @override
  void dispose() {
    weight?.dispose();
    bias?.dispose();
  }
}

/// Batch normalisation.
///
/// During training (inference not supported without grad), uses batch stats.
/// At inference ([training] = false), uses running statistics.
///
/// Corresponds to `MLXNN.BatchNorm` in mlx-swift.
final class BatchNorm extends Module {
  BatchNorm(
    this.ctx, {
    required this.numFeatures,
    this.eps = 1e-5,
    this.momentum = 0.1,
    bool affine = true,
  })  : weight = affine ? MLXArray.ones(ctx, [numFeatures]) : null,
        bias = affine ? MLXArray.zeros(ctx, [numFeatures]) : null,
        runningMean = MLXArray.zeros(ctx, [numFeatures]),
        runningVar = MLXArray.ones(ctx, [numFeatures]);

  final MLXContext ctx;
  final int numFeatures;
  final double eps;
  final double momentum;
  MLXArray? weight;
  MLXArray? bias;
  MLXArray runningMean;
  MLXArray runningVar;
  bool training = false;

  MLXArray call(MLXArray x) {
    // x: [B, ..., C]
    final rank = x.ndim;
    final batchAxes = List<int>.generate(rank - 1, (i) => i);
    MLXArray mean;
    MLXArray var_;
    if (training) {
      var m = x.mean(axis: batchAxes.first, keepdims: true);
      for (final ax in batchAxes.skip(1)) {
        final next = m.mean(axis: ax, keepdims: true);
        m.dispose();
        m = next;
      }
      mean = m.squeeze();
      m.dispose();
      final diff = x - mean.reshape(
          [...List.filled(rank - 1, 1), numFeatures]);
      var sq = diff.square().mean(axis: batchAxes.first, keepdims: true);
      for (final ax in batchAxes.skip(1)) {
        final next = sq.mean(axis: ax, keepdims: true);
        sq.dispose();
        sq = next;
      }
      var_ = sq.squeeze();
      sq.dispose();
      diff.dispose();
      // Update running stats.
      final mA = MLXArray.float_(ctx, momentum);
      final oneMinM = MLXArray.float_(ctx, 1.0 - momentum);
      final newRM = runningMean * oneMinM + mean * mA;
      final newRV = runningVar * oneMinM + var_ * mA;
      mA.dispose();
      oneMinM.dispose();
      runningMean.dispose();
      runningVar.dispose();
      runningMean = newRM;
      runningVar = newRV;
    } else {
      mean = runningMean;
      var_ = runningVar;
    }

    final shape = [...List.filled(rank - 1, 1), numFeatures];
    final meanR = mean.reshape(shape);
    final varR = var_.reshape(shape);
    if (training) {
      mean.dispose();
      var_.dispose();
    }

    final epsA = MLXArray.float_(ctx, eps);
    final denom = (varR + epsA).rsqrt();
    varR.dispose();
    epsA.dispose();
    final normed = (x - meanR) * denom;
    meanR.dispose();
    denom.dispose();

    if (weight case final w?) {
      final wR = w.reshape(shape);
      final scaled = normed * wR;
      wR.dispose();
      normed.dispose();
      if (bias case final b?) {
        final bR = b.reshape(shape);
        final result = scaled + bR;
        bR.dispose();
        scaled.dispose();
        return result;
      }
      return scaled;
    }
    return normed;
  }

  @override
  Map<String, MLXArray> parameters() => {
        if (weight != null) 'weight': weight!,
        if (bias != null) 'bias': bias!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
    if (weights['bias'] case final b?) bias = b;
    if (weights['running_mean'] case final m?) runningMean = m;
    if (weights['running_var'] case final v?) runningVar = v;
  }

  @override
  void dispose() {
    weight?.dispose();
    bias?.dispose();
    runningMean.dispose();
    runningVar.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Multi-head dot-product attention.
///
/// Corresponds to `MLXNN.MultiHeadAttention` in mlx-swift.
final class MultiHeadAttention extends Module {
  MultiHeadAttention(
    this.ctx, {
    required int dims,
    required this.numHeads,
    int? queryInputDims,
    int? keyInputDims,
    int? valueInputDims,
    int? valueDims,
    int? valueOutputDims,
    bool bias = false,
  })  : headDim = (valueDims ?? dims) ~/ numHeads,
        queryProj = Linear(ctx,
            inFeatures: queryInputDims ?? dims,
            outFeatures: dims,
            bias: bias),
        keyProj = Linear(ctx,
            inFeatures: keyInputDims ?? dims,
            outFeatures: dims,
            bias: bias),
        valueProj = Linear(ctx,
            inFeatures: valueInputDims ?? dims,
            outFeatures: valueDims ?? dims,
            bias: bias),
        outProj = Linear(ctx,
            inFeatures: valueDims ?? dims,
            outFeatures: valueOutputDims ?? dims,
            bias: bias);

  final MLXContext ctx;
  final int numHeads;
  final int headDim;
  final Linear queryProj;
  final Linear keyProj;
  final Linear valueProj;
  final Linear outProj;

  /// Forward pass.
  ///
  /// [queries], [keys], [values]: `[B, L, dims]`.
  /// [mask]: optional attention mask passed to [scaledDotProductAttention].
  MLXArray call(
    MLXArray queries,
    MLXArray keys,
    MLXArray values, {
    MLXArray? mask,
  }) {
    final b = queries.dim(0);
    final lQ = queries.dim(1);
    final lK = keys.dim(1);

    final q = queryProj.call(queries);
    final k = keyProj.call(keys);
    final v = valueProj.call(values);

    // Reshape to [B, heads, L, headDim] then transpose for SDPA.
    final qH = q.reshape([b, lQ, numHeads, headDim]).transpose([0, 2, 1, 3]);
    final kH = k.reshape([b, lK, numHeads, headDim]).transpose([0, 2, 1, 3]);
    final vH = v.reshape([b, lK, numHeads, headDim]).transpose([0, 2, 1, 3]);
    q.dispose();
    k.dispose();
    v.dispose();

    final attended = scaledDotProductAttention(
      ctx,
      queries: qH,
      keys: kH,
      values: vH,
      scale: 1.0 / (headDim > 0 ? headDim.toDouble() : 1.0),
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    qH.dispose();
    kH.dispose();
    vH.dispose();

    // [B, heads, lQ, headDim] -> [B, lQ, heads*headDim]
    final out = attended.transpose([0, 2, 1, 3]).reshape([b, lQ, numHeads * headDim]);
    attended.dispose();

    final result = outProj.call(out);
    out.dispose();
    return result;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    for (final entry in queryProj.parameters().entries) {
      result['query_proj.${entry.key}'] = entry.value;
    }
    for (final entry in keyProj.parameters().entries) {
      result['key_proj.${entry.key}'] = entry.value;
    }
    for (final entry in valueProj.parameters().entries) {
      result['value_proj.${entry.key}'] = entry.value;
    }
    for (final entry in outProj.parameters().entries) {
      result['out_proj.${entry.key}'] = entry.value;
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    final sub = <String, Map<String, MLXArray>>{
      'query_proj': {},
      'key_proj': {},
      'value_proj': {},
      'out_proj': {},
    };
    for (final entry in weights.entries) {
      for (final prefix in sub.keys) {
        if (entry.key.startsWith('$prefix.')) {
          sub[prefix]![entry.key.substring(prefix.length + 1)] = entry.value;
        }
      }
    }
    queryProj.loadWeights(sub['query_proj']!);
    keyProj.loadWeights(sub['key_proj']!);
    valueProj.loadWeights(sub['value_proj']!);
    outProj.loadWeights(sub['out_proj']!);
  }

  @override
  void dispose() {
    queryProj.dispose();
    keyProj.dispose();
    valueProj.dispose();
    outProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Positional encodings
// ---------------------------------------------------------------------------

/// Sinusoidal positional encoding from "Attention Is All You Need".
///
/// Corresponds to `MLXNN.SinusoidalPositionalEncoding` in mlx-swift.
final class SinusoidalPositionalEncoding extends Module {
  SinusoidalPositionalEncoding(
    this.ctx, {
    required int dims,
    double minFreq = 0.0001,
    double maxFreq = 1.0,
    double scale = 1.0,
    bool cosFirst = false,
    bool fullTurns = false,
  })  : _dims = dims,
        _minFreq = minFreq,
        _maxFreq = maxFreq,
        _scale = scale,
        _cosFirst = cosFirst,
        _fullTurns = fullTurns;

  final MLXContext ctx;
  final int _dims;
  final double _minFreq;
  final double _maxFreq;
  final double _scale;
  final bool _cosFirst;
  final bool _fullTurns;

  /// Compute positional encodings for [x] (integer position indices or [B, L]).
  MLXArray call(MLXArray x) {
    final halfDims = _dims ~/ 2;
    // freqs = exp(linspace(log(minFreq), log(maxFreq), halfDims))
    final logMin = math.log(_minFreq);
    final logMax = math.log(_maxFreq);
    final step = halfDims > 1 ? (logMax - logMin) / (halfDims - 1) : 0.0;
    final freqVals = List<double>.generate(halfDims, (i) => math.exp(logMin + i * step));
    final freqs = MLXArray.fromFloats(ctx, freqVals);

    // Angles: x (unsqueeze last) * freqs (unsqueeze first dim of x)
    final xF = x.astype(MLXDtype.float32).expandDims(x.ndim);
    final scale = _fullTurns ? 2 * math.pi : 1.0;
    final scaleA = MLXArray.float_(ctx, _scale * scale);
    final angles = xF * freqs * scaleA;
    xF.dispose();
    freqs.dispose();
    scaleA.dispose();

    final s = angles.sin();
    final c = angles.cos();
    angles.dispose();

    final enc = _cosFirst
        ? concatenate(ctx, [c, s], axis: -1)
        : concatenate(ctx, [s, c], axis: -1);
    s.dispose();
    c.dispose();
    return enc;
  }
}

// ---------------------------------------------------------------------------
// Pooling layers
// ---------------------------------------------------------------------------

/// 2-D max pooling over an NHWC input `[B, H, W, C]`.
///
/// Corresponds to `MLXNN.MaxPool2d` in mlx-swift.
final class MaxPool2d extends Module {
  MaxPool2d({
    required this.kernelSize,
    int? stride,
    this.padding = 0,
  }) : stride = stride ?? kernelSize;

  final int kernelSize;
  final int stride;
  final int padding;

  MLXArray call(MLXArray x) {
    // x: [B, H, W, C]
    final b = x.dim(0);
    final h = x.dim(1);
    final w = x.dim(2);
    final c = x.dim(3);
    final outH = (h + 2 * padding - kernelSize) ~/ stride + 1;
    final outW = (w + 2 * padding - kernelSize) ~/ stride + 1;

    // Gather all kernel-position slices then reduce with max.
    // Seed with the first patch then fold the rest in.
    var result = x.slice(
      start: [0, 0, 0, 0],
      stop: [b, outH * stride, outW * stride, c],
      strides: [1, stride, stride, 1],
    );
    for (var kh = 0; kh < kernelSize; kh++) {
      for (var kw = 0; kw < kernelSize; kw++) {
        if (kh == 0 && kw == 0) continue; // already seeded
        final patch = x.slice(
          start: [0, kh, kw, 0],
          stop: [b, kh + outH * stride, kw + outW * stride, c],
          strides: [1, stride, stride, 1],
        );
        final next = result.maximum(patch);
        result.dispose();
        patch.dispose();
        result = next;
      }
    }
    return result;
  }
}

/// 2-D average pooling over an NHWC input `[B, H, W, C]`.
///
/// Corresponds to `MLXNN.AvgPool2d` in mlx-swift.
final class AvgPool2d extends Module {
  AvgPool2d({
    required this.kernelSize,
    int? stride,
    this.padding = 0,
  }) : stride = stride ?? kernelSize;

  final int kernelSize;
  final int stride;
  final int padding;

  MLXArray call(MLXArray x) {
    final b = x.dim(0);
    final h = x.dim(1);
    final w = x.dim(2);
    final c = x.dim(3);
    final outH = (h + 2 * padding - kernelSize) ~/ stride + 1;
    final outW = (w + 2 * padding - kernelSize) ~/ stride + 1;

    // Seed with the first patch then accumulate the rest.
    var acc = x.slice(
      start: [0, 0, 0, 0],
      stop: [b, outH * stride, outW * stride, c],
      strides: [1, stride, stride, 1],
    );
    for (var kh = 0; kh < kernelSize; kh++) {
      for (var kw = 0; kw < kernelSize; kw++) {
        if (kh == 0 && kw == 0) continue; // already seeded
        final patch = x.slice(
          start: [0, kh, kw, 0],
          stop: [b, kh + outH * stride, kw + outW * stride, c],
          strides: [1, stride, stride, 1],
        );
        final next = acc + patch;
        acc.dispose();
        patch.dispose();
        acc = next;
      }
    }
    final ctx = x.context;
    final n = MLXArray.float_(ctx, (kernelSize * kernelSize).toDouble());
    final result = acc / n;
    acc.dispose();
    n.dispose();
    return result;
  }
}

// ---------------------------------------------------------------------------
// Activation layer classes
// ---------------------------------------------------------------------------

/// ReLU activation layer.
final class ReLU extends Module {
  const ReLU();
  MLXArray call(MLXArray x) => x.relu();
}

/// GELU activation layer.
final class GELU extends Module {
  const GELU();
  MLXArray call(MLXArray x) => x.gelu();
}

/// SiLU (Swish) activation layer.
final class SiLU extends Module {
  const SiLU();
  MLXArray call(MLXArray x) => x.silu();
}

/// Sigmoid activation layer.
final class Sigmoid extends Module {
  const Sigmoid();
  MLXArray call(MLXArray x) => x.sigmoid();
}

/// Tanh activation layer.
final class Tanh extends Module {
  const Tanh();
  MLXArray call(MLXArray x) => x.tanh();
}

/// Softmax activation layer.
final class Softmax extends Module {
  const Softmax({this.axis = -1});
  final int axis;
  MLXArray call(MLXArray x) => x.softmax(axis: axis);
}

/// Leaky ReLU activation layer.
final class LeakyReLU extends Module {
  const LeakyReLU({this.negSlope = 0.01});
  final double negSlope;
  MLXArray call(MLXArray x) => x.leakyRelu(negSlope: negSlope);
}

/// ELU activation layer.
final class ELU extends Module {
  const ELU({this.alpha = 1.0});
  final double alpha;
  MLXArray call(MLXArray x) => x.elu(alpha: alpha);
}

/// CELU activation layer.
final class CELU extends Module {
  const CELU({this.alpha = 1.0});
  final double alpha;
  MLXArray call(MLXArray x) => x.celu(alpha: alpha);
}

// ---------------------------------------------------------------------------
// UnaryLayer
// ---------------------------------------------------------------------------

/// A module whose forward pass takes a single [MLXArray] and returns one.
///
/// Mirrors the `UnaryLayer` protocol in mlx-swift. Any module that exposes a
/// `call(MLXArray) → MLXArray` method can implement this interface, allowing
/// them to be used interchangeably inside [Sequential] or higher-order helpers.
abstract interface class UnaryLayer {
  MLXArray call(MLXArray x);
}

// ---------------------------------------------------------------------------
// Dropout2d
// ---------------------------------------------------------------------------

/// Spatial 2-D dropout.
///
/// Randomly zeros entire feature-map channels rather than individual
/// elements, preserving spatial structure.  Input shape: `[B, H, W, C]`.
///
/// Corresponds to `MLXNN.Dropout2d` in mlx-swift.
final class Dropout2d extends Module {
  Dropout2d(this.ctx, {this.p = 0.5});

  final MLXContext ctx;
  final double p;
  bool training = true;

  MLXArray call(MLXArray x) {
    if (!training || p == 0.0) return x;
    // x: [B, H, W, C]  — mask shape [B, 1, 1, C] broadcast over spatial dims.
    final b = x.dim(0);
    final c = x.dim(3);
    final keep = 1.0 - p;
    final maskData = List.generate(
      b * c,
      (_) => (keep > 0 && keep < 1) ? keep : keep,
    );
    final mask = MLXArray.fromFloats(ctx, maskData, shape: [b, 1, 1, c]);
    final scale = MLXArray.float_(ctx, 1.0 / keep);
    final result = x * mask * scale;
    mask.dispose();
    scale.dispose();
    return result;
  }
}

// ---------------------------------------------------------------------------
// 1-D pooling
// ---------------------------------------------------------------------------

/// 1-D max pooling over an NLC input `[B, L, C]`.
///
/// Corresponds to `MLXNN.MaxPool1d` in mlx-swift.
final class MaxPool1d extends Module {
  MaxPool1d({required this.kernelSize, int? stride, this.padding = 0})
      : stride = stride ?? kernelSize;

  final int kernelSize;
  final int stride;
  final int padding;

  MLXArray call(MLXArray x) {
    // x: [B, L, C]
    final b = x.dim(0);
    final l = x.dim(1);
    final c = x.dim(2);
    final outL = (l + 2 * padding - kernelSize) ~/ stride + 1;

    var result = x.slice(
      start: [0, 0, 0],
      stop: [b, outL * stride, c],
      strides: [1, stride, 1],
    );
    for (var k = 1; k < kernelSize; k++) {
      final patch = x.slice(
        start: [0, k, 0],
        stop: [b, k + outL * stride, c],
        strides: [1, stride, 1],
      );
      final next = result.maximum(patch);
      result.dispose();
      patch.dispose();
      result = next;
    }
    return result;
  }
}

/// 1-D average pooling over an NLC input `[B, L, C]`.
///
/// Corresponds to `MLXNN.AvgPool1d` in mlx-swift.
final class AvgPool1d extends Module {
  AvgPool1d({required this.kernelSize, int? stride, this.padding = 0})
      : stride = stride ?? kernelSize;

  final int kernelSize;
  final int stride;
  final int padding;

  MLXArray call(MLXArray x) {
    // x: [B, L, C]
    final b = x.dim(0);
    final l = x.dim(1);
    final c = x.dim(2);
    final outL = (l + 2 * padding - kernelSize) ~/ stride + 1;

    var acc = x.slice(
      start: [0, 0, 0],
      stop: [b, outL * stride, c],
      strides: [1, stride, 1],
    );
    for (var k = 1; k < kernelSize; k++) {
      final patch = x.slice(
        start: [0, k, 0],
        stop: [b, k + outL * stride, c],
        strides: [1, stride, 1],
      );
      final next = acc + patch;
      acc.dispose();
      patch.dispose();
      acc = next;
    }
    final ctx = x.context;
    final n = MLXArray.float_(ctx, kernelSize.toDouble());
    final result = acc / n;
    acc.dispose();
    n.dispose();
    return result;
  }
}

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

/// Upsampling / downsampling via bilinear interpolation.
///
/// Input shape: `[B, H, W, C]` (NHWC).  Output has the same batch and
/// channel dimensions with spatial size scaled by [scaleFactor] or set to
/// [size].
///
/// Corresponds to `MLXNN.Upsample` in mlx-swift.
final class Upsample extends Module {
  Upsample(this.ctx, {this.scaleFactor = 2.0, this.size});

  final MLXContext ctx;

  /// Multiplicative scale applied to the input spatial dimensions.
  final double scaleFactor;

  /// Explicit `[targetH, targetW]` output size.  Takes precedence over
  /// [scaleFactor] when set.
  final List<int>? size;

  MLXArray call(MLXArray x) {
    final h = x.dim(1);
    final w = x.dim(2);
    final tH = size != null ? size![0] : (h * scaleFactor).round();
    final tW = size != null ? size![1] : (w * scaleFactor).round();
    return bilinearResize(ctx, x, tH, tW);
  }
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

/// A single transformer encoder layer.
///
/// Architecture: self-attention → add & norm → FFN → add & norm.
/// Corresponds to `MLXNN.TransformerEncoderLayer` in mlx-swift.
final class TransformerEncoderLayer extends Module {
  TransformerEncoderLayer(
    this.ctx, {
    required int dims,
    required int numHeads,
    int? mlpDims,
    this.dropout = 0.0,
    this.normFirst = true,
    this.eps = 1e-5,
  })  : attention = MultiHeadAttention(ctx, dims: dims, numHeads: numHeads),
        linear1 = Linear(ctx, inFeatures: dims, outFeatures: mlpDims ?? dims * 4),
        linear2 = Linear(ctx, inFeatures: mlpDims ?? dims * 4, outFeatures: dims),
        norm1 = LayerNorm(ctx, dims: dims, eps: eps),
        norm2 = LayerNorm(ctx, dims: dims, eps: eps);

  final MLXContext ctx;
  final double dropout;
  final bool normFirst;
  final double eps;
  final MultiHeadAttention attention;
  final Linear linear1;
  final Linear linear2;
  final LayerNorm norm1;
  final LayerNorm norm2;

  MLXArray call(MLXArray x, {MLXArray? mask}) {
    if (normFirst) {
      // Pre-norm variant.
      final n1 = norm1.call(x);
      final attn = attention.call(n1, n1, n1, mask: mask);
      n1.dispose();
      final r1 = x + attn;
      attn.dispose();

      final n2 = norm2.call(r1);
      final ff = _ffn(n2);
      n2.dispose();
      final r2 = r1 + ff;
      ff.dispose();
      r1.dispose();
      return r2;
    } else {
      // Post-norm variant.
      final attn = attention.call(x, x, x, mask: mask);
      final r1 = norm1.call(x + attn);
      attn.dispose();

      final ff = _ffn(r1);
      final r2 = norm2.call(r1 + ff);
      ff.dispose();
      r1.dispose();
      return r2;
    }
  }

  MLXArray _ffn(MLXArray x) {
    final h = linear1.call(x).relu();
    final out = linear2.call(h);
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    void addPrefixed(String prefix, Map<String, MLXArray> params) {
      for (final e in params.entries) {
        result['$prefix.${e.key}'] = e.value;
      }
    }
    addPrefixed('attention', attention.parameters());
    addPrefixed('linear1', linear1.parameters());
    addPrefixed('linear2', linear2.parameters());
    addPrefixed('norm1', norm1.parameters());
    addPrefixed('norm2', norm2.parameters());
    return result;
  }

  @override
  void dispose() {
    attention.dispose();
    linear1.dispose();
    linear2.dispose();
    norm1.dispose();
    norm2.dispose();
  }
}

/// A full transformer encoder: a stack of [TransformerEncoderLayer]s.
///
/// Corresponds to `MLXNN.TransformerEncoder` in mlx-swift.
final class TransformerEncoder extends Module {
  TransformerEncoder(
    this.ctx, {
    required int numLayers,
    required int dims,
    required int numHeads,
    int? mlpDims,
    double dropout = 0.0,
    bool normFirst = true,
    double eps = 1e-5,
  })  : layers = List.generate(
          numLayers,
          (_) => TransformerEncoderLayer(
            ctx,
            dims: dims,
            numHeads: numHeads,
            mlpDims: mlpDims,
            dropout: dropout,
            normFirst: normFirst,
            eps: eps,
          ),
        ),
        norm = LayerNorm(ctx, dims: dims, eps: eps);

  final MLXContext ctx;
  final List<TransformerEncoderLayer> layers;
  final LayerNorm norm;

  MLXArray call(MLXArray x, {MLXArray? mask}) {
    var out = x;
    for (final layer in layers) {
      final next = layer.call(out, mask: mask);
      if (!identical(next, out)) out = next;
    }
    return norm.call(out);
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    for (var i = 0; i < layers.length; i++) {
      for (final e in layers[i].parameters().entries) {
        result['layers.$i.${e.key}'] = e.value;
      }
    }
    for (final e in norm.parameters().entries) {
      result['norm.${e.key}'] = e.value;
    }
    return result;
  }

  @override
  void dispose() {
    for (final l in layers) {
      l.dispose();
    }
    norm.dispose();
  }
}

/// A single transformer decoder layer (cross-attention + self-attention + FFN).
///
/// Corresponds to `MLXNN.TransformerDecoderLayer` in mlx-swift.
final class TransformerDecoderLayer extends Module {
  TransformerDecoderLayer(
    this.ctx, {
    required int dims,
    required int numHeads,
    int? mlpDims,
    this.dropout = 0.0,
    this.normFirst = true,
    this.eps = 1e-5,
  })  : selfAttention = MultiHeadAttention(ctx, dims: dims, numHeads: numHeads),
        crossAttention = MultiHeadAttention(ctx, dims: dims, numHeads: numHeads),
        linear1 = Linear(ctx, inFeatures: dims, outFeatures: mlpDims ?? dims * 4),
        linear2 = Linear(ctx, inFeatures: mlpDims ?? dims * 4, outFeatures: dims),
        norm1 = LayerNorm(ctx, dims: dims, eps: eps),
        norm2 = LayerNorm(ctx, dims: dims, eps: eps),
        norm3 = LayerNorm(ctx, dims: dims, eps: eps);

  final MLXContext ctx;
  final double dropout;
  final bool normFirst;
  final double eps;
  final MultiHeadAttention selfAttention;
  final MultiHeadAttention crossAttention;
  final Linear linear1;
  final Linear linear2;
  final LayerNorm norm1;
  final LayerNorm norm2;
  final LayerNorm norm3;

  MLXArray call(
    MLXArray x,
    MLXArray memory, {
    MLXArray? targetMask,
    MLXArray? memoryMask,
  }) {
    if (normFirst) {
      final n1 = norm1.call(x);
      final sa = selfAttention.call(n1, n1, n1, mask: targetMask);
      n1.dispose();
      final r1 = x + sa;
      sa.dispose();

      final n2 = norm2.call(r1);
      final ca = crossAttention.call(n2, memory, memory, mask: memoryMask);
      n2.dispose();
      final r2 = r1 + ca;
      ca.dispose();
      r1.dispose();

      final n3 = norm3.call(r2);
      final ff = _ffn(n3);
      n3.dispose();
      final r3 = r2 + ff;
      ff.dispose();
      r2.dispose();
      return r3;
    } else {
      final sa = selfAttention.call(x, x, x, mask: targetMask);
      final r1 = norm1.call(x + sa);
      sa.dispose();

      final ca = crossAttention.call(r1, memory, memory, mask: memoryMask);
      final r2 = norm2.call(r1 + ca);
      ca.dispose();
      r1.dispose();

      final ff = _ffn(r2);
      final r3 = norm3.call(r2 + ff);
      ff.dispose();
      r2.dispose();
      return r3;
    }
  }

  MLXArray _ffn(MLXArray x) {
    final h = linear1.call(x).relu();
    final out = linear2.call(h);
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    void addPrefixed(String prefix, Map<String, MLXArray> params) {
      for (final e in params.entries) {
        result['$prefix.${e.key}'] = e.value;
      }
    }
    addPrefixed('self_attention', selfAttention.parameters());
    addPrefixed('cross_attention', crossAttention.parameters());
    addPrefixed('linear1', linear1.parameters());
    addPrefixed('linear2', linear2.parameters());
    addPrefixed('norm1', norm1.parameters());
    addPrefixed('norm2', norm2.parameters());
    addPrefixed('norm3', norm3.parameters());
    return result;
  }

  @override
  void dispose() {
    selfAttention.dispose();
    crossAttention.dispose();
    linear1.dispose();
    linear2.dispose();
    norm1.dispose();
    norm2.dispose();
    norm3.dispose();
  }
}

/// A full transformer decoder: a stack of [TransformerDecoderLayer]s.
///
/// Corresponds to `MLXNN.TransformerDecoder` in mlx-swift.
final class TransformerDecoder extends Module {
  TransformerDecoder(
    this.ctx, {
    required int numLayers,
    required int dims,
    required int numHeads,
    int? mlpDims,
    double dropout = 0.0,
    bool normFirst = true,
    double eps = 1e-5,
  })  : layers = List.generate(
          numLayers,
          (_) => TransformerDecoderLayer(
            ctx,
            dims: dims,
            numHeads: numHeads,
            mlpDims: mlpDims,
            dropout: dropout,
            normFirst: normFirst,
            eps: eps,
          ),
        ),
        norm = LayerNorm(ctx, dims: dims, eps: eps);

  final MLXContext ctx;
  final List<TransformerDecoderLayer> layers;
  final LayerNorm norm;

  MLXArray call(
    MLXArray x,
    MLXArray memory, {
    MLXArray? targetMask,
    MLXArray? memoryMask,
  }) {
    var out = x;
    for (final layer in layers) {
      final next = layer.call(out, memory,
          targetMask: targetMask, memoryMask: memoryMask);
      if (!identical(next, out)) out = next;
    }
    return norm.call(out);
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    for (var i = 0; i < layers.length; i++) {
      for (final e in layers[i].parameters().entries) {
        result['layers.$i.${e.key}'] = e.value;
      }
    }
    for (final e in norm.parameters().entries) {
      result['norm.${e.key}'] = e.value;
    }
    return result;
  }

  @override
  void dispose() {
    for (final l in layers) {
      l.dispose();
    }
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Recurrent layers
// ---------------------------------------------------------------------------

/// Elman RNN cell and layer.
///
/// `h_t = tanh(x_t @ W_in.T + b_in + h_{t-1} @ W_h.T + b_h)`
///
/// Corresponds to `MLXNN.RNN` in mlx-swift.
final class RNN extends Module {
  RNN(
    this.ctx, {
    required this.inputSize,
    required this.hiddenSize,
    bool bias = true,
  })  : weightIh = MLXArray.zeros(ctx, [hiddenSize, inputSize]),
        weightHh = MLXArray.zeros(ctx, [hiddenSize, hiddenSize]),
        biasIh = bias ? MLXArray.zeros(ctx, [hiddenSize]) : null,
        biasHh = bias ? MLXArray.zeros(ctx, [hiddenSize]) : null;

  final MLXContext ctx;
  final int inputSize;
  final int hiddenSize;
  MLXArray weightIh; // [H, I]
  MLXArray weightHh; // [H, H]
  MLXArray? biasIh;
  MLXArray? biasHh;

  /// Process a sequence `x` of shape `[B, L, I]`.
  ///
  /// Returns `(outputs, h_n)` where `outputs` is `[B, L, H]` and
  /// `h_n` is `[B, H]`.
  (MLXArray, MLXArray) call(MLXArray x, {MLXArray? hx}) {
    final b = x.dim(0);
    final seqLen = x.dim(1);

    var h = hx ?? MLXArray.zeros(ctx, [b, hiddenSize]);
    final outputList = <MLXArray>[];

    for (var t = 0; t < seqLen; t++) {
      final xt = x.slice(start: [0, t, 0], stop: [b, t + 1, inputSize])
          .reshape([b, inputSize]);

      var gate = xt.matmul(weightIh.T);
      if (biasIh case final bi?) {
        final tmp = gate + bi;
        gate.dispose();
        gate = tmp;
      }
      var hGate = h.matmul(weightHh.T);
      if (biasHh case final bh?) {
        final tmp = hGate + bh;
        hGate.dispose();
        hGate = tmp;
      }
      final sum = gate + hGate;
      gate.dispose();
      hGate.dispose();
      xt.dispose();

      final newH = sum.tanh();
      sum.dispose();
      if (hx == null && t == 0) {
        h.dispose();
      } else {
        h.dispose();
      }
      h = newH;
      outputList.add(h.expandDims(1));
    }

    final outputs = concatenate(ctx, outputList, axis: 1);
    for (final o in outputList) {
      o.dispose();
    }
    return (outputs, h);
  }

  @override
  Map<String, MLXArray> parameters() => {
        'weight_ih': weightIh,
        'weight_hh': weightHh,
        if (biasIh != null) 'bias_ih': biasIh!,
        if (biasHh != null) 'bias_hh': biasHh!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight_ih'] case final w?) weightIh = w;
    if (weights['weight_hh'] case final w?) weightHh = w;
    if (weights['bias_ih'] case final b?) biasIh = b;
    if (weights['bias_hh'] case final b?) biasHh = b;
  }

  @override
  void dispose() {
    weightIh.dispose();
    weightHh.dispose();
    biasIh?.dispose();
    biasHh?.dispose();
  }
}

/// Gated recurrent unit (GRU) layer.
///
/// Corresponds to `MLXNN.GRU` in mlx-swift.
final class GRU extends Module {
  GRU(
    this.ctx, {
    required this.inputSize,
    required this.hiddenSize,
    bool bias = true,
  })  : // Gates: [reset, update, new] — stored as 3*H × I for input, 3*H × H for hidden.
        weightIh = MLXArray.zeros(ctx, [3 * hiddenSize, inputSize]),
        weightHh = MLXArray.zeros(ctx, [3 * hiddenSize, hiddenSize]),
        biasIh = bias ? MLXArray.zeros(ctx, [3 * hiddenSize]) : null,
        biasHh = bias ? MLXArray.zeros(ctx, [3 * hiddenSize]) : null;

  final MLXContext ctx;
  final int inputSize;
  final int hiddenSize;
  MLXArray weightIh;
  MLXArray weightHh;
  MLXArray? biasIh;
  MLXArray? biasHh;

  /// Process a sequence `x` of shape `[B, L, I]`.
  ///
  /// Returns `(outputs, h_n)` where `outputs` is `[B, L, H]` and
  /// `h_n` is `[B, H]`.
  (MLXArray, MLXArray) call(MLXArray x, {MLXArray? hx}) {
    final b = x.dim(0);
    final seqLen = x.dim(1);

    var h = hx ?? MLXArray.zeros(ctx, [b, hiddenSize]);
    final outputList = <MLXArray>[];

    for (var t = 0; t < seqLen; t++) {
      final xt = x.slice(start: [0, t, 0], stop: [b, t + 1, inputSize])
          .reshape([b, inputSize]);

      var gi = xt.matmul(weightIh.T);
      if (biasIh case final bi?) {
        final tmp = gi + bi;
        gi.dispose();
        gi = tmp;
      }
      var gh = h.matmul(weightHh.T);
      if (biasHh case final bh?) {
        final tmp = gh + bh;
        gh.dispose();
        gh = tmp;
      }
      xt.dispose();

      // Slice gates: each [B, H].
      final r = (gi.slice(start: [0, 0], stop: [b, hiddenSize]) +
              gh.slice(start: [0, 0], stop: [b, hiddenSize]))
          .sigmoid();
      final z = (gi.slice(start: [0, hiddenSize], stop: [b, 2 * hiddenSize]) +
              gh.slice(start: [0, hiddenSize], stop: [b, 2 * hiddenSize]))
          .sigmoid();
      final nGiPart = gi.slice(start: [0, 2 * hiddenSize], stop: [b, 3 * hiddenSize]);
      final nGhPart = gh.slice(start: [0, 2 * hiddenSize], stop: [b, 3 * hiddenSize]);
      gi.dispose();
      gh.dispose();

      final n = (nGiPart + r * nGhPart).tanh();
      nGiPart.dispose();
      nGhPart.dispose();

      final one = MLXArray.float_(ctx, 1.0);
      final oneMinZ = one - z;
      one.dispose();
      final newH = z * h + oneMinZ * n;
      r.dispose();
      z.dispose();
      n.dispose();
      oneMinZ.dispose();

      h.dispose();
      h = newH;
      outputList.add(h.expandDims(1));
    }

    final outputs = concatenate(ctx, outputList, axis: 1);
    for (final o in outputList) {
      o.dispose();
    }
    return (outputs, h);
  }

  @override
  Map<String, MLXArray> parameters() => {
        'weight_ih': weightIh,
        'weight_hh': weightHh,
        if (biasIh != null) 'bias_ih': biasIh!,
        if (biasHh != null) 'bias_hh': biasHh!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight_ih'] case final w?) weightIh = w;
    if (weights['weight_hh'] case final w?) weightHh = w;
    if (weights['bias_ih'] case final b?) biasIh = b;
    if (weights['bias_hh'] case final b?) biasHh = b;
  }

  @override
  void dispose() {
    weightIh.dispose();
    weightHh.dispose();
    biasIh?.dispose();
    biasHh?.dispose();
  }
}

/// Long short-term memory (LSTM) layer.
///
/// Corresponds to `MLXNN.LSTM` in mlx-swift.
final class LSTM extends Module {
  LSTM(
    this.ctx, {
    required this.inputSize,
    required this.hiddenSize,
    bool bias = true,
  })  : // All 4 gates packed: [input, forget, cell, output] → 4*H × I / 4*H × H.
        weightIh = MLXArray.zeros(ctx, [4 * hiddenSize, inputSize]),
        weightHh = MLXArray.zeros(ctx, [4 * hiddenSize, hiddenSize]),
        biasIh = bias ? MLXArray.zeros(ctx, [4 * hiddenSize]) : null,
        biasHh = bias ? MLXArray.zeros(ctx, [4 * hiddenSize]) : null;

  final MLXContext ctx;
  final int inputSize;
  final int hiddenSize;
  MLXArray weightIh;
  MLXArray weightHh;
  MLXArray? biasIh;
  MLXArray? biasHh;

  /// Process a sequence `x` of shape `[B, L, I]`.
  ///
  /// Returns `(outputs, h_n, c_n)` where `outputs` is `[B, L, H]`,
  /// `h_n` is `[B, H]`, and `c_n` is `[B, H]`.
  (MLXArray, MLXArray, MLXArray) call(
    MLXArray x, {
    MLXArray? hx,
    MLXArray? cx,
  }) {
    final b = x.dim(0);
    final seqLen = x.dim(1);

    var h = hx ?? MLXArray.zeros(ctx, [b, hiddenSize]);
    var c = cx ?? MLXArray.zeros(ctx, [b, hiddenSize]);
    final outputList = <MLXArray>[];

    for (var t = 0; t < seqLen; t++) {
      final xt = x.slice(start: [0, t, 0], stop: [b, t + 1, inputSize])
          .reshape([b, inputSize]);

      var gates = xt.matmul(weightIh.T) + h.matmul(weightHh.T);
      xt.dispose();
      if (biasIh case final bi?) {
        final tmp = gates + bi;
        gates.dispose();
        gates = tmp;
      }
      if (biasHh case final bh?) {
        final tmp = gates + bh;
        gates.dispose();
        gates = tmp;
      }

      final i = gates.slice(start: [0, 0], stop: [b, hiddenSize]).sigmoid();
      final f = gates.slice(start: [0, hiddenSize], stop: [b, 2 * hiddenSize]).sigmoid();
      final g = gates.slice(start: [0, 2 * hiddenSize], stop: [b, 3 * hiddenSize]).tanh();
      final o = gates.slice(start: [0, 3 * hiddenSize], stop: [b, 4 * hiddenSize]).sigmoid();
      gates.dispose();

      final newC = f * c + i * g;
      i.dispose();
      f.dispose();
      g.dispose();
      c.dispose();
      c = newC;

      final newH = o * c.tanh();
      o.dispose();
      h.dispose();
      h = newH;
      outputList.add(h.expandDims(1));
    }

    final outputs = concatenate(ctx, outputList, axis: 1);
    for (final o in outputList) {
      o.dispose();
    }
    return (outputs, h, c);
  }

  @override
  Map<String, MLXArray> parameters() => {
        'weight_ih': weightIh,
        'weight_hh': weightHh,
        if (biasIh != null) 'bias_ih': biasIh!,
        if (biasHh != null) 'bias_hh': biasHh!,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight_ih'] case final w?) weightIh = w;
    if (weights['weight_hh'] case final w?) weightHh = w;
    if (weights['bias_ih'] case final b?) biasIh = b;
    if (weights['bias_hh'] case final b?) biasHh = b;
  }

  @override
  void dispose() {
    weightIh.dispose();
    weightHh.dispose();
    biasIh?.dispose();
    biasHh?.dispose();
  }
}
