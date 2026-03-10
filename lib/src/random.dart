import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mlx_c_ffi/mlx_c_ffi.dart';

import 'array.dart';
import 'context.dart';

/// Random number generation — mirrors `MLXRandom` in mlx-swift.
///
/// All functions accept an optional [key] array (PRNG key). When omitted, MLX
/// uses a zero-initialised key that defers to its global PRNG state.
abstract final class MLXRandom {
  MLXRandom._();

  /// Set the global random seed.
  static void seed(MLXContext ctx, int seed) {
    ctx.check(ctx.bindings.mlx_random_seed(seed), 'mlx_random_seed');
  }

  /// Create a PRNG key from an integer [seed].
  static MLXArray key(MLXContext ctx, int seed) {
    final ptr = calloc<mlx_array>();
    ctx.check(ctx.bindings.mlx_random_key(ptr, seed), 'mlx_random_key');
    return MLXArray.owned(ctx, ptr);
  }

  /// Split a PRNG key into two independent keys.
  static (MLXArray, MLXArray) split(MLXContext ctx, MLXArray key) {
    final out1 = calloc<mlx_array>();
    final out2 = calloc<mlx_array>();
    ctx.check(
        ctx.bindings.mlx_random_split(out1, out2, key.raw, ctx.stream),
        'mlx_random_split');
    return (MLXArray.owned(ctx, out1), MLXArray.owned(ctx, out2));
  }

  /// Uniform random values in `[low, high)` with given [shape].
  static MLXArray uniform(
    MLXContext ctx, {
    List<int> shape = const [1],
    double low = 0.0,
    double high = 1.0,
    MLXDtype dtype = MLXDtype.float32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final lowArr = MLXArray.float_(ctx, low);
    final highArr = MLXArray.float_(ctx, high);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_uniform(out, lowArr.raw, highArr.raw,
              shapePtr, shape.length, dtype.raw, keyArr.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
      lowArr.dispose();
      highArr.dispose();
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Normal (Gaussian) samples with given [shape], [loc], and [scale].
  static MLXArray normal(
    MLXContext ctx, {
    List<int> shape = const [1],
    double loc = 0.0,
    double scale = 1.0,
    MLXDtype dtype = MLXDtype.float32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_normal(out, shapePtr, shape.length,
              dtype.raw, loc, scale, keyArr.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Integer samples uniformly in `[low, high)`.
  static MLXArray randInt(
    MLXContext ctx, {
    required int low,
    required int high,
    List<int> shape = const [1],
    MLXDtype dtype = MLXDtype.int32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final lowArr = MLXArray.int_(ctx, low);
    final highArr = MLXArray.int_(ctx, high);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_randint(out, lowArr.raw, highArr.raw,
              shapePtr, shape.length, dtype.raw, keyArr.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
      lowArr.dispose();
      highArr.dispose();
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Bernoulli samples with success probability [p].
  ///
  /// [p] may be a scalar float or an array broadcastable to [shape].
  static MLXArray bernoulli(
    MLXContext ctx, {
    double p = 0.5,
    List<int> shape = const [1],
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final pArr = MLXArray.float_(ctx, p);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_bernoulli(
              out, pArr.raw, shapePtr, shape.length, keyArr.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
      pArr.dispose();
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Samples from a truncated normal distribution in `[lower, upper]`.
  static MLXArray truncatedNormal(
    MLXContext ctx, {
    double lower = -2.0,
    double upper = 2.0,
    List<int> shape = const [1],
    MLXDtype dtype = MLXDtype.float32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final lowerArr = MLXArray.float_(ctx, lower);
    final upperArr = MLXArray.float_(ctx, upper);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_truncated_normal(
              out,
              lowerArr.raw,
              upperArr.raw,
              shapePtr,
              shape.length,
              dtype.raw,
              keyArr.raw,
              ctx.stream));
    } finally {
      calloc.free(shapePtr);
      lowerArr.dispose();
      upperArr.dispose();
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Gumbel-distributed samples.
  static MLXArray gumbel(
    MLXContext ctx, {
    List<int> shape = const [1],
    MLXDtype dtype = MLXDtype.float32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_gumbel(
              out, shapePtr, shape.length, dtype.raw, keyArr.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Laplace-distributed samples with location [loc] and [scale].
  static MLXArray laplace(
    MLXContext ctx, {
    List<int> shape = const [1],
    double loc = 0.0,
    double scale = 1.0,
    MLXDtype dtype = MLXDtype.float32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_laplace(out, shapePtr, shape.length,
              dtype.raw, loc, scale, keyArr.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
      if (keyOwned) keyArr.dispose();
    }
  }

  /// Samples from a multivariate normal distribution.
  ///
  /// [mean] shape: `[D]`, [cov] shape: `[D, D]`.
  static MLXArray multivariateNormal(
    MLXContext ctx, {
    required MLXArray mean,
    required MLXArray cov,
    List<int> shape = const [1],
    MLXDtype dtype = MLXDtype.float32,
    MLXArray? key,
  }) {
    final shapePtr = _allocShapePtr(shape);
    final keyArr = key ?? _zeroKey(ctx);
    final keyOwned = key == null;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_random_multivariate_normal(
              out,
              mean.raw,
              cov.raw,
              shapePtr,
              shape.length,
              dtype.raw,
              keyArr.raw,
              ctx.stream));
    } finally {
      calloc.free(shapePtr);
      if (keyOwned) keyArr.dispose();
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  static MLXArray _zeroKey(MLXContext ctx) {
    final ptr = calloc<mlx_array>();
    return MLXArray.owned(ctx, ptr);
  }

  static MLXArray _opChecked(
    MLXContext ctx,
    int Function(ffi.Pointer<mlx_array>) fn,
  ) {
    final out = calloc<mlx_array>();
    final code = fn(out);
    if (code != 0) {
      ctx.bindings.mlx_array_free(out.ref);
      calloc.free(out);
      throw MLXException('mlx_random', code);
    }
    return MLXArray.owned(ctx, out);
  }

  static ffi.Pointer<ffi.Int> _allocShapePtr(List<int> shape) {
    final s = shape.isEmpty ? [0] : shape;
    final ptr = calloc<ffi.Int>(s.length);
    for (var i = 0; i < s.length; i++) {
      ptr[i] = s[i];
    }
    return ptr;
  }
}
