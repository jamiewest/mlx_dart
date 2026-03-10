import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:mlx_c_ffi/mlx_c_ffi.dart';

import 'context.dart';

/// Data type of an [MLXArray].
enum MLXDtype {
  bool_(mlx_dtype_.MLX_BOOL),
  uint8(mlx_dtype_.MLX_UINT8),
  uint16(mlx_dtype_.MLX_UINT16),
  uint32(mlx_dtype_.MLX_UINT32),
  uint64(mlx_dtype_.MLX_UINT64),
  int8(mlx_dtype_.MLX_INT8),
  int16(mlx_dtype_.MLX_INT16),
  int32(mlx_dtype_.MLX_INT32),
  int64(mlx_dtype_.MLX_INT64),
  float16(mlx_dtype_.MLX_FLOAT16),
  bfloat16(mlx_dtype_.MLX_BFLOAT16),
  float32(mlx_dtype_.MLX_FLOAT32),
  float64(mlx_dtype_.MLX_FLOAT64),
  complex64(mlx_dtype_.MLX_COMPLEX64);

  const MLXDtype(this.raw);
  final mlx_dtype_ raw;

  static MLXDtype fromRaw(mlx_dtype_ raw) =>
      values.firstWhere((d) => d.raw == raw);
}

/// An n-dimensional array backed by the MLX runtime.
///
/// [MLXArray] owns the underlying `mlx_array` and must be [dispose]d when no
/// longer needed.
///
/// Corresponds to `MLXArray` in mlx-swift.
final class MLXArray {
  /// Internal: takes ownership of [ptr].
  MLXArray.owned(this.context, this._ptr);

  /// The execution context this array belongs to.
  final MLXContext context;
  final ffi.Pointer<mlx_array> _ptr;
  bool _disposed = false;

  MlxCBindings get _b => context.bindings;
  mlx_stream get _s => context.stream;
  mlx_array get _raw {
    _assertAlive();
    return _ptr.ref;
  }

  /// The underlying `mlx_array` value.
  ///
  /// For use by other libraries in this package (e.g. random, linalg, fft,
  /// io). Do not call [dispose] on the returned value directly.
  mlx_array get raw => _raw;

  // ---------------------------------------------------------------------------
  // Factory constructors
  // ---------------------------------------------------------------------------

  factory MLXArray.bool_(MLXContext ctx, bool value) {
    final ptr = calloc<mlx_array>();
    ptr.ref = ctx.bindings.mlx_array_new_bool(value);
    return MLXArray.owned(ctx, ptr);
  }

  factory MLXArray.int_(MLXContext ctx, int value) {
    final ptr = calloc<mlx_array>();
    ptr.ref = ctx.bindings.mlx_array_new_int(value);
    return MLXArray.owned(ctx, ptr);
  }

  factory MLXArray.float_(MLXContext ctx, double value) {
    final ptr = calloc<mlx_array>();
    ptr.ref = ctx.bindings.mlx_array_new_float32(value);
    return MLXArray.owned(ctx, ptr);
  }

  factory MLXArray.fromInts(
    MLXContext ctx,
    List<int> data, {
    List<int>? shape,
    MLXDtype dtype = MLXDtype.int32,
  }) {
    final resolvedShape = shape ?? [data.length];
    final dataPtr = calloc<ffi.Int32>(data.length);
    final shapePtr = _allocShapePtr(resolvedShape);
    try {
      for (var i = 0; i < data.length; i++) {
        dataPtr[i] = data[i];
      }
      final arr = ctx.bindings.mlx_array_new_data(
        dataPtr.cast<ffi.Void>(),
        shapePtr,
        resolvedShape.length,
        dtype.raw,
      );
      final ptr = calloc<mlx_array>();
      ptr.ref = arr;
      return MLXArray.owned(ctx, ptr);
    } finally {
      calloc.free(dataPtr);
      calloc.free(shapePtr);
    }
  }

  factory MLXArray.fromFloats(
    MLXContext ctx,
    List<double> data, {
    List<int>? shape,
    MLXDtype dtype = MLXDtype.float32,
  }) {
    final resolvedShape = shape ?? [data.length];
    final dataPtr = calloc<ffi.Float>(data.length);
    final shapePtr = _allocShapePtr(resolvedShape);
    try {
      for (var i = 0; i < data.length; i++) {
        dataPtr[i] = data[i];
      }
      final arr = ctx.bindings.mlx_array_new_data(
        dataPtr.cast<ffi.Void>(),
        shapePtr,
        resolvedShape.length,
        dtype.raw,
      );
      final ptr = calloc<mlx_array>();
      ptr.ref = arr;
      return MLXArray.owned(ctx, ptr);
    } finally {
      calloc.free(dataPtr);
      calloc.free(shapePtr);
    }
  }

  factory MLXArray.zeros(
    MLXContext ctx,
    List<int> shape, {
    MLXDtype dtype = MLXDtype.float32,
  }) {
    final shapePtr = _allocShapePtr(shape);
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings
              .mlx_zeros(out, shapePtr, shape.length, dtype.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
    }
  }

  factory MLXArray.ones(
    MLXContext ctx,
    List<int> shape, {
    MLXDtype dtype = MLXDtype.float32,
  }) {
    final shapePtr = _allocShapePtr(shape);
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings
              .mlx_ones(out, shapePtr, shape.length, dtype.raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
    }
  }

  factory MLXArray.arange(
    MLXContext ctx,
    double start,
    double stop,
    double step, {
    MLXDtype dtype = MLXDtype.float32,
  }) =>
      _opChecked(
          ctx,
          (out) => ctx.bindings
              .mlx_arange(out, start, stop, step, dtype.raw, ctx.stream));

  /// Identity matrix of size [n]×[m] (defaults to square) with ones on diagonal [k].
  factory MLXArray.eye(
    MLXContext ctx,
    int n, {
    int? m,
    int k = 0,
    MLXDtype dtype = MLXDtype.float32,
  }) =>
      _opChecked(ctx,
          (out) => ctx.bindings.mlx_eye(out, n, m ?? n, k, dtype.raw, ctx.stream));

  /// Array filled with [value] broadcast to [shape].
  factory MLXArray.full(
    MLXContext ctx,
    List<int> shape,
    MLXArray value, {
    MLXDtype? dtype,
  }) {
    final shapePtr = _allocShapePtr(shape);
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_full(out, shapePtr, shape.length, value._raw,
              (dtype ?? value.dtype).raw, ctx.stream));
    } finally {
      calloc.free(shapePtr);
    }
  }

  /// Evenly-spaced values in [start, stop] with [num] points.
  factory MLXArray.linspace(
    MLXContext ctx,
    double start,
    double stop, {
    int num = 50,
    MLXDtype dtype = MLXDtype.float32,
  }) =>
      _opChecked(ctx,
          (out) => ctx.bindings.mlx_linspace(out, start, stop, num, dtype.raw, ctx.stream));

  // ---------------------------------------------------------------------------
  // Properties
  // ---------------------------------------------------------------------------

  int get ndim => _b.mlx_array_ndim(_raw);
  int get size => _b.mlx_array_size(_raw);
  int get nbytes => _b.mlx_array_nbytes(_raw);
  MLXDtype get dtype => MLXDtype.fromRaw(_b.mlx_array_dtype(_raw));

  List<int> get shape {
    final rank = ndim;
    final ptr = _b.mlx_array_shape(_raw);
    return List<int>.generate(rank, (i) => ptr[i], growable: false);
  }

  int dim(int axis) {
    final s = shape;
    return s[axis < 0 ? s.length + axis : axis];
  }

  // ---------------------------------------------------------------------------
  // Evaluation
  // ---------------------------------------------------------------------------

  void eval() => context.check(_b.mlx_array_eval(_raw), 'mlx_array_eval');

  // ---------------------------------------------------------------------------
  // Data extraction
  // ---------------------------------------------------------------------------

  int itemInt() {
    eval();
    context.synchronize();
    final out = calloc<ffi.Int32>();
    try {
      context.check(_b.mlx_array_item_int32(out, _raw), 'mlx_array_item_int32');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  double itemFloat() {
    eval();
    context.synchronize();
    final out = calloc<ffi.Float>();
    try {
      context.check(
          _b.mlx_array_item_float32(out, _raw), 'mlx_array_item_float32');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  Float32List toFloat32List() {
    final casted = dtype == MLXDtype.float32 ? this : astype(MLXDtype.float32);
    casted.eval();
    context.synchronize();
    final ptr = _b.mlx_array_data_float32(casted._raw);
    final result = Float32List(casted.size);
    for (var i = 0; i < result.length; i++) {
      result[i] = ptr[i];
    }
    if (!identical(casted, this)) casted.dispose();
    return result;
  }

  Int32List toInt32List() {
    final casted = dtype == MLXDtype.int32 ? this : astype(MLXDtype.int32);
    casted.eval();
    context.synchronize();
    final ptr = _b.mlx_array_data_int32(casted._raw);
    final result = Int32List(casted.size);
    for (var i = 0; i < result.length; i++) {
      result[i] = ptr[i];
    }
    if (!identical(casted, this)) casted.dispose();
    return result;
  }

  // ---------------------------------------------------------------------------
  // Arithmetic
  // ---------------------------------------------------------------------------

  MLXArray operator +(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_add(out, _raw, other._raw, _s));
  MLXArray operator -(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_subtract(out, _raw, other._raw, _s));
  MLXArray operator *(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_multiply(out, _raw, other._raw, _s));
  MLXArray operator /(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_divide(out, _raw, other._raw, _s));

  // ---------------------------------------------------------------------------
  // Linear algebra
  // ---------------------------------------------------------------------------

  MLXArray matmul(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_matmul(out, _raw, other._raw, _s));

  // ---------------------------------------------------------------------------
  // Shape manipulation
  // ---------------------------------------------------------------------------

  MLXArray reshape(List<int> newShape) {
    final ptr = _allocShapePtr(newShape);
    try {
      return _opChecked(context,
          (out) => _b.mlx_reshape(out, _raw, ptr, newShape.length, _s));
    } finally {
      calloc.free(ptr);
    }
  }

  MLXArray expandDims(int axis) =>
      _opChecked(context, (out) => _b.mlx_expand_dims(out, _raw, axis, _s));

  MLXArray squeeze({int? axis}) => axis != null
      ? _opChecked(context, (out) => _b.mlx_squeeze_axis(out, _raw, axis, _s))
      : _opChecked(context, (out) => _b.mlx_squeeze(out, _raw, _s));

  MLXArray transpose([List<int>? axes]) {
    if (axes == null || axes.isEmpty) {
      return _opChecked(context, (out) => _b.mlx_transpose(out, _raw, _s));
    }
    final ptr = calloc<ffi.Int>(axes.length);
    for (var i = 0; i < axes.length; i++) {
      ptr[i] = axes[i];
    }
    try {
      return _opChecked(context,
          (out) => _b.mlx_transpose_axes(out, _raw, ptr, axes.length, _s));
    } finally {
      calloc.free(ptr);
    }
  }

  /// Transpose shorthand for 2-D matrices.
  MLXArray get T => transpose();

  MLXArray swapAxes(int a, int b) =>
      _opChecked(context, (out) => _b.mlx_swapaxes(out, _raw, a, b, _s));

  MLXArray flatten({int startAxis = 0, int endAxis = -1}) {
    final s = shape;
    final end = endAxis < 0 ? s.length + endAxis : endAxis;
    var flatDim = 1;
    for (var i = startAxis; i <= end; i++) {
      flatDim *= s[i];
    }
    return reshape(
        [...s.sublist(0, startAxis), flatDim, ...s.sublist(end + 1)]);
  }

  // ---------------------------------------------------------------------------
  // Slicing & indexing
  // ---------------------------------------------------------------------------

  MLXArray slice({
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) {
    final rank = ndim;
    assert(start.length == rank && stop.length == rank);
    final stride = strides ?? List.filled(rank, 1);
    final startPtr = _allocIntPtr(start);
    final stopPtr = _allocIntPtr(stop);
    final stridePtr = _allocIntPtr(stride);
    try {
      return _opChecked(
          context,
          (out) => _b.mlx_slice(
              out, _raw, startPtr, rank, stopPtr, rank, stridePtr, rank, _s));
    } finally {
      calloc.free(startPtr);
      calloc.free(stopPtr);
      calloc.free(stridePtr);
    }
  }

  /// Functional scatter-update: writes [update] into `self[start:stop:stride]`.
  MLXArray sliceUpdate(
    MLXArray update, {
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) {
    final rank = ndim;
    final stride = strides ?? List.filled(rank, 1);
    final startPtr = _allocIntPtr(start);
    final stopPtr = _allocIntPtr(stop);
    final stridePtr = _allocIntPtr(stride);
    try {
      return _opChecked(
          context,
          (out) => _b.mlx_slice_update(out, _raw, update._raw, startPtr, rank,
              stopPtr, rank, stridePtr, rank, _s));
    } finally {
      calloc.free(startPtr);
      calloc.free(stopPtr);
      calloc.free(stridePtr);
    }
  }

  MLXArray take(MLXArray indices, {int axis = 0}) => _opChecked(
      context, (out) => _b.mlx_take_axis(out, _raw, indices._raw, axis, _s));

  // ---------------------------------------------------------------------------
  // Reductions
  // ---------------------------------------------------------------------------

  MLXArray sum({int? axis, bool keepdims = false}) => axis != null
      ? _opChecked(
          context, (out) => _b.mlx_sum_axis(out, _raw, axis, keepdims, _s))
      : _opChecked(context, (out) => _b.mlx_sum(out, _raw, keepdims, _s));

  MLXArray mean({int? axis, bool keepdims = false}) => axis != null
      ? _opChecked(
          context, (out) => _b.mlx_mean_axis(out, _raw, axis, keepdims, _s))
      : _opChecked(context, (out) => _b.mlx_mean(out, _raw, keepdims, _s));

  MLXArray max({int? axis, bool keepdims = false}) => axis != null
      ? _opChecked(
          context, (out) => _b.mlx_max_axis(out, _raw, axis, keepdims, _s))
      : _opChecked(context, (out) => _b.mlx_max(out, _raw, keepdims, _s));

  MLXArray argmax({int axis = -1, bool keepdims = false}) => _opChecked(
      context, (out) => _b.mlx_argmax_axis(out, _raw, axis, keepdims, _s));

  MLXArray softmax({int axis = -1}) => _opChecked(
      context, (out) => _b.mlx_softmax_axis(out, _raw, axis, false, _s));

  MLXArray cumsum(
          {int axis = 0, bool exclusive = false, bool reverse = false}) =>
      _opChecked(context,
          (out) => _b.mlx_cumsum(out, _raw, axis, exclusive, reverse, _s));

  MLXArray sort({int axis = -1}) =>
      _opChecked(context, (out) => _b.mlx_sort_axis(out, _raw, axis, _s));

  MLXArray argsort({int axis = -1}) =>
      _opChecked(context, (out) => _b.mlx_argsort_axis(out, _raw, axis, _s));

  MLXArray min({int? axis, bool keepdims = false}) => axis != null
      ? _opChecked(
          context, (out) => _b.mlx_min_axis(out, _raw, axis, keepdims, _s))
      : _opChecked(context, (out) => _b.mlx_min(out, _raw, keepdims, _s));

  MLXArray prod({int? axis, bool keepdims = false}) => axis != null
      ? _opChecked(
          context, (out) => _b.mlx_prod_axis(out, _raw, axis, keepdims, _s))
      : _opChecked(context, (out) => _b.mlx_prod(out, _raw, keepdims, _s));

  MLXArray std({int? axis, bool keepdims = false, int ddof = 0}) => axis != null
      ? _opChecked(context,
          (out) => _b.mlx_std_axis(out, _raw, axis, keepdims, ddof, _s))
      : _opChecked(
          context, (out) => _b.mlx_std(out, _raw, keepdims, ddof, _s));

  MLXArray variance({int? axis, bool keepdims = false, int ddof = 0}) =>
      axis != null
          ? _opChecked(context,
              (out) => _b.mlx_var_axis(out, _raw, axis, keepdims, ddof, _s))
          : _opChecked(
              context, (out) => _b.mlx_var(out, _raw, keepdims, ddof, _s));

  MLXArray logSumExp({int? axis, bool keepdims = false}) => axis != null
      ? _opChecked(context,
          (out) => _b.mlx_logsumexp_axis(out, _raw, axis, keepdims, _s))
      : _opChecked(
          context, (out) => _b.mlx_logsumexp(out, _raw, keepdims, _s));

  /// Returns the top-[k] values along [axis].
  MLXArray topk(int k, {int axis = -1}) => _opChecked(
      context, (out) => _b.mlx_topk_axis(out, _raw, k, axis, _s));

  // ---------------------------------------------------------------------------
  // Element-wise math
  // ---------------------------------------------------------------------------

  MLXArray exp() => _opChecked(context, (out) => _b.mlx_exp(out, _raw, _s));
  MLXArray log() => _opChecked(context, (out) => _b.mlx_log(out, _raw, _s));
  MLXArray sqrt() => _opChecked(context, (out) => _b.mlx_sqrt(out, _raw, _s));
  MLXArray rsqrt() => _opChecked(context, (out) => _b.mlx_rsqrt(out, _raw, _s));
  MLXArray abs() => _opChecked(context, (out) => _b.mlx_abs(out, _raw, _s));
  MLXArray sigmoid() =>
      _opChecked(context, (out) => _b.mlx_sigmoid(out, _raw, _s));
  MLXArray tanh() => _opChecked(context, (out) => _b.mlx_tanh(out, _raw, _s));
  MLXArray sin() => _opChecked(context, (out) => _b.mlx_sin(out, _raw, _s));
  MLXArray cos() => _opChecked(context, (out) => _b.mlx_cos(out, _raw, _s));
  MLXArray square() =>
      _opChecked(context, (out) => _b.mlx_square(out, _raw, _s));

  MLXArray erf() => _opChecked(context, (out) => _b.mlx_erf(out, _raw, _s));

  MLXArray floor() => _opChecked(context, (out) => _b.mlx_floor(out, _raw, _s));

  MLXArray ceil() => _opChecked(context, (out) => _b.mlx_ceil(out, _raw, _s));

  MLXArray round({int decimals = 0}) =>
      _opChecked(context, (out) => _b.mlx_round(out, _raw, decimals, _s));

  MLXArray clip({required MLXArray min, required MLXArray max}) => _opChecked(
      context, (out) => _b.mlx_clip(out, _raw, min._raw, max._raw, _s));

  /// Pad [this] array with [padValue] along each listed [axes].
  ///
  /// [lowPad] and [highPad] specify how many elements to add before and after
  /// each axis respectively.
  MLXArray pad({
    required List<int> axes,
    required List<int> lowPad,
    required List<int> highPad,
    MLXArray? padValue,
    String mode = 'constant',
  }) {
    assert(axes.length == lowPad.length && axes.length == highPad.length);
    final axesPtr = _allocIntPtr(axes);
    final lowPtr = _allocIntPtr(lowPad);
    final highPtr = _allocIntPtr(highPad);
    final modePtr = mode.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final valOwned = padValue == null;
    final val = padValue ?? MLXArray.zeros(context, []);
    try {
      return _opChecked(
          context,
          (out) => _b.mlx_pad(out, _raw, axesPtr, axes.length, lowPtr,
              lowPad.length, highPtr, highPad.length, val._raw, modePtr, _s));
    } finally {
      calloc.free(axesPtr);
      calloc.free(lowPtr);
      calloc.free(highPtr);
      calloc.free(modePtr);
      if (valOwned) val.dispose();
    }
  }

  /// Gaussian Error Linear Unit: `x * 0.5 * (1 + erf(x / √2))`.
  MLXArray gelu() {
    const c = 0.7071067811865476; // 1/sqrt(2)
    final cArr = MLXArray.float_(context, c);
    final scaled = this * cArr;
    cArr.dispose();
    final e = scaled.erf();
    scaled.dispose();
    final one = MLXArray.float_(context, 1.0);
    final onePlusE = one + e;
    one.dispose();
    e.dispose();
    final half = MLXArray.float_(context, 0.5);
    final result = this * onePlusE * half;
    onePlusE.dispose();
    half.dispose();
    return result;
  }

  MLXArray silu() {
    final s = sigmoid();
    final r = this * s;
    s.dispose();
    return r;
  }

  /// Leaky ReLU: `max(negSlope * x, x)`.
  MLXArray leakyRelu({double negSlope = 0.01}) {
    final s = MLXArray.float_(context, negSlope);
    final scaled = this * s;
    s.dispose();
    final r = scaled.maximum(this);
    scaled.dispose();
    return r;
  }

  /// Exponential Linear Unit: `x` if `x ≥ 0`, else `alpha * (exp(x) - 1)`.
  MLXArray elu({double alpha = 1.0}) {
    final expM1 = expm1();
    final aA = MLXArray.float_(context, alpha);
    final neg = expM1 * aA;
    expM1.dispose();
    aA.dispose();
    final zero = MLXArray.zeros(context, [], dtype: dtype);
    final r = where(context, greaterEqual(zero), this, neg);
    zero.dispose();
    neg.dispose();
    return r;
  }

  /// Continuously-differentiable ELU:
  /// `max(x, 0) + min(0, alpha * (exp(x / alpha) - 1))`.
  MLXArray celu({double alpha = 1.0}) {
    final aA = MLXArray.float_(context, alpha);
    final invA = MLXArray.float_(context, 1.0 / alpha);
    final scaled = this * invA;
    invA.dispose();
    final e = scaled.expm1();
    scaled.dispose();
    final neg = aA * e;
    aA.dispose();
    e.dispose();
    final zero = MLXArray.zeros(context, [], dtype: dtype);
    final posX = maximum(zero);
    final negPart = neg.minimum(zero);
    neg.dispose();
    zero.dispose();
    final r = posX + negPart;
    posX.dispose();
    negPart.dispose();
    return r;
  }

  /// ReLU clipped at 6: `min(max(x, 0), 6)`.
  MLXArray relu6() {
    final r = relu();
    final six = MLXArray.float_(context, 6.0);
    final clipped = r.minimum(six);
    r.dispose();
    six.dispose();
    return clipped;
  }

  /// Squared ReLU: `relu(x)²`.
  MLXArray reluSquared() {
    final r = relu();
    final sq = r.square();
    r.dispose();
    return sq;
  }

  /// Log-softmax along [axis].
  MLXArray logSoftmax({int axis = -1}) {
    final lse = logSumExp(axis: axis, keepdims: true);
    final r = this - lse;
    lse.dispose();
    return r;
  }

  /// Softplus: `log(1 + exp(x))`.
  MLXArray softPlus() {
    final e = exp();
    final one = MLXArray.float_(context, 1.0);
    final onePlusE = one + e;
    one.dispose();
    e.dispose();
    final r = onePlusE.log();
    onePlusE.dispose();
    return r;
  }

  /// Softsign: `x / (1 + |x|)`.
  MLXArray softsign() {
    final a = abs();
    final one = MLXArray.float_(context, 1.0);
    final denom = one + a;
    one.dispose();
    a.dispose();
    final r = this / denom;
    denom.dispose();
    return r;
  }

  /// Softshrink: `x - lambda` if `x > lambda`, `x + lambda` if `x < -lambda`, else `0`.
  MLXArray softshrink({double lambda = 0.5}) {
    final lA = MLXArray.float_(context, lambda);
    final negLA = MLXArray.float_(context, -lambda);
    final zero = MLXArray.zeros(context, [], dtype: dtype);
    final upper = this - lA;
    final lower = this + lA;
    final condPos = greaterEqual(lA);    // x >= lambda -> x - lambda, else check lower
    final condNeg = less(negLA);         // x < -lambda -> x + lambda
    final r1 = where(context, condNeg, lower, zero);
    final r = where(context, condPos, upper, r1);
    lA.dispose();
    negLA.dispose();
    zero.dispose();
    upper.dispose();
    lower.dispose();
    condPos.dispose();
    condNeg.dispose();
    r1.dispose();
    return r;
  }

  MLXArray relu() {
    final zero = MLXArray.zeros(context, [], dtype: dtype);
    final r =
        _opChecked(context, (out) => _b.mlx_maximum(out, _raw, zero._raw, _s));
    zero.dispose();
    return r;
  }

  MLXArray negative() =>
      _opChecked(context, (out) => _b.mlx_negative(out, _raw, _s));

  MLXArray log1p() =>
      _opChecked(context, (out) => _b.mlx_log1p(out, _raw, _s));

  MLXArray log2() =>
      _opChecked(context, (out) => _b.mlx_log2(out, _raw, _s));

  MLXArray expm1() =>
      _opChecked(context, (out) => _b.mlx_expm1(out, _raw, _s));

  MLXArray erfInverse() =>
      _opChecked(context, (out) => _b.mlx_erfinv(out, _raw, _s));

  MLXArray sign() =>
      _opChecked(context, (out) => _b.mlx_sign(out, _raw, _s));

  MLXArray isNaN() =>
      _opChecked(context, (out) => _b.mlx_isnan(out, _raw, _s));

  MLXArray isInf() =>
      _opChecked(context, (out) => _b.mlx_isinf(out, _raw, _s));

  MLXArray sinh() =>
      _opChecked(context, (out) => _b.mlx_sinh(out, _raw, _s));

  MLXArray cosh() =>
      _opChecked(context, (out) => _b.mlx_cosh(out, _raw, _s));

  MLXArray tan() =>
      _opChecked(context, (out) => _b.mlx_tan(out, _raw, _s));

  MLXArray atan() =>
      _opChecked(context, (out) => _b.mlx_arctan(out, _raw, _s));

  MLXArray acos() =>
      _opChecked(context, (out) => _b.mlx_arccos(out, _raw, _s));

  MLXArray asin() =>
      _opChecked(context, (out) => _b.mlx_arcsin(out, _raw, _s));

  MLXArray degrees() =>
      _opChecked(context, (out) => _b.mlx_degrees(out, _raw, _s));

  MLXArray radians() =>
      _opChecked(context, (out) => _b.mlx_radians(out, _raw, _s));

  MLXArray maximum(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_maximum(out, _raw, other._raw, _s));

  MLXArray minimum(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_minimum(out, _raw, other._raw, _s));

  MLXArray power(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_power(out, _raw, other._raw, _s));

  MLXArray nanToNum({double nan = 0.0, double? posInf, double? negInf}) {
    final optPosInf = calloc<mlx_optional_float>();
    final optNegInf = calloc<mlx_optional_float>();
    optPosInf.ref.has_value = posInf != null;
    optPosInf.ref.value = posInf ?? 0.0;
    optNegInf.ref.has_value = negInf != null;
    optNegInf.ref.value = negInf ?? 0.0;
    try {
      return _opChecked(context,
          (out) => _b.mlx_nan_to_num(out, _raw, nan, optPosInf.ref, optNegInf.ref, _s));
    } finally {
      calloc.free(optPosInf);
      calloc.free(optNegInf);
    }
  }

  /// Extract lower triangle (k=0 is main diagonal).
  MLXArray tril({int k = 0}) =>
      _opChecked(context, (out) => _b.mlx_tril(out, _raw, k, _s));

  /// Extract upper triangle (k=0 is main diagonal).
  MLXArray triu({int k = 0}) =>
      _opChecked(context, (out) => _b.mlx_triu(out, _raw, k, _s));

  /// Trace (sum of diagonal elements).
  MLXArray trace({int offset = 0, int axis1 = 0, int axis2 = 1}) =>
      _opChecked(context,
          (out) => _b.mlx_trace(out, _raw, offset, axis1, axis2, dtype.raw, _s));

  /// Tile this array by repeating [reps] times along each axis.
  MLXArray tile(List<int> reps) {
    final ptr = _allocIntPtr(reps);
    try {
      return _opChecked(
          context, (out) => _b.mlx_tile(out, _raw, ptr, reps.length, _s));
    } finally {
      calloc.free(ptr);
    }
  }

  /// Roll array elements by [shift] positions (all axes flattened).
  MLXArray roll(int shift) {
    final shiftPtr = calloc<ffi.Int>(1);
    shiftPtr[0] = shift;
    try {
      return _opChecked(
          context, (out) => _b.mlx_roll(out, _raw, shiftPtr, 1, _s));
    } finally {
      calloc.free(shiftPtr);
    }
  }

  /// Roll array elements by [shift] positions along [axis].
  MLXArray rollAxis(int shift, {required int axis}) {
    final shiftPtr = calloc<ffi.Int>(1);
    shiftPtr[0] = shift;
    try {
      return _opChecked(context,
          (out) => _b.mlx_roll_axis(out, _raw, shiftPtr, 1, axis, _s));
    } finally {
      calloc.free(shiftPtr);
    }
  }

  /// Repeat this array [repeats] times along [axis].
  MLXArray repeat(int repeats, {int axis = 0}) => _opChecked(
      context, (out) => _b.mlx_repeat_axis(out, _raw, repeats, axis, _s));

  /// Split into [numSplits] equal parts along [axis].
  List<MLXArray> split(int numSplits, {int axis = 0}) {
    final vec = calloc<mlx_vector_array>();
    try {
      context.check(_b.mlx_split(vec, _raw, numSplits, axis, _s), 'mlx_split');
      final count = _b.mlx_vector_array_size(vec.ref);
      return List.generate(count, (i) {
        final ptr = calloc<mlx_array>();
        context.check(
            _b.mlx_vector_array_get(ptr, vec.ref, i), 'mlx_vector_array_get');
        return MLXArray.owned(context, ptr);
      });
    } finally {
      _b.mlx_vector_array_free(vec.ref);
      calloc.free(vec);
    }
  }

  // ---------------------------------------------------------------------------
  // Comparisons
  // ---------------------------------------------------------------------------

  MLXArray greaterEqual(MLXArray other) => _opChecked(
      context, (out) => _b.mlx_greater_equal(out, _raw, other._raw, _s));

  MLXArray less(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_less(out, _raw, other._raw, _s));

  MLXArray logicalAnd(MLXArray other) => _opChecked(
      context, (out) => _b.mlx_logical_and(out, _raw, other._raw, _s));

  MLXArray equal(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_equal(out, _raw, other._raw, _s));

  MLXArray greater(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_greater(out, _raw, other._raw, _s));

  MLXArray lessEqual(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_less_equal(out, _raw, other._raw, _s));

  MLXArray notEqual(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_not_equal(out, _raw, other._raw, _s));

  MLXArray logicalNot() =>
      _opChecked(context, (out) => _b.mlx_logical_not(out, _raw, _s));

  MLXArray logicalOr(MLXArray other) =>
      _opChecked(context, (out) => _b.mlx_logical_or(out, _raw, other._raw, _s));

  // ---------------------------------------------------------------------------
  // Type casting
  // ---------------------------------------------------------------------------

  MLXArray astype(MLXDtype newDtype) =>
      _opChecked(context, (out) => _b.mlx_astype(out, _raw, newDtype.raw, _s));

  // ---------------------------------------------------------------------------
  // Fast NN primitives (Metal-accelerated)
  // ---------------------------------------------------------------------------

  MLXArray rmsNorm(MLXArray weight, {double eps = 1e-5}) => _opChecked(
      context, (out) => _b.mlx_fast_rms_norm(out, _raw, weight._raw, eps, _s));

  MLXArray layerNorm(MLXArray weight, MLXArray? bias, {double eps = 1e-5}) {
    if (bias != null) {
      return _opChecked(
          context,
          (out) => _b.mlx_fast_layer_norm(
              out, _raw, weight._raw, bias._raw, eps, _s));
    }
    final zeroBias = MLXArray.zeros(context, [weight.shape.last]);
    try {
      return _opChecked(
          context,
          (out) => _b.mlx_fast_layer_norm(
              out, _raw, weight._raw, zeroBias._raw, eps, _s));
    } finally {
      zeroBias.dispose();
    }
  }

  /// Rotary position embedding via `mlx_fast_rope`.
  MLXArray rope({
    required int dims,
    bool traditional = false,
    double? base,
    double scale = 1.0,
    int offset = 0,
    MLXArray? freqs,
  }) {
    final optBase = calloc<mlx_optional_float>();
    optBase.ref.has_value = base != null;
    optBase.ref.value = base ?? 0.0;
    final freqsArr = freqs ?? MLXArray.zeros(context, [0]);
    final freqsOwned = freqs == null;
    try {
      return _opChecked(
          context,
          (out) => _b.mlx_fast_rope(out, _raw, dims, traditional, optBase.ref,
              scale, offset, freqsArr._raw, _s));
    } finally {
      calloc.free(optBase);
      if (freqsOwned) freqsArr.dispose();
    }
  }

  // ---------------------------------------------------------------------------
  // Memory management
  // ---------------------------------------------------------------------------

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _b.mlx_array_free(_ptr.ref);
    calloc.free(_ptr);
  }

  bool get isDisposed => _disposed;

  @override
  String toString() {
    if (_disposed) return 'MLXArray(disposed)';
    final out = calloc<mlx_string>();
    try {
      _b.mlx_array_tostring(out, _raw);
      return context.readString(out.ref);
    } finally {
      calloc.free(out);
    }
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  void _assertAlive() {
    if (_disposed) throw StateError('MLXArray has already been disposed.');
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
      throw MLXException('operation', code);
    }
    return MLXArray.owned(ctx, out);
  }

  static ffi.Pointer<ffi.Int> _allocShapePtr(List<int> shape) =>
      _allocIntPtr(shape.isEmpty ? [0] : shape);

  static ffi.Pointer<ffi.Int> _allocIntPtr(List<int> values) {
    final ptr = calloc<ffi.Int>(values.isEmpty ? 1 : values.length);
    for (var i = 0; i < values.length; i++) {
      ptr[i] = values[i];
    }
    return ptr;
  }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// 2-D convolution — input `[B, H, W, C]`, weight `[outC, kH, kW, inC]`.
MLXArray conv2d(
  MLXContext ctx,
  MLXArray input,
  MLXArray weight, {
  int stride = 1,
  int padding = 0,
  int dilation = 1,
  int groups = 1,
}) =>
    MLXArray._opChecked(
        ctx,
        (out) => ctx.bindings.mlx_conv2d(out, input._raw, weight._raw, stride,
            stride, padding, padding, dilation, dilation, groups, ctx.stream));

/// Concatenate [arrays] along [axis].
MLXArray concatenate(MLXContext ctx, List<MLXArray> arrays, {int axis = 0}) {
  assert(arrays.isNotEmpty, 'arrays must not be empty');
  final vec = calloc<mlx_vector_array>();
  try {
    vec.ref = ctx.bindings.mlx_vector_array_new();
    for (final a in arrays) {
      ctx.check(ctx.bindings.mlx_vector_array_append_value(vec.ref, a._raw),
          'mlx_vector_array_append_value');
    }
    return MLXArray._opChecked(
        ctx,
        (out) =>
            ctx.bindings.mlx_concatenate_axis(out, vec.ref, axis, ctx.stream));
  } finally {
    ctx.bindings.mlx_vector_array_free(vec.ref);
    calloc.free(vec);
  }
}

/// Stack [arrays] along a new [axis].
MLXArray stack(MLXContext ctx, List<MLXArray> arrays, {int axis = 0}) {
  assert(arrays.isNotEmpty, 'arrays must not be empty');
  final vec = calloc<mlx_vector_array>();
  try {
    vec.ref = ctx.bindings.mlx_vector_array_new();
    for (final a in arrays) {
      ctx.check(ctx.bindings.mlx_vector_array_append_value(vec.ref, a._raw),
          'mlx_vector_array_append_value');
    }
    return MLXArray._opChecked(ctx,
        (out) => ctx.bindings.mlx_stack_axis(out, vec.ref, axis, ctx.stream));
  } finally {
    ctx.bindings.mlx_vector_array_free(vec.ref);
    calloc.free(vec);
  }
}

/// Element-wise conditional select.
MLXArray where(MLXContext ctx, MLXArray condition, MLXArray x, MLXArray y) =>
    MLXArray._opChecked(
        ctx,
        (out) => ctx.bindings
            .mlx_where(out, condition._raw, x._raw, y._raw, ctx.stream));

/// Scaled dot-product attention.
///
/// [maskMode]: `'none'`, `'causal'`, or `'array'` (requires [mask]).
MLXArray scaledDotProductAttention(
  MLXContext ctx, {
  required MLXArray queries,
  required MLXArray keys,
  required MLXArray values,
  required double scale,
  String maskMode = 'none',
  MLXArray? mask,
}) {
  final modePtr = maskMode.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final maskArr = mask ?? MLXArray.zeros(ctx, [0]);
  final maskOwned = mask == null;
  final sinksArr = MLXArray.zeros(ctx, [0]);
  try {
    return MLXArray._opChecked(
        ctx,
        (out) => ctx.bindings.mlx_fast_scaled_dot_product_attention(
              out,
              queries._raw,
              keys._raw,
              values._raw,
              scale,
              modePtr,
              maskArr._raw,
              sinksArr._raw,
              ctx.stream,
            ));
  } finally {
    calloc.free(modePtr);
    if (maskOwned) maskArr.dispose();
    sinksArr.dispose();
  }
}

/// Synchronously evaluate a collection of arrays.
void evalAll(MLXContext ctx, Iterable<MLXArray> arrays) {
  final vec = calloc<mlx_vector_array>();
  try {
    vec.ref = ctx.bindings.mlx_vector_array_new();
    for (final a in arrays) {
      ctx.check(ctx.bindings.mlx_vector_array_append_value(vec.ref, a._raw),
          'mlx_vector_array_append_value');
    }
    ctx.check(ctx.bindings.mlx_eval(vec.ref), 'mlx_eval');
  } finally {
    ctx.bindings.mlx_vector_array_free(vec.ref);
    calloc.free(vec);
  }
}

/// Create a boolean causal mask — shape `[n, offset+n]`.
MLXArray createCausalMask(
  MLXContext ctx, {
  required int n,
  required int offset,
  int? windowSize,
}) {
  final total = offset + n;
  final linds = MLXArray.arange(
          ctx, offset.toDouble(), (offset + n).toDouble(), 1.0,
          dtype: MLXDtype.int32)
      .expandDims(1);
  final rinds =
      MLXArray.arange(ctx, 0.0, total.toDouble(), 1.0, dtype: MLXDtype.int32)
          .expandDims(0);

  var mask = linds.greaterEqual(rinds);

  if (windowSize != null) {
    final ws = MLXArray.int_(ctx, windowSize);
    final shifted = rinds + ws;
    final wm = linds.less(shifted);
    final combined = mask.logicalAnd(wm);
    mask.dispose();
    ws.dispose();
    shifted.dispose();
    wm.dispose();
    mask = combined;
  }
  linds.dispose();
  rinds.dispose();
  return mask;
}

/// Bilinear resize of an image array.
///
/// [image] must have shape `[H, W, C]` (float32, NHWC without batch dim).
/// Returns a new array of shape `[targetH, targetW, C]`.
///
/// Uses half-pixel-centre convention (align_corners = false), the same
/// convention used by PyTorch `F.interpolate` with `align_corners=False` and
/// `antialias=False`.
MLXArray bilinearResize(
  MLXContext ctx,
  MLXArray image,
  int targetH,
  int targetW,
) {
  final srcH = image.dim(0);
  final srcW = image.dim(1);

  if (srcH == targetH && srcW == targetW) {
    // Make a copy so ownership semantics are consistent.
    return image
        .slice(start: List.filled(3, 0), stop: [srcH, srcW, image.dim(2)]);
  }

  final scaleH = srcH / targetH;
  final scaleW = srcW / targetW;

  // Source coords: yS[i] = (i + 0.5) * scaleH - 0.5
  final iRange = MLXArray.arange(ctx, 0.0, targetH.toDouble(), 1.0);
  final jRange = MLXArray.arange(ctx, 0.0, targetW.toDouble(), 1.0);

  final halfScH = MLXArray.float_(ctx, 0.5 * scaleH - 0.5);
  final scHArr = MLXArray.float_(ctx, scaleH);
  final halfScW = MLXArray.float_(ctx, 0.5 * scaleW - 0.5);
  final scWArr = MLXArray.float_(ctx, scaleW);

  final yS = iRange * scHArr + halfScH; // [targetH]
  final xS = jRange * scWArr + halfScW; // [targetW]
  iRange.dispose();
  jRange.dispose();
  halfScH.dispose();
  scHArr.dispose();
  halfScW.dispose();
  scWArr.dispose();

  // Clamp and compute integer + fractional parts.
  final zeroF = MLXArray.float_(ctx, 0.0);
  final maxYF = MLXArray.float_(ctx, (srcH - 1).toDouble());
  final maxXF = MLXArray.float_(ctx, (srcW - 1).toDouble());

  final ySc = yS.clip(min: zeroF, max: maxYF);
  final xSc = xS.clip(min: zeroF, max: maxXF);
  yS.dispose();
  xS.dispose();

  final y0F = ySc.floor(); // [targetH]
  final x0F = xSc.floor(); // [targetW]

  // Fractional weights: [targetH,1,1] and [1,targetW,1] for broadcasting.
  final wy = (ySc - y0F).reshape([targetH, 1, 1]); // [targetH,1,1]
  final wx = (xSc - x0F).reshape([1, targetW, 1]); // [1,targetW,1]
  ySc.dispose();
  xSc.dispose();

  final one = MLXArray.float_(ctx, 1.0);
  final wy1 = one - wy; // [targetH,1,1]
  final wx1 = one - wx; // [1,targetW,1]
  one.dispose();

  // y1 = clamp(y0 + 1, 0, srcH-1)
  final oneI = MLXArray.float_(ctx, 1.0);
  final y1F = (y0F + oneI).clip(min: zeroF, max: maxYF);
  final x1F = (x0F + oneI).clip(min: zeroF, max: maxXF);
  oneI.dispose();
  zeroF.dispose();
  maxYF.dispose();
  maxXF.dispose();

  // Integer indices for gather.
  final y0I = y0F.astype(MLXDtype.int32);
  final y1I = y1F.astype(MLXDtype.int32);
  final x0I = x0F.astype(MLXDtype.int32);
  final x1I = x1F.astype(MLXDtype.int32);
  y0F.dispose();
  y1F.dispose();
  x0F.dispose();
  x1F.dispose();

  // Gather corner rows then columns.
  // take(y, axis=0): [srcH,W,C] → [targetH,W,C]
  // .take(x, axis=1): [targetH,srcW,C] → [targetH,targetW,C]
  final r0 = image.take(y0I, axis: 0); // [targetH, srcW, C]
  final r1 = image.take(y1I, axis: 0);
  y0I.dispose();
  y1I.dispose();

  final topLeft = r0.take(x0I, axis: 1); // [targetH, targetW, C]
  final topRight = r0.take(x1I, axis: 1);
  r0.dispose();
  final botLeft = r1.take(x0I, axis: 1);
  final botRight = r1.take(x1I, axis: 1);
  r1.dispose();
  x0I.dispose();
  x1I.dispose();

  // Bilinear blend.
  // result = wy1*wx1*tl + wy1*wx*tr + wy*wx1*bl + wy*wx*br
  final tl = wy1 * wx1 * topLeft;
  topLeft.dispose();
  final tr = wy1 * wx * topRight;
  topRight.dispose();
  final bl = wy * wx1 * botLeft;
  botLeft.dispose();
  final br = wy * wx * botRight;
  botRight.dispose();
  wy.dispose();
  wx.dispose();
  wy1.dispose();
  wx1.dispose();

  final result = tl + tr + bl + br;
  tl.dispose();
  tr.dispose();
  bl.dispose();
  br.dispose();
  return result;
}

/// Inner (dot) product of two 1-D arrays, or sum of element-wise products for higher rank.
MLXArray inner(MLXContext ctx, MLXArray a, MLXArray b) =>
    MLXArray._opChecked(
        ctx, (out) => ctx.bindings.mlx_inner(out, a._raw, b._raw, ctx.stream));

/// Outer product of two 1-D arrays.
MLXArray outer(MLXContext ctx, MLXArray a, MLXArray b) =>
    MLXArray._opChecked(
        ctx, (out) => ctx.bindings.mlx_outer(out, a._raw, b._raw, ctx.stream));

/// Kronecker product.
MLXArray kron(MLXContext ctx, MLXArray a, MLXArray b) =>
    MLXArray._opChecked(
        ctx, (out) => ctx.bindings.mlx_kron(out, a._raw, b._raw, ctx.stream));

/// Einsum with subscript string, e.g. `'ij,jk->ik'`.
MLXArray einsum(MLXContext ctx, String subscripts, List<MLXArray> operands) {
  assert(operands.isNotEmpty);
  final subsPtr = subscripts.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final vec = calloc<mlx_vector_array>();
  try {
    vec.ref = ctx.bindings.mlx_vector_array_new();
    for (final a in operands) {
      ctx.check(ctx.bindings.mlx_vector_array_append_value(vec.ref, a._raw),
          'mlx_vector_array_append_value');
    }
    return MLXArray._opChecked(
        ctx, (out) => ctx.bindings.mlx_einsum(out, subsPtr, vec.ref, ctx.stream));
  } finally {
    calloc.free(subsPtr);
    ctx.bindings.mlx_vector_array_free(vec.ref);
    calloc.free(vec);
  }
}

/// Tensor dot product along [axis] pairs (default: last of a vs first of b).
MLXArray tensordot(
  MLXContext ctx,
  MLXArray a,
  MLXArray b, {
  int axes = 2,
}) =>
    MLXArray._opChecked(ctx,
        (out) => ctx.bindings.mlx_tensordot_axis(out, a._raw, b._raw, axes, ctx.stream));

/// 1-D convolution — input `[B, L, C]`, weight `[outC, kW, inC]`.
MLXArray conv1d(
  MLXContext ctx,
  MLXArray input,
  MLXArray weight, {
  int stride = 1,
  int padding = 0,
  int dilation = 1,
  int groups = 1,
}) =>
    MLXArray._opChecked(
        ctx,
        (out) => ctx.bindings.mlx_conv1d(
            out, input._raw, weight._raw, stride, padding, dilation, groups, ctx.stream));

/// Transposed 2-D convolution — input `[B, H, W, C]`, weight `[outC, kH, kW, inC]`.
MLXArray convTranspose2d(
  MLXContext ctx,
  MLXArray input,
  MLXArray weight, {
  int stride = 1,
  int padding = 0,
  int dilation = 1,
  int outputPadding = 0,
  int groups = 1,
}) =>
    MLXArray._opChecked(
        ctx,
        (out) => ctx.bindings.mlx_conv_transpose2d(
            out,
            input._raw,
            weight._raw,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation,
            outputPadding,
            outputPadding,
            groups,
            ctx.stream));

/// Transposed 1-D convolution — input `[B, L, C]`, weight `[outC, kW, inC]`.
MLXArray convTranspose1d(
  MLXContext ctx,
  MLXArray input,
  MLXArray weight, {
  int stride = 1,
  int padding = 0,
  int dilation = 1,
  int outputPadding = 0,
  int groups = 1,
}) =>
    MLXArray._opChecked(
        ctx,
        (out) => ctx.bindings.mlx_conv_transpose1d(out, input._raw, weight._raw,
            stride, padding, dilation, outputPadding, groups, ctx.stream));

/// Argmax sampling — greedy / deterministic.
MLXArray argmaxSample(MLXContext ctx, MLXArray logits) =>
    logits.argmax(axis: -1);

/// Temperature-scaled categorical sampling.
MLXArray categoricalSample(
    MLXContext ctx, MLXArray logits, double temperature) {
  final temp = MLXArray.float_(ctx, temperature);
  final scaled = logits / temp;
  temp.dispose();
  // Pass a zero-initialised key struct → MLX uses its global PRNG state.
  final keyPtr = calloc<mlx_array>();
  final result = MLXArray._opChecked(
      ctx,
      (out) => ctx.bindings.mlx_random_categorical(
          out, scaled._raw, -1, keyPtr.ref, ctx.stream));
  calloc.free(keyPtr);
  scaled.dispose();
  return result;
}
