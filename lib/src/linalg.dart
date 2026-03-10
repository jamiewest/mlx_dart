import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mlx_c_ffi/mlx_c_ffi.dart';

import 'array.dart';
import 'context.dart';

/// Linear algebra operations — mirrors `MLXLinalg` in mlx-swift.
abstract final class MLXLinalg {
  MLXLinalg._();

  /// Vector or matrix norm.
  ///
  /// [ord] selects the norm type (2.0 = L2/Frobenius, 1.0 = L1, etc.).
  /// Pass [axis] to compute along specific axes; [keepdims] preserves rank.
  static MLXArray norm(
    MLXContext ctx,
    MLXArray a, {
    double ord = 2.0,
    List<int>? axis,
    bool keepdims = false,
  }) {
    final axisPtr = axis != null ? _allocIntPtr(axis) : ffi.nullptr.cast<ffi.Int>();
    final axisLen = axis?.length ?? 0;
    try {
      return _opChecked(
          ctx,
          (out) => ctx.bindings.mlx_linalg_norm(
              out, a.raw, ord, axisPtr, axisLen, keepdims, ctx.stream));
    } finally {
      if (axis != null) calloc.free(axisPtr);
    }
  }

  /// Singular value decomposition. Returns `[U, S, Vt]` when [computeUV] is
  /// true (default), or `[S]` otherwise.
  static List<MLXArray> svd(
    MLXContext ctx,
    MLXArray a, {
    bool computeUV = true,
  }) {
    final vec = calloc<mlx_vector_array>();
    try {
      ctx.check(
          ctx.bindings.mlx_linalg_svd(vec, a.raw, computeUV, ctx.stream),
          'mlx_linalg_svd');
      final count = ctx.bindings.mlx_vector_array_size(vec.ref);
      return List.generate(count, (i) {
        final ptr = calloc<mlx_array>();
        ctx.check(ctx.bindings.mlx_vector_array_get(ptr, vec.ref, i),
            'mlx_vector_array_get');
        return MLXArray.owned(ctx, ptr);
      });
    } finally {
      ctx.bindings.mlx_vector_array_free(vec.ref);
      calloc.free(vec);
    }
  }

  /// QR decomposition. Returns `(Q, R)`.
  static (MLXArray, MLXArray) qr(MLXContext ctx, MLXArray a) {
    final q = calloc<mlx_array>();
    final r = calloc<mlx_array>();
    ctx.check(ctx.bindings.mlx_linalg_qr(q, r, a.raw, ctx.stream),
        'mlx_linalg_qr');
    return (MLXArray.owned(ctx, q), MLXArray.owned(ctx, r));
  }

  /// Matrix inverse.
  static MLXArray inv(MLXContext ctx, MLXArray a) => _opChecked(
      ctx, (out) => ctx.bindings.mlx_linalg_inv(out, a.raw, ctx.stream));

  /// Cross product along [axis] (default -1).
  static MLXArray cross(
    MLXContext ctx,
    MLXArray a,
    MLXArray b, {
    int axis = -1,
  }) =>
      _opChecked(ctx,
          (out) => ctx.bindings.mlx_linalg_cross(out, a.raw, b.raw, axis, ctx.stream));

  /// Cholesky decomposition. [upper] returns the upper triangle when true.
  static MLXArray cholesky(
    MLXContext ctx,
    MLXArray a, {
    bool upper = false,
  }) =>
      _opChecked(ctx,
          (out) => ctx.bindings.mlx_linalg_cholesky(out, a.raw, upper, ctx.stream));

  /// Solve the linear system `a @ x = b`.
  static MLXArray solve(MLXContext ctx, MLXArray a, MLXArray b) => _opChecked(
      ctx,
      (out) => ctx.bindings.mlx_linalg_solve(out, a.raw, b.raw, ctx.stream));

  /// LU factorisation. Returns `[P, L, U]` where `P @ L @ U = a`.
  static List<MLXArray> lu(MLXContext ctx, MLXArray a) {
    final vec = calloc<mlx_vector_array>();
    try {
      ctx.check(
          ctx.bindings.mlx_linalg_lu(vec, a.raw, ctx.stream),
          'mlx_linalg_lu');
      final count = ctx.bindings.mlx_vector_array_size(vec.ref);
      return List.generate(count, (i) {
        final ptr = calloc<mlx_array>();
        ctx.check(ctx.bindings.mlx_vector_array_get(ptr, vec.ref, i),
            'mlx_vector_array_get');
        return MLXArray.owned(ctx, ptr);
      });
    } finally {
      ctx.bindings.mlx_vector_array_free(vec.ref);
      calloc.free(vec);
    }
  }

  /// Inverse of a triangular matrix.
  /// [upper] selects whether [a] is upper (true) or lower (false) triangular.
  static MLXArray triInv(MLXContext ctx, MLXArray a, {bool upper = false}) =>
      _opChecked(ctx,
          (out) => ctx.bindings.mlx_linalg_tri_inv(out, a.raw, upper, ctx.stream));

  /// Inverse from a Cholesky factor.
  static MLXArray choleskyInv(MLXContext ctx, MLXArray a, {bool upper = false}) =>
      _opChecked(ctx,
          (out) => ctx.bindings.mlx_linalg_cholesky_inv(out, a.raw, upper, ctx.stream));

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  static MLXArray _opChecked(
    MLXContext ctx,
    int Function(ffi.Pointer<mlx_array>) fn,
  ) {
    final out = calloc<mlx_array>();
    final code = fn(out);
    if (code != 0) {
      ctx.bindings.mlx_array_free(out.ref);
      calloc.free(out);
      throw MLXException('mlx_linalg', code);
    }
    return MLXArray.owned(ctx, out);
  }

  static ffi.Pointer<ffi.Int> _allocIntPtr(List<int> values) {
    final ptr = calloc<ffi.Int>(values.isEmpty ? 1 : values.length);
    for (var i = 0; i < values.length; i++) {
      ptr[i] = values[i];
    }
    return ptr;
  }
}
