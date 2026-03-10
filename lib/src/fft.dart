import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mlx_c_ffi/mlx_c_ffi.dart';

import 'array.dart';
import 'context.dart';

/// Fast Fourier Transform operations — mirrors `MLXFFT` in mlx-swift.
///
/// All functions return complex arrays except the real-valued variants
/// (`rfft`, `rfft2`, `rfftn`) which return half-spectrum arrays.
abstract final class MLXFFT {
  MLXFFT._();

  /// 1-D FFT along [axis]. [n] truncates/pads; defaults to axis size.
  static MLXArray fft(MLXContext ctx, MLXArray a, {int n = -1, int axis = -1}) =>
      _op1(ctx, a, n, axis, ctx.bindings.mlx_fft_fft);

  /// 1-D inverse FFT along [axis].
  static MLXArray ifft(MLXContext ctx, MLXArray a, {int n = -1, int axis = -1}) =>
      _op1(ctx, a, n, axis, ctx.bindings.mlx_fft_ifft);

  /// 1-D real FFT (output is half-spectrum).
  static MLXArray rfft(MLXContext ctx, MLXArray a, {int n = -1, int axis = -1}) =>
      _op1(ctx, a, n, axis, ctx.bindings.mlx_fft_rfft);

  /// 1-D inverse real FFT.
  static MLXArray irfft(MLXContext ctx, MLXArray a, {int n = -1, int axis = -1}) =>
      _op1(ctx, a, n, axis, ctx.bindings.mlx_fft_irfft);

  /// 2-D FFT. [ns] and [axes] default to last two axes.
  static MLXArray fft2(
    MLXContext ctx,
    MLXArray a, {
    List<int>? ns,
    List<int>? axes,
  }) =>
      _op2(ctx, a, ns, axes, ctx.bindings.mlx_fft_fft2);

  /// 2-D inverse FFT.
  static MLXArray ifft2(
    MLXContext ctx,
    MLXArray a, {
    List<int>? ns,
    List<int>? axes,
  }) =>
      _op2(ctx, a, ns, axes, ctx.bindings.mlx_fft_ifft2);

  /// 2-D real FFT.
  static MLXArray rfft2(
    MLXContext ctx,
    MLXArray a, {
    List<int>? ns,
    List<int>? axes,
  }) =>
      _op2(ctx, a, ns, axes, ctx.bindings.mlx_fft_rfft2);

  /// 2-D inverse real FFT.
  static MLXArray irfft2(
    MLXContext ctx,
    MLXArray a, {
    List<int>? ns,
    List<int>? axes,
  }) =>
      _op2(ctx, a, ns, axes, ctx.bindings.mlx_fft_irfft2);

  /// N-D FFT over [axes] (default: all).
  static MLXArray fftn(
    MLXContext ctx,
    MLXArray a, {
    List<int>? ns,
    List<int>? axes,
  }) =>
      _op2(ctx, a, ns, axes, ctx.bindings.mlx_fft_fftn);

  /// N-D inverse FFT.
  static MLXArray ifftn(
    MLXContext ctx,
    MLXArray a, {
    List<int>? ns,
    List<int>? axes,
  }) =>
      _op2(ctx, a, ns, axes, ctx.bindings.mlx_fft_ifftn);

  // ---------------------------------------------------------------------------
  // Internal dispatch helpers
  // ---------------------------------------------------------------------------

  static MLXArray _op1(
    MLXContext ctx,
    MLXArray a,
    int n,
    int axis,
    int Function(ffi.Pointer<mlx_array>, mlx_array, int, int, mlx_stream) fn,
  ) {
    final out = calloc<mlx_array>();
    final code = fn(out, a.raw, n, axis, ctx.stream);
    if (code != 0) {
      ctx.bindings.mlx_array_free(out.ref);
      calloc.free(out);
      throw MLXException('mlx_fft', code);
    }
    return MLXArray.owned(ctx, out);
  }

  static MLXArray _op2(
    MLXContext ctx,
    MLXArray a,
    List<int>? ns,
    List<int>? axes,
    int Function(ffi.Pointer<mlx_array>, mlx_array, ffi.Pointer<ffi.Int>, int,
            ffi.Pointer<ffi.Int>, int, mlx_stream)
        fn,
  ) {
    final nsPtr = ns != null ? _allocIntPtr(ns) : ffi.nullptr.cast<ffi.Int>();
    final axesPtr =
        axes != null ? _allocIntPtr(axes) : ffi.nullptr.cast<ffi.Int>();
    try {
      final out = calloc<mlx_array>();
      final code = fn(out, a.raw, nsPtr, ns?.length ?? 0, axesPtr,
          axes?.length ?? 0, ctx.stream);
      if (code != 0) {
        ctx.bindings.mlx_array_free(out.ref);
        calloc.free(out);
        throw MLXException('mlx_fft2', code);
      }
      return MLXArray.owned(ctx, out);
    } finally {
      if (ns != null) calloc.free(nsPtr);
      if (axes != null) calloc.free(axesPtr);
    }
  }

  static ffi.Pointer<ffi.Int> _allocIntPtr(List<int> values) {
    final ptr = calloc<ffi.Int>(values.isEmpty ? 1 : values.length);
    for (var i = 0; i < values.length; i++) {
      ptr[i] = values[i];
    }
    return ptr;
  }
}
