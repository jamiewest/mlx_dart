import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mlx_c_ffi/mlx_c_ffi.dart';

/// Execution context wrapping MLX bindings and a default compute stream.
///
/// [MLXContext] is the entry-point for creating arrays and running operations.
/// It corresponds to the implicit global context in mlx-swift where the
/// default stream is managed automatically.
///
/// A single context is typically shared across an entire model.
final class MLXContext {
  MLXContext._(this.bindings, this.stream);

  final MlxCBindings bindings;
  final mlx_stream stream;
  bool _disposed = false;

  /// Opens a context using the GPU (Metal) default stream.
  factory MLXContext.gpu({String? libraryPath}) {
    final dylib = _openDylib(libraryPath);
    final b = MlxCBindings(dylib);
    final s = b.mlx_default_gpu_stream_new();
    return MLXContext._(b, s);
  }

  /// Opens a context using the CPU default stream.
  factory MLXContext.cpu({String? libraryPath}) {
    final dylib = _openDylib(libraryPath);
    final b = MlxCBindings(dylib);
    final s = b.mlx_default_cpu_stream_new();
    return MLXContext._(b, s);
  }

  String get version {
    final out = calloc<mlx_string>();
    try {
      check(bindings.mlx_version(out), 'mlx_version');
      return readString(out.ref);
    } finally {
      calloc.free(out);
    }
  }

  void synchronize() {
    check(bindings.mlx_synchronize(stream), 'mlx_synchronize');
  }

  // ---------------------------------------------------------------------------
  // Memory management
  // ---------------------------------------------------------------------------

  /// Active GPU memory currently in use, in bytes.
  int get activeMemory {
    final out = calloc<ffi.Size>();
    try {
      check(bindings.mlx_get_active_memory(out), 'mlx_get_active_memory');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// GPU memory held in the cache (available for reuse), in bytes.
  int get cacheMemory {
    final out = calloc<ffi.Size>();
    try {
      check(bindings.mlx_get_cache_memory(out), 'mlx_get_cache_memory');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Peak GPU memory usage since the last [resetPeakMemory] call, in bytes.
  int get peakMemory {
    final out = calloc<ffi.Size>();
    try {
      check(bindings.mlx_get_peak_memory(out), 'mlx_get_peak_memory');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Current memory limit in bytes.
  int get memoryLimit {
    final out = calloc<ffi.Size>();
    try {
      check(bindings.mlx_get_memory_limit(out), 'mlx_get_memory_limit');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Free all cached GPU memory.
  void clearCache() {
    check(bindings.mlx_clear_cache(), 'mlx_clear_cache');
  }

  /// Reset the peak memory counter to zero.
  void resetPeakMemory() {
    check(bindings.mlx_reset_peak_memory(), 'mlx_reset_peak_memory');
  }

  /// Set the memory limit. Returns the previous limit in bytes.
  int setMemoryLimit(int bytes) {
    final out = calloc<ffi.Size>();
    try {
      check(
        bindings.mlx_set_memory_limit(out, bytes),
        'mlx_set_memory_limit',
      );
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Set the cache limit. Returns the previous limit in bytes.
  int setCacheLimit(int bytes) {
    final out = calloc<ffi.Size>();
    try {
      check(bindings.mlx_set_cache_limit(out, bytes), 'mlx_set_cache_limit');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    check(bindings.mlx_stream_free(stream), 'mlx_stream_free');
  }

  // ---------------------------------------------------------------------------
  // Internal helpers used by MLXArray and ops.
  // ---------------------------------------------------------------------------

  /// Throws [MLXException] if [code] is non-zero.
  void check(int code, String op) {
    if (code != 0) throw MLXException(op, code);
  }

  /// Reads an [mlx_string] value into a Dart [String] and frees the source.
  String readString(mlx_string str) {
    try {
      final data = bindings.mlx_string_data(str);
      if (data.address == 0) return '';
      return data.cast<Utf8>().toDartString();
    } finally {
      bindings.mlx_string_free(str);
    }
  }

  static ffi.DynamicLibrary _openDylib(String? path) {
    if (path != null) return ffi.DynamicLibrary.open(path);
    // Walk the default search paths the same way mlx_ffi does.
    try {
      return ffi.DynamicLibrary.open('libmlxc.dylib');
    } on ArgumentError {
      return ffi.DynamicLibrary.open('libmlx.dylib');
    }
  }
}

/// Thrown when an MLX C operation returns a non-zero error code.
final class MLXException implements Exception {
  const MLXException(this.operation, this.code, [this.message]);

  final String operation;
  final int code;
  final String? message;

  @override
  String toString() {
    final msg = message != null ? ', message=$message' : '';
    return 'MLXException($operation): code=$code$msg';
  }
}
