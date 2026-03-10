import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mlx_c_ffi/mlx_c_ffi.dart';

import 'array.dart';
import 'context.dart';

/// Load and save arrays in the SafeTensors format.
///
/// SafeTensors is the primary checkpoint format used by Hugging Face models.
/// The file extension is typically `.safetensors`.
abstract final class MLXSafetensors {
  MLXSafetensors._();

  /// Load arrays from a SafeTensors [file].
  ///
  /// Returns a map of tensor name → [MLXArray]. Metadata (string→string) is
  /// discarded; use [loadWithMetadata] if you need it.
  static Map<String, MLXArray> load(MLXContext ctx, String file) {
    final (arrays, _) = loadWithMetadata(ctx, file);
    return arrays;
  }

  /// Load arrays and metadata from a SafeTensors [file].
  ///
  /// Returns `(arrays, metadata)`.
  static (Map<String, MLXArray>, Map<String, String>) loadWithMetadata(
      MLXContext ctx, String file) {
    final filePtr = file.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final arrMap = calloc<mlx_map_string_to_array>();
    final strMap = calloc<mlx_map_string_to_string>();
    try {
      ctx.check(
          ctx.bindings.mlx_load_safetensors(
              arrMap, strMap, filePtr, ctx.stream),
          'mlx_load_safetensors');
      final arrays = _readArrayMap(ctx, arrMap.ref);
      final metadata = _readStringMap(ctx, strMap.ref);
      return (arrays, metadata);
    } finally {
      calloc.free(filePtr);
      ctx.bindings.mlx_map_string_to_array_free(arrMap.ref);
      ctx.bindings.mlx_map_string_to_string_free(strMap.ref);
      calloc.free(arrMap);
      calloc.free(strMap);
    }
  }

  /// Save [arrays] to a SafeTensors [file].
  ///
  /// Optional [metadata] key-value strings are embedded in the file header.
  static void save(
    MLXContext ctx,
    String file,
    Map<String, MLXArray> arrays, {
    Map<String, String> metadata = const {},
  }) {
    final filePtr = file.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final arrMap = ctx.bindings.mlx_map_string_to_array_new();
    final strMap = ctx.bindings.mlx_map_string_to_string_new();
    final keyPtrs = <ffi.Pointer<ffi.Char>>[];
    try {
      for (final entry in arrays.entries) {
        final k = entry.key.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
        keyPtrs.add(k);
        ctx.check(
            ctx.bindings.mlx_map_string_to_array_insert(arrMap, k, entry.value.raw),
            'mlx_map_string_to_array_insert');
      }
      for (final entry in metadata.entries) {
        final k = entry.key.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
        final v = entry.value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
        keyPtrs.add(k);
        keyPtrs.add(v);
        ctx.check(
            ctx.bindings.mlx_map_string_to_string_insert(strMap, k, v),
            'mlx_map_string_to_string_insert');
      }
      ctx.check(
          ctx.bindings.mlx_save_safetensors(filePtr, arrMap, strMap),
          'mlx_save_safetensors');
    } finally {
      calloc.free(filePtr);
      for (final p in keyPtrs) {
        calloc.free(p);
      }
      ctx.bindings.mlx_map_string_to_array_free(arrMap);
      ctx.bindings.mlx_map_string_to_string_free(strMap);
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  static Map<String, MLXArray> _readArrayMap(
      MLXContext ctx, mlx_map_string_to_array map) {
    final result = <String, MLXArray>{};
    final it = ctx.bindings.mlx_map_string_to_array_iterator_new(map);
    final keyPtr = calloc<ffi.Pointer<ffi.Char>>();
    final valPtr = calloc<mlx_array>();
    try {
      while (ctx.bindings
              .mlx_map_string_to_array_iterator_next(keyPtr, valPtr, it) ==
          0) {
        final key = keyPtr.value.cast<Utf8>().toDartString();
        final arrPtr = calloc<mlx_array>();
        arrPtr.ref = valPtr.ref;
        result[key] = MLXArray.owned(ctx, arrPtr);
      }
    } finally {
      ctx.bindings.mlx_map_string_to_array_iterator_free(it);
      calloc.free(keyPtr);
      calloc.free(valPtr);
    }
    return result;
  }

  static Map<String, String> _readStringMap(
      MLXContext ctx, mlx_map_string_to_string map) {
    final result = <String, String>{};
    final it = ctx.bindings.mlx_map_string_to_string_iterator_new(map);
    final keyPtr = calloc<ffi.Pointer<ffi.Char>>();
    final valPtr = calloc<ffi.Pointer<ffi.Char>>();
    try {
      while (ctx.bindings
              .mlx_map_string_to_string_iterator_next(keyPtr, valPtr, it) ==
          0) {
        final key = keyPtr.value.cast<Utf8>().toDartString();
        final value = valPtr.value.cast<Utf8>().toDartString();
        result[key] = value;
      }
    } finally {
      ctx.bindings.mlx_map_string_to_string_iterator_free(it);
      calloc.free(keyPtr);
      calloc.free(valPtr);
    }
    return result;
  }
}
