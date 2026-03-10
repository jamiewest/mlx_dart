import 'array.dart';
import 'context.dart';

/// Key-value cache for transformer attention layers.
///
/// Stores past key and value arrays so that auto-regressive decoding only
/// attends to new tokens rather than recomputing the full sequence.
/// Mirrors `mlx-swift`'s `KVCache`.
class KVCache {
  KVCache(this._ctx);

  final MLXContext _ctx;
  MLXContext get ctx => _ctx;

  MLXArray? _keys;
  MLXArray? _values;

  /// Number of tokens currently stored in the cache.
  int get offset => _offset;
  int _offset = 0;

  /// Whether the cache has been populated with at least one step.
  bool get isInitialized => _keys != null;

  /// Update the cache with [newKeys] and [newValues] (shape `[B, H, S, D]`),
  /// then return the full accumulated keys and values.
  ///
  /// On the first call the arrays are stored as-is.  On subsequent calls they
  /// are concatenated along the sequence dimension (axis 2).
  (MLXArray, MLXArray) update(MLXArray newKeys, MLXArray newValues) {
    if (_keys == null) {
      _keys = newKeys;
      _values = newValues;
      _offset = newKeys.shape[2];
    } else {
      final prevK = _keys!;
      final prevV = _values!;
      _keys = concatenate(_ctx, [prevK, newKeys], axis: 2);
      _values = concatenate(_ctx, [prevV, newValues], axis: 2);
      prevK.dispose();
      prevV.dispose();
      _offset = _keys!.shape[2];
    }
    return (_keys!, _values!);
  }

  /// Release all cached arrays.
  void dispose() {
    _keys?.dispose();
    _values?.dispose();
    _keys = null;
    _values = null;
    _offset = 0;
  }
}

/// Rotating key-value cache with a fixed maximum capacity.
///
/// Once [maxSize] tokens have been cached the oldest entries are evicted
/// in FIFO order, keeping memory bounded.  Mirrors `mlx-swift`'s
/// `RotatingKVCache`.
class RotatingKVCache {
  RotatingKVCache(this._ctx, {required this.maxSize});

  final MLXContext _ctx;
  MLXContext get ctx => _ctx;

  /// Maximum number of past tokens to retain.
  final int maxSize;

  MLXArray? _keys;
  MLXArray? _values;
  int _offset = 0;

  /// Total number of tokens that have passed through this cache.
  int get offset => _offset;

  /// Whether the cache has been populated.
  bool get isInitialized => _keys != null;

  /// Current number of tokens stored in the cache (≤ [maxSize]).
  int get size => _keys == null ? 0 : _keys!.shape[2];

  /// Update with [newKeys] / [newValues] (shape `[B, H, S, D]`).
  ///
  /// Returns the (possibly truncated) accumulated keys and values.
  (MLXArray, MLXArray) update(MLXArray newKeys, MLXArray newValues) {
    _offset += newKeys.shape[2];

    if (_keys == null) {
      // First call — store directly, slicing to maxSize if needed.
      if (newKeys.shape[2] <= maxSize) {
        _keys = newKeys;
        _values = newValues;
      } else {
        final start = newKeys.shape[2] - maxSize;
        _keys = _sliceSeq(newKeys, start, newKeys.shape[2]);
        _values = _sliceSeq(newValues, start, newValues.shape[2]);
      }
    } else {
      final prevK = _keys!;
      final prevV = _values!;
      final catK = concatenate(_ctx, [prevK, newKeys], axis: 2);
      final catV = concatenate(_ctx, [prevV, newValues], axis: 2);
      prevK.dispose();
      prevV.dispose();

      if (catK.shape[2] > maxSize) {
        final start = catK.shape[2] - maxSize;
        _keys = _sliceSeq(catK, start, catK.shape[2]);
        _values = _sliceSeq(catV, start, catV.shape[2]);
        catK.dispose();
        catV.dispose();
      } else {
        _keys = catK;
        _values = catV;
      }
    }

    return (_keys!, _values!);
  }

  /// Release all cached arrays.
  void dispose() {
    _keys?.dispose();
    _values?.dispose();
    _keys = null;
    _values = null;
    _offset = 0;
  }

  // Slice along the sequence axis (axis=2): result has shape [B, H, end-start, D].
  MLXArray _sliceSeq(MLXArray a, int start, int end) {
    final shape = a.shape; // [B, H, S, D]
    return a.slice(
      start: [0, 0, start, 0],
      stop: [shape[0], shape[1], end, shape[3]],
    );
  }
}
