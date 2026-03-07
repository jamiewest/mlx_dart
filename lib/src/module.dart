import 'array.dart';
import 'context.dart';

/// Base class for all neural network modules, analogous to `Module` in mlx-swift.
///
/// Subclasses declare their parameters as public [MLXArray] fields. The
/// [loadWeights] mechanism walks a weight map and assigns matching arrays,
/// supporting hierarchical dot-separated keys (e.g. `'layers.0.weight'`).
abstract class Module {
  const Module();

  /// Human-readable name for debugging.
  String get name => runtimeType.toString();

  /// Recursively collect all [MLXArray] parameters keyed by their dotted path.
  ///
  /// Override if the default reflection-based approach is insufficient.
  Map<String, MLXArray> parameters() => const {};

  /// Load weights from a flat map of dotted-path keys to arrays.
  ///
  /// Unknown keys are silently ignored; this mirrors the Swift behaviour where
  /// extra weights in a checkpoint are harmless.
  void loadWeights(Map<String, MLXArray> weights) {}

  /// Sanitise a raw weight map before loading.
  ///
  /// Override this to rename, delete or transform weights as needed (e.g. to
  /// strip a model-specific prefix or fuse QKV projections).
  Map<String, MLXArray> sanitize(Map<String, MLXArray> weights) => weights;

  /// Free all arrays owned by this module.
  void dispose() {}
}

/// Mixin that makes a [Module]'s parameter map type-safe and discoverable.
///
/// Extend this in leaf modules to get automatic weight-loading by tracking
/// named parameters in a simple [Map]. This is the idiomatic Dart equivalent
/// of the Swift `Module.parameters()` machinery.
mixin ParametersMixin on Module {
  final Map<String, MLXArray> _params = {};
  final Map<String, Module> _children = {};

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    // Own params
    result.addAll(_params);
    // Recurse into children
    for (final entry in _children.entries) {
      for (final p in entry.value.parameters().entries) {
        result['${entry.key}.${p.key}'] = p.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    for (final entry in weights.entries) {
      _setParameter(entry.key.split('.'), entry.value);
    }
  }

  void _setParameter(List<String> path, MLXArray value) {
    if (path.isEmpty) return;
    if (path.length == 1) {
      _params[path.first] = value;
      return;
    }
    final child = _children[path.first];
    if (child != null) {
      child._setParameter(path.sublist(1), value);
    }
  }

  @override
  void dispose() {
    for (final v in _params.values) {
      v.dispose();
    }
    for (final c in _children.values) {
      c.dispose();
    }
  }
}

extension on Module {
  void _setParameter(List<String> path, MLXArray value) {
    if (this case final ParametersMixin m) {
      m._setParameter(path, value);
    }
  }
}

/// Context passed to [Module.loadWeights] to allow the module to resolve
/// its weights from a nested map structure.
///
/// This mirrors the Swift approach where models implement `sanitize(weights:)`
/// to strip/rename keys before the loader assigns them.
final class WeightLoader {
  const WeightLoader(this._ctx, this._weights);

  final MLXContext _ctx;
  final Map<String, MLXArray> _weights;

  MLXArray? operator [](String key) => _weights[key];

  bool containsKey(String key) => _weights.containsKey(key);

  WeightLoader scoped(String prefix) {
    final filtered = <String, MLXArray>{};
    for (final entry in _weights.entries) {
      if (entry.key.startsWith('$prefix.')) {
        filtered[entry.key.substring(prefix.length + 1)] = entry.value;
      }
    }
    return WeightLoader(_ctx, filtered);
  }

  MLXContext get context => _ctx;
}
