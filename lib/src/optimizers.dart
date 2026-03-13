import 'dart:math' as math;

import 'array.dart';
import 'context.dart';

// ignore_for_file: public_member_api_docs

/// Base class for all MLX optimizers, mirroring `mlx-swift`'s `Optimizer`.
///
/// An optimizer takes a map of named gradients and a map of current parameters,
/// and returns updated parameters.  Call [applyGradients] after computing
/// gradients (e.g. via a manual finite-difference loop or a future `grad`
/// transform).
abstract class Optimizer {
  Optimizer(this._ctx);

  final MLXContext _ctx;
  MLXContext get ctx => _ctx;

  /// Apply [gradients] to [parameters] and return the updated parameters.
  ///
  /// Both maps use the same dotted-path keys as [Module.parameters].
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  );
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------

/// Stochastic gradient descent, optionally with momentum and weight decay.
///
/// ```
/// w_{t+1} = w_t - lr * (g_t + wd * w_t)          // no momentum
/// v_{t+1} = m * v_t + g_t                          // momentum buffer
/// w_{t+1} = w_t - lr * v_{t+1}                    // with momentum
/// ```
final class SGD extends Optimizer {
  SGD(
    super.ctx, {
    required this.learningRate,
    this.momentum = 0.0,
    this.weightDecay = 0.0,
  });

  double learningRate;
  double momentum;
  double weightDecay;

  // State: velocity buffers per parameter key.
  final Map<String, MLXArray> _velocities = {};

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    final updated = <String, MLXArray>{};

    for (final entry in gradients.entries) {
      final key = entry.key;
      var g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      // Weight decay.
      if (weightDecay != 0.0) {
        final wd = MLXArray.float_(ctx, weightDecay);
        g = g + wd * w;
        wd.dispose();
      }

      MLXArray newW;
      if (momentum != 0.0) {
        final prev = _velocities[key];
        final v = prev != null
            ? _axpby(momentum, prev, 1.0, g)
            : g;
        _velocities[key]?.dispose();
        _velocities[key] = v;
        newW = _axpby(1.0, w, -learningRate, v);
      } else {
        newW = _axpby(1.0, w, -learningRate, g);
      }

      updated[key] = newW;
    }

    return updated;
  }

  // Returns a * x + b * y.
  MLXArray _axpby(double a, MLXArray x, double b, MLXArray y) {
    final aArr = MLXArray.float_(ctx, a);
    final bArr = MLXArray.float_(ctx, b);
    final ax = aArr * x;
    final by = bArr * y;
    final result = ax + by;
    aArr.dispose();
    bArr.dispose();
    ax.dispose();
    by.dispose();
    return result;
  }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/// Adam optimizer (Kingma & Ba, 2015).
///
/// ```
/// m_{t+1} = β1 * m_t + (1 - β1) * g_t
/// v_{t+1} = β2 * v_t + (1 - β2) * g_t²
/// m̂ = m_{t+1} / (1 - β1^t)
/// v̂ = v_{t+1} / (1 - β2^t)
/// w_{t+1} = w_t - lr * m̂ / (√v̂ + ε)
/// ```
class Adam extends Optimizer {
  Adam(
    super.ctx, {
    required this.learningRate,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
  });

  double learningRate;
  double beta1;
  double beta2;
  double epsilon;

  final Map<String, MLXArray> _m = {};
  final Map<String, MLXArray> _v = {};
  int _t = 0;

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    _t++;
    final updated = <String, MLXArray>{};

    final b1 = MLXArray.float_(ctx, beta1);
    final b2 = MLXArray.float_(ctx, beta2);
    final one = MLXArray.float_(ctx, 1.0);
    final b1c = MLXArray.float_(ctx, 1.0 - beta1);
    final b2c = MLXArray.float_(ctx, 1.0 - beta2);
    final bias1 = MLXArray.float_(ctx, 1.0 - _pow(beta1, _t));
    final bias2 = MLXArray.float_(ctx, 1.0 - _pow(beta2, _t));
    final eps = MLXArray.float_(ctx, epsilon);
    final lr = MLXArray.float_(ctx, learningRate);

    for (final entry in gradients.entries) {
      final key = entry.key;
      final g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      // m = β1*m + (1-β1)*g
      final prevM = _m[key];
      final m = prevM != null
          ? b1 * prevM + b1c * g
          : b1c * g;
      _m[key]?.dispose();
      _m[key] = m;

      // v = β2*v + (1-β2)*g²
      final gSq = g * g;
      final prevV = _v[key];
      final v = prevV != null
          ? b2 * prevV + b2c * gSq
          : b2c * gSq;
      gSq.dispose();
      _v[key]?.dispose();
      _v[key] = v;

      // bias-corrected
      final mHat = m / bias1;
      final vHat = v / bias2;

      // w = w - lr * mHat / (sqrt(vHat) + eps)
      final sqrtV = vHat.sqrt();
      final denom = sqrtV + eps;
      final step = lr * mHat / denom;
      final newW = w - step;

      mHat.dispose();
      vHat.dispose();
      sqrtV.dispose();
      denom.dispose();
      step.dispose();

      updated[key] = newW;
    }

    b1.dispose();
    b2.dispose();
    one.dispose();
    b1c.dispose();
    b2c.dispose();
    bias1.dispose();
    bias2.dispose();
    eps.dispose();
    lr.dispose();

    return updated;
  }

  static double _pow(double base, int exp) {
    var result = 1.0;
    for (var i = 0; i < exp; i++) {
      result *= base;
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

/// AdamW optimizer — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
///
/// Weight decay is applied directly to weights before the Adam update:
/// ```
/// w_{t+1} = (1 - lr * wd) * w_t - Adam update
/// ```
final class AdamW extends Adam {
  AdamW(
    super.ctx, {
    required super.learningRate,
    super.beta1,
    super.beta2,
    super.epsilon,
    this.weightDecay = 1e-2,
  });

  double weightDecay;

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    // Apply decoupled weight decay first.
    final decayed = <String, MLXArray>{};
    if (weightDecay != 0.0) {
      final factor = MLXArray.float_(ctx, 1.0 - learningRate * weightDecay);
      for (final entry in parameters.entries) {
        if (gradients.containsKey(entry.key)) {
          decayed[entry.key] = factor * entry.value;
        }
      }
      factor.dispose();
    }

    final base = decayed.isEmpty ? parameters : {...parameters, ...decayed};
    return super.applyGradients(gradients, base);
  }
}

// ---------------------------------------------------------------------------
// Adamax
// ---------------------------------------------------------------------------

/// Adamax optimizer — infinity-norm variant of Adam (Kingma & Ba, 2015).
final class Adamax extends Optimizer {
  Adamax(
    super.ctx, {
    required this.learningRate,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
  });

  double learningRate;
  double beta1;
  double beta2;
  double epsilon;

  final Map<String, MLXArray> _m = {};
  final Map<String, MLXArray> _u = {}; // max norm
  int _t = 0;

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    _t++;
    final updated = <String, MLXArray>{};

    final b1 = MLXArray.float_(ctx, beta1);
    final b1c = MLXArray.float_(ctx, 1.0 - beta1);
    final b2 = MLXArray.float_(ctx, beta2);
    final bias1 = MLXArray.float_(ctx, 1.0 - Adam._pow(beta1, _t));
    final eps = MLXArray.float_(ctx, epsilon);
    final lr = MLXArray.float_(ctx, learningRate);

    for (final entry in gradients.entries) {
      final key = entry.key;
      final g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      // m = β1*m + (1-β1)*g
      final prevM = _m[key];
      final m = prevM != null ? b1 * prevM + b1c * g : b1c * g;
      _m[key]?.dispose();
      _m[key] = m;

      // u = max(β2*u, |g|)
      final absGk = g.abs();
      final prevU = _u[key];
      final u = prevU != null
          ? (b2 * prevU).maximum(absGk)
          : absGk;
      absGk.dispose();
      _u[key]?.dispose();
      _u[key] = u;

      // w = w - (lr / (1-β1^t)) * m / (u + ε)
      final mHat = m / bias1;
      final denom = u + eps;
      final step = lr * mHat / denom;
      final newW = w - step;

      mHat.dispose();
      denom.dispose();
      step.dispose();

      updated[key] = newW;
    }

    b1.dispose();
    b1c.dispose();
    b2.dispose();
    bias1.dispose();
    eps.dispose();
    lr.dispose();

    return updated;
  }
}

// ---------------------------------------------------------------------------
// RMSprop
// ---------------------------------------------------------------------------

/// RMSprop optimizer (Hinton, 2012).
final class RMSprop extends Optimizer {
  RMSprop(
    super.ctx, {
    required this.learningRate,
    this.alpha = 0.99,
    this.epsilon = 1e-8,
    this.momentum = 0.0,
    this.weightDecay = 0.0,
  });

  double learningRate;
  double alpha;
  double epsilon;
  double momentum;
  double weightDecay;

  final Map<String, MLXArray> _v = {};
  final Map<String, MLXArray> _buf = {};

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    final updated = <String, MLXArray>{};

    final a = MLXArray.float_(ctx, alpha);
    final ac = MLXArray.float_(ctx, 1.0 - alpha);
    final eps = MLXArray.float_(ctx, epsilon);
    final lr = MLXArray.float_(ctx, learningRate);

    for (final entry in gradients.entries) {
      final key = entry.key;
      var g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      if (weightDecay != 0.0) {
        final wd = MLXArray.float_(ctx, weightDecay);
        g = g + wd * w;
        wd.dispose();
      }

      // v = α*v + (1-α)*g²
      final gSq = g * g;
      final prevV = _v[key];
      final v = prevV != null ? a * prevV + ac * gSq : ac * gSq;
      gSq.dispose();
      _v[key]?.dispose();
      _v[key] = v;

      // update = lr * g / (sqrt(v) + ε)
      final sqrtV = v.sqrt();
      final denom = sqrtV + eps;
      var update = lr * g / denom;
      sqrtV.dispose();
      denom.dispose();

      if (momentum != 0.0) {
        final m = MLXArray.float_(ctx, momentum);
        final prevBuf = _buf[key];
        final buf = prevBuf != null ? m * prevBuf + update : update;
        _buf[key]?.dispose();
        _buf[key] = buf;
        update = buf;
        m.dispose();
      }

      updated[key] = w - update;
    }

    a.dispose();
    ac.dispose();
    eps.dispose();
    lr.dispose();

    return updated;
  }
}

// ---------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------

/// AdaGrad optimizer (Duchi et al., 2011).
final class AdaGrad extends Optimizer {
  AdaGrad(
    super.ctx, {
    required this.learningRate,
    this.epsilon = 1e-8,
  });

  double learningRate;
  double epsilon;

  final Map<String, MLXArray> _g2 = {};

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    final updated = <String, MLXArray>{};
    final eps = MLXArray.float_(ctx, epsilon);
    final lr = MLXArray.float_(ctx, learningRate);

    for (final entry in gradients.entries) {
      final key = entry.key;
      final g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      final gSq = g * g;
      final prev = _g2[key];
      final g2 = prev != null ? prev + gSq : gSq;
      gSq.dispose();
      _g2[key]?.dispose();
      _g2[key] = g2;

      final sqrtG2 = g2.sqrt();
      final denom = sqrtG2 + eps;
      final step = lr * g / denom;
      sqrtG2.dispose();
      denom.dispose();
      updated[key] = w - step;
      step.dispose();
    }

    eps.dispose();
    lr.dispose();
    return updated;
  }
}

// ---------------------------------------------------------------------------
// AdaDelta
// ---------------------------------------------------------------------------

/// AdaDelta optimizer (Zeiler, 2012).
final class AdaDelta extends Optimizer {
  AdaDelta(
    super.ctx, {
    this.learningRate = 1.0,
    this.rho = 0.9,
    this.epsilon = 1e-8,
  });

  double learningRate;
  double rho;
  double epsilon;

  final Map<String, MLXArray> _eg2 = {};  // E[g²]
  final Map<String, MLXArray> _edx2 = {}; // E[Δx²]

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    final updated = <String, MLXArray>{};
    final r = MLXArray.float_(ctx, rho);
    final rc = MLXArray.float_(ctx, 1.0 - rho);
    final eps = MLXArray.float_(ctx, epsilon);
    final lr = MLXArray.float_(ctx, learningRate);

    for (final entry in gradients.entries) {
      final key = entry.key;
      final g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      final gSq = g * g;
      final prevEg2 = _eg2[key];
      final eg2 = prevEg2 != null ? r * prevEg2 + rc * gSq : rc * gSq;
      gSq.dispose();
      _eg2[key]?.dispose();
      _eg2[key] = eg2;

      final prevEdx2 = _edx2[key];
      final rmsEdx2 = prevEdx2 != null
          ? (prevEdx2 + eps).sqrt()
          : eps.sqrt();
      final rmsEg2 = (eg2 + eps).sqrt();
      final dx = rmsEdx2 / rmsEg2 * g;

      final edx2 = prevEdx2 != null
          ? r * prevEdx2 + rc * dx * dx
          : rc * dx * dx;
      rmsEdx2.dispose();
      rmsEg2.dispose();
      _edx2[key]?.dispose();
      _edx2[key] = edx2;

      final step = lr * dx;
      dx.dispose();
      updated[key] = w - step;
      step.dispose();
    }

    r.dispose();
    rc.dispose();
    eps.dispose();
    lr.dispose();
    return updated;
  }
}

// ---------------------------------------------------------------------------
// Lion
// ---------------------------------------------------------------------------

/// Lion optimizer (Chen et al., 2023) — sign-based update with momentum.
///
/// ```
/// update = sign(β1 * m + (1 - β1) * g)
/// w -= lr * (update + wd * w)
/// m = β2 * m + (1 - β2) * g
/// ```
final class Lion extends Optimizer {
  Lion(
    super.ctx, {
    required this.learningRate,
    this.beta1 = 0.9,
    this.beta2 = 0.99,
    this.weightDecay = 0.0,
  });

  double learningRate;
  double beta1;
  double beta2;
  double weightDecay;

  final Map<String, MLXArray> _m = {};

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    final updated = <String, MLXArray>{};
    final b1 = MLXArray.float_(ctx, beta1);
    final b1c = MLXArray.float_(ctx, 1.0 - beta1);
    final b2 = MLXArray.float_(ctx, beta2);
    final b2c = MLXArray.float_(ctx, 1.0 - beta2);
    final lr = MLXArray.float_(ctx, learningRate);

    for (final entry in gradients.entries) {
      final key = entry.key;
      final g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      final prevM = _m[key];
      // c = β1*m + (1-β1)*g
      final c = prevM != null ? b1 * prevM + b1c * g : b1c * g;
      final update = c.sign();
      c.dispose();

      // update = sign + wd * w
      var step = lr * update;
      if (weightDecay != 0.0) {
        final wd = MLXArray.float_(ctx, weightDecay);
        final wdStep = wd * w;
        final newStep = step + wdStep;
        step.dispose();
        wdStep.dispose();
        wd.dispose();
        step = newStep;
      }

      updated[key] = w - step;
      update.dispose();
      step.dispose();

      // m = β2*m + (1-β2)*g
      final newM = prevM != null ? b2 * prevM + b2c * g : b2c * g;
      _m[key]?.dispose();
      _m[key] = newM;
    }

    b1.dispose();
    b1c.dispose();
    b2.dispose();
    b2c.dispose();
    lr.dispose();
    return updated;
  }
}

// ---------------------------------------------------------------------------
// Adafactor
// ---------------------------------------------------------------------------

/// Adafactor optimizer (Shazeer & Stern, 2018).
///
/// Memory-efficient adaptive optimizer that factorises the second-moment
/// estimate into row and column factors for matrices, requiring O(n + m)
/// storage instead of O(n*m).
///
/// When [learningRate] is null the effective rate is computed from the
/// parameter scale: `lr = max(eps2, rms(w)) * relativeStep`.
final class Adafactor extends Optimizer {
  Adafactor(
    super.ctx, {
    this.learningRate,
    this.eps1 = 1e-30,
    this.eps2 = 1e-3,
    this.clippingThreshold = 1.0,
    this.decayRate = -0.8,
    this.relativeStep = true,
    this.scaleParameter = true,
    this.warmupInit = false,
  });

  final double? learningRate;
  final double eps1;
  final double eps2;
  final double clippingThreshold;
  final double decayRate;
  final bool relativeStep;
  final bool scaleParameter;
  final bool warmupInit;

  // State per parameter.
  final Map<String, MLXArray> _vr = {}; // row factor (rank-2+ params)
  final Map<String, MLXArray> _vc = {}; // column factor
  final Map<String, MLXArray> _v = {}; // full (rank-1 params)
  int _t = 0;

  @override
  Map<String, MLXArray> applyGradients(
    Map<String, MLXArray> gradients,
    Map<String, MLXArray> parameters,
  ) {
    _t++;
    final updated = <String, MLXArray>{};

    for (final entry in gradients.entries) {
      final key = entry.key;
      final g = entry.value;
      final w = parameters[key];
      if (w == null) continue;

      // Effective learning rate.
      final lr = _effectiveLr(w);

      // Second-moment decay factor ρ = 1 - t^decayRate.
      final rhoT = 1.0 - math.pow(_t, decayRate).toDouble();

      // Update second-moment estimate.
      final gSq = g * g; // g²
      final gSqEps = _addScalar(gSq, eps1);
      gSq.dispose();

      MLXArray vHat;
      if (w.ndim >= 2) {
        // Factored: maintain row (Vr) and column (Vc) factors.
        final prevVr = _vr[key];
        final prevVc = _vc[key];

        // Row factor: mean over last axis → shape [..., rows].
        final gSqRowMean = gSqEps.mean(axis: -1);
        final vr = prevVr != null
            ? _axpby(rhoT, prevVr, 1 - rhoT, gSqRowMean)
            : _scale(1 - rhoT, gSqRowMean);
        gSqRowMean.dispose();
        _vr[key]?.dispose();
        _vr[key] = vr;

        // Col factor: mean over second-to-last axis → shape [..., cols].
        final gSqColMean = gSqEps.mean(axis: -2);
        final vc = prevVc != null
            ? _axpby(rhoT, prevVc, 1 - rhoT, gSqColMean)
            : _scale(1 - rhoT, gSqColMean);
        gSqColMean.dispose();
        _vc[key]?.dispose();
        _vc[key] = vc;

        // Reconstruct: vr[..., :, None] * vc[..., None, :] / mean(vr, axis=-1)[..., None, None].
        final vrMean = vr.mean(axis: -1);
        final vrNorm = vr / vrMean.expandDims(-1);
        vrMean.dispose();
        vHat = vrNorm.expandDims(-1) * vc.expandDims(-2);
        vrNorm.dispose();
      } else {
        // Full second moment for rank-1 tensors.
        final prevV = _v[key];
        final v = prevV != null
            ? _axpby(rhoT, prevV, 1 - rhoT, gSqEps)
            : _scale(1 - rhoT, gSqEps);
        _v[key]?.dispose();
        _v[key] = v;
        vHat = v;
      }
      gSqEps.dispose();

      // Update = g / sqrt(vHat).
      final update = g / vHat.sqrt();
      if (!identical(vHat, _v[key])) vHat.dispose();

      // RMS clipping.
      final updateClipped = _rmsClip(update, clippingThreshold);
      update.dispose();

      // Scale by learning rate (and parameter scale if enabled).
      final effectiveLr = scaleParameter
          ? lr * math.max(eps2, _rmsFloat(w))
          : lr;

      final lrA = MLXArray.float_(ctx, effectiveLr);
      final newW = w - lrA * updateClipped;
      lrA.dispose();
      updateClipped.dispose();

      updated[key] = newW;
    }

    return updated;
  }

  double _effectiveLr(MLXArray w) {
    if (learningRate != null) return learningRate!;
    // Relative step: lr ∝ 1/sqrt(t).
    final minStep = warmupInit ? 1e-6 : 1e-2;
    return math.min(minStep, 1.0 / math.sqrt(_t));
  }

  MLXArray _addScalar(MLXArray a, double s) {
    final sA = MLXArray.float_(ctx, s);
    final result = a + sA;
    sA.dispose();
    return result;
  }

  MLXArray _scale(double s, MLXArray a) {
    final sA = MLXArray.float_(ctx, s);
    final result = sA * a;
    sA.dispose();
    return result;
  }

  MLXArray _axpby(double a, MLXArray x, double b, MLXArray y) {
    final aA = MLXArray.float_(ctx, a);
    final bA = MLXArray.float_(ctx, b);
    final ax = aA * x;
    final by = bA * y;
    final result = ax + by;
    aA.dispose();
    bA.dispose();
    ax.dispose();
    by.dispose();
    return result;
  }

  /// Root mean square of [a] as a scalar double.
  double _rmsFloat(MLXArray a) {
    final sq = a * a;
    final mean = sq.mean();
    final val = mean.itemFloat();
    sq.dispose();
    mean.dispose();
    return math.sqrt(val);
  }

  /// Clip [update] so its RMS ≤ [threshold].
  MLXArray _rmsClip(MLXArray update, double threshold) {
    final rms = _rmsFloat(update);
    if (rms <= threshold) return update;
    final scale = MLXArray.float_(ctx, threshold / rms);
    final result = update * scale;
    scale.dispose();
    return result;
  }
}
