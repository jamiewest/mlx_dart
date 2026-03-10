import 'array.dart';
import 'context.dart';

/// How to aggregate per-element losses into a scalar.
enum Reduction {
  /// Return per-element losses unchanged.
  none,

  /// Return the mean over all elements.
  mean,

  /// Return the sum over all elements.
  sum,
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

MLXArray _reduce(MLXContext ctx, MLXArray losses, Reduction reduction) {
  switch (reduction) {
    case Reduction.none:
      return losses;
    case Reduction.mean:
      final r = losses.mean();
      losses.dispose();
      return r;
    case Reduction.sum:
      final r = losses.sum();
      losses.dispose();
      return r;
  }
}

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

/// Cross-entropy loss for multi-class classification.
///
/// [logits] shape: `[B, C]`. [targets] may be integer class indices (`[B]`)
/// or soft probability distributions (`[B, C]`).
MLXArray crossEntropy(
  MLXContext ctx,
  MLXArray logits,
  MLXArray targets, {
  MLXArray? weights,
  int axis = -1,
  double labelSmoothing = 0.0,
  Reduction reduction = Reduction.mean,
}) {
  // log_softmax(logits) = logits - logSumExp(logits, keepdims=true)
  final lse = logits.logSumExp(axis: axis, keepdims: true);
  final logProbs = logits - lse;
  lse.dispose();

  MLXArray losses;
  if (targets.dtype == MLXDtype.int32 ||
      targets.dtype == MLXDtype.int64 ||
      targets.dtype == MLXDtype.uint32) {
    // Integer class indices — select the log-prob at each target index.
    final idx = targets.astype(MLXDtype.int32);
    final selected = logProbs.take(idx, axis: axis);
    idx.dispose();
    losses = selected.negative();
    selected.dispose();
  } else {
    // Soft targets (one-hot or arbitrary distributions).
    var t = targets;
    if (labelSmoothing > 0.0) {
      final numClasses = logits.shape.last;
      final s = MLXArray.float_(ctx, 1.0 - labelSmoothing);
      final u = MLXArray.float_(ctx, labelSmoothing / numClasses);
      final smoothed = targets * s + u;
      s.dispose();
      u.dispose();
      t = smoothed;
    }
    final prod = t * logProbs;
    if (!identical(t, targets)) t.dispose();
    final s = prod.sum(axis: axis);
    prod.dispose();
    losses = s.negative();
    s.dispose();
  }
  logProbs.dispose();

  if (weights != null) {
    final w = losses * weights;
    losses.dispose();
    losses = w;
  }
  return _reduce(ctx, losses, reduction);
}

/// Binary cross-entropy loss.
///
/// When [withLogits] is true (default), [inputs] are raw logits and the
/// numerically stable formulation is used. When false, [inputs] are
/// probabilities in `(0, 1)`.
MLXArray binaryCrossEntropy(
  MLXContext ctx,
  MLXArray inputs,
  MLXArray targets, {
  MLXArray? weights,
  bool withLogits = true,
  Reduction reduction = Reduction.mean,
}) {
  MLXArray losses;
  if (withLogits) {
    // max(x,0) - x*y + log(1 + exp(-|x|))
    final zero = MLXArray.zeros(ctx, [], dtype: inputs.dtype);
    final pos = inputs.maximum(zero);
    zero.dispose();
    final xy = inputs * targets;
    final negAbs = inputs.abs().negative();
    final logTerm = (negAbs.exp() + MLXArray.float_(ctx, 1.0)).log();
    negAbs.dispose();
    losses = pos - xy + logTerm;
    pos.dispose();
    xy.dispose();
    logTerm.dispose();
  } else {
    final eps = MLXArray.float_(ctx, 1e-7);
    final one = MLXArray.float_(ctx, 1.0);
    final oneMinusP = one - inputs;
    final oneMinusY = one - targets;
    one.dispose();
    final lp = inputs.maximum(eps);
    final lq = oneMinusP.maximum(eps);
    oneMinusP.dispose();
    eps.dispose();
    final t1 = targets * lp.log();
    final t2 = oneMinusY * lq.log();
    lp.dispose();
    lq.dispose();
    oneMinusY.dispose();
    losses = (t1 + t2).negative();
    t1.dispose();
    t2.dispose();
  }

  if (weights != null) {
    final w = losses * weights;
    losses.dispose();
    losses = w;
  }
  return _reduce(ctx, losses, reduction);
}

/// Mean absolute error (L1) loss.
MLXArray l1Loss(
  MLXContext ctx,
  MLXArray predictions,
  MLXArray targets, {
  Reduction reduction = Reduction.mean,
}) {
  final diff = (predictions - targets).abs();
  return _reduce(ctx, diff, reduction);
}

/// Mean squared error (L2) loss.
MLXArray mseLoss(
  MLXContext ctx,
  MLXArray predictions,
  MLXArray targets, {
  Reduction reduction = Reduction.mean,
}) {
  final diff = predictions - targets;
  final sq = diff.square();
  diff.dispose();
  return _reduce(ctx, sq, reduction);
}

/// Negative log-likelihood loss.
///
/// [inputs] should already be log-probabilities (e.g. output of log-softmax).
/// [targets] are integer class indices.
MLXArray nllLoss(
  MLXContext ctx,
  MLXArray inputs,
  MLXArray targets, {
  int axis = -1,
  Reduction reduction = Reduction.mean,
}) {
  final idx = targets.astype(MLXDtype.int32);
  final selected = inputs.take(idx, axis: axis);
  idx.dispose();
  final losses = selected.negative();
  selected.dispose();
  return _reduce(ctx, losses, reduction);
}

/// KL divergence loss: `KL(targets ‖ inputs)`.
///
/// [inputs] should be log-probabilities; [targets] should be probabilities.
MLXArray klDivLoss(
  MLXContext ctx,
  MLXArray inputs,
  MLXArray targets, {
  int axis = -1,
  Reduction reduction = Reduction.mean,
}) {
  final logT = targets.log();
  final diff = logT - inputs;
  logT.dispose();
  final prod = targets * diff;
  diff.dispose();
  final losses = prod.sum(axis: axis);
  prod.dispose();
  return _reduce(ctx, losses, reduction);
}

/// Smooth L1 (Huber with explicit [beta] threshold).
///
/// For `|diff| < beta`: `0.5 * diff² / beta`.
/// For `|diff| ≥ beta`: `|diff| - 0.5 * beta`.
MLXArray smoothL1Loss(
  MLXContext ctx,
  MLXArray predictions,
  MLXArray targets, {
  double beta = 1.0,
  Reduction reduction = Reduction.mean,
}) {
  final diff = (predictions - targets).abs();
  final betaA = MLXArray.float_(ctx, beta);
  final sq = diff.square() * MLXArray.float_(ctx, 0.5 / beta);
  final lin = diff - MLXArray.float_(ctx, 0.5 * beta);
  final cond = diff.less(betaA);
  final losses = where(ctx, cond, sq, lin);
  diff.dispose();
  betaA.dispose();
  sq.dispose();
  lin.dispose();
  cond.dispose();
  return _reduce(ctx, losses, reduction);
}

/// Huber loss.
///
/// For `|diff| ≤ delta`: `0.5 * diff²`.
/// For `|diff| > delta`: `delta * (|diff| - 0.5 * delta)`.
MLXArray huberLoss(
  MLXContext ctx,
  MLXArray predictions,
  MLXArray targets, {
  double delta = 1.0,
  Reduction reduction = Reduction.mean,
}) {
  final diff = (predictions - targets).abs();
  final dA = MLXArray.float_(ctx, delta);
  final sq = diff.square() * MLXArray.float_(ctx, 0.5);
  final lin = dA * (diff - MLXArray.float_(ctx, 0.5 * delta));
  final cond = diff.lessEqual(dA);
  final losses = where(ctx, cond, sq, lin);
  diff.dispose();
  dA.dispose();
  sq.dispose();
  lin.dispose();
  cond.dispose();
  return _reduce(ctx, losses, reduction);
}

/// Hinge loss: `max(0, 1 - targets * inputs)`.
MLXArray hingeLoss(
  MLXContext ctx,
  MLXArray inputs,
  MLXArray targets, {
  Reduction reduction = Reduction.mean,
}) {
  final zero = MLXArray.zeros(ctx, [], dtype: inputs.dtype);
  final one = MLXArray.float_(ctx, 1.0);
  final margin = one - targets * inputs;
  one.dispose();
  final losses = margin.maximum(zero);
  margin.dispose();
  zero.dispose();
  return _reduce(ctx, losses, reduction);
}

/// Triplet margin loss.
MLXArray tripletLoss(
  MLXContext ctx,
  MLXArray anchors,
  MLXArray positives,
  MLXArray negatives, {
  int axis = -1,
  double p = 2.0,
  double margin = 1.0,
  double eps = 1e-6,
  Reduction reduction = Reduction.mean,
}) {
  final dPos = _lpDist(ctx, anchors, positives, p, eps, axis);
  final dNeg = _lpDist(ctx, anchors, negatives, p, eps, axis);
  final mA = MLXArray.float_(ctx, margin);
  final zero = MLXArray.zeros(ctx, [], dtype: anchors.dtype);
  final raw = dPos - dNeg + mA;
  dPos.dispose();
  dNeg.dispose();
  mA.dispose();
  final losses = raw.maximum(zero);
  raw.dispose();
  zero.dispose();
  return _reduce(ctx, losses, reduction);
}

MLXArray _lpDist(
    MLXContext ctx, MLXArray a, MLXArray b, double p, double eps, int axis) {
  final diff = (a - b).abs();
  if (p == 2.0) {
    final sq = diff.square();
    diff.dispose();
    final s = sq.sum(axis: axis);
    sq.dispose();
    final e = MLXArray.float_(ctx, eps);
    final safe = s + e;
    s.dispose();
    e.dispose();
    final r = safe.sqrt();
    safe.dispose();
    return r;
  } else {
    final pA = MLXArray.float_(ctx, p);
    final pw = diff.power(pA);
    diff.dispose();
    pA.dispose();
    final s = pw.sum(axis: axis);
    pw.dispose();
    final invP = MLXArray.float_(ctx, 1.0 / p);
    final r = s.power(invP);
    s.dispose();
    invP.dispose();
    return r;
  }
}

/// Log-cosh loss: `log(cosh(predictions - targets))`.
MLXArray logCoshLoss(
  MLXContext ctx,
  MLXArray inputs,
  MLXArray targets, {
  Reduction reduction = Reduction.mean,
}) {
  final diff = inputs - targets;
  final c = diff.cosh();
  diff.dispose();
  final losses = c.log();
  c.dispose();
  return _reduce(ctx, losses, reduction);
}

/// Cosine similarity loss: `1 - cosine_similarity(x1, x2, axis)`.
MLXArray cosineSimilarityLoss(
  MLXContext ctx,
  MLXArray x1,
  MLXArray x2, {
  int axis = -1,
  double eps = 1e-8,
  Reduction reduction = Reduction.mean,
}) {
  final dot = (x1 * x2).sum(axis: axis);
  final n1 = x1.square().sum(axis: axis).sqrt();
  final n2 = x2.square().sum(axis: axis).sqrt();
  final epsA = MLXArray.float_(ctx, eps);
  final denom = (n1 * n2).maximum(epsA);
  n1.dispose();
  n2.dispose();
  epsA.dispose();
  final sim = dot / denom;
  dot.dispose();
  denom.dispose();
  final one = MLXArray.float_(ctx, 1.0);
  final losses = one - sim;
  one.dispose();
  sim.dispose();
  return _reduce(ctx, losses, reduction);
}
