// ignore_for_file: avoid_print

/// This example requires `libmlxc.dylib` on the library search path.
/// Build it from https://github.com/ml-explore/mlx-c before running.
///
/// Run with:
///   dart run example/example.dart
library;

import 'package:mlx_dart/mlx_dart.dart';

void main() {
  // Open a GPU (Metal) context — all arrays and ops run on Apple Silicon.
  final ctx = MLXContext.gpu();
  print('MLX version: ${ctx.version}');

  // ── Basic array construction ───────────────────────────────────────────────
  print('\n── Array construction ──');
  final zeros = MLXArray.zeros(ctx, [2, 3]);
  print('zeros [2,3]: $zeros');

  final ones = MLXArray.ones(ctx, [2, 3]);
  print('ones  [2,3]: $ones');

  final range = MLXArray.arange(ctx, 0.0, 6.0, 1.0).reshape([2, 3]);
  print('arange 0..5 reshaped [2,3]: $range');

  zeros.dispose();
  ones.dispose();

  // ── Arithmetic ────────────────────────────────────────────────────────────
  print('\n── Arithmetic ──');
  final a = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
  final b = MLXArray.fromFloats(ctx, [5.0, 6.0, 7.0, 8.0], shape: [2, 2]);

  final sum = a + b;
  print('a + b = $sum');

  final product = a * b;
  print('a * b = $product');

  final mm = a.matmul(b);
  print('a @ b (matmul) = $mm');

  sum.dispose();
  product.dispose();
  mm.dispose();

  // ── Reductions ────────────────────────────────────────────────────────────
  print('\n── Reductions ──');
  print('sum(a) = ${a.sum().itemFloat()}');
  print('mean(a) = ${a.mean().itemFloat()}');
  print('max(a) = ${a.max().itemFloat()}');

  // ── Shape manipulation ────────────────────────────────────────────────────
  print('\n── Shape ops ──');
  final flat = a.flatten();
  print('a.flatten() shape: ${flat.shape}');
  final t = a.T;
  print('a.T values: ${t.toFloat32List()}');
  flat.dispose();
  t.dispose();

  // ── Activations ───────────────────────────────────────────────────────────
  print('\n── Activations ──');
  final logits = MLXArray.fromFloats(ctx, [-2.0, -1.0, 0.0, 1.0, 2.0]);
  print('relu:    ${logits.relu().toFloat32List()}');
  print('sigmoid: ${logits.sigmoid().toFloat32List()}');
  print('softmax: ${logits.softmax().toFloat32List()}');
  logits.dispose();

  // ── Neural network layers ─────────────────────────────────────────────────
  print('\n── Layers ──');
  final linear = Linear(ctx, inFeatures: 4, outFeatures: 2);
  final x = MLXArray.fromFloats(ctx, [0.1, 0.2, 0.3, 0.4]);
  final y = linear.call(x);
  print('Linear(4→2) output shape: ${y.shape}');
  y.dispose();
  x.dispose();
  linear.dispose();

  final embed = Embedding(ctx, vocabSize: 100, dims: 8);
  final tokens = MLXArray.fromInts(ctx, [0, 5, 42]);
  final embedOut = embed.call(tokens);
  print('Embedding(100, 8) output shape: ${embedOut.shape}');
  embedOut.dispose();
  tokens.dispose();
  embed.dispose();

  // ── Concatenate / stack ───────────────────────────────────────────────────
  print('\n── Free functions ──');
  final c = concatenate(ctx, [a, b]);
  print('concatenate([a,b], axis=0) shape: ${c.shape}');
  c.dispose();

  final s = stack(ctx, [a, b]);
  print('stack([a,b], axis=0) shape: ${s.shape}');
  s.dispose();

  // ── Causal mask ───────────────────────────────────────────────────────────
  final mask = createCausalMask(ctx, n: 4, offset: 0);
  print('\ncausal mask [4×4]:\n$mask');
  mask.dispose();

  // ── Cleanup ───────────────────────────────────────────────────────────────
  range.dispose();
  a.dispose();
  b.dispose();
  ctx.dispose();
  print('\nDone.');
}
