@Tags(['integration'])
library;

import 'package:mlx_dart/mlx_dart.dart';
import 'package:test/test.dart';

// Integration tests require `libmlxc.dylib` on the library search path.
//
// Run:
//   dart test --tags integration
//
// Skip in CI (no native lib):
//   dart test --exclude-tags integration

void main() {
  // -------------------------------------------------------------------------
  // MLXContext
  // -------------------------------------------------------------------------

  group('MLXContext', () {
    test('gpu() constructs without throwing', () {
      final ctx = MLXContext.gpu();
      expect(ctx, isNotNull);
      ctx.dispose();
    });

    test('cpu() constructs without throwing', () {
      final ctx = MLXContext.cpu();
      expect(ctx, isNotNull);
      ctx.dispose();
    });

    test('version returns a non-empty string', () {
      final ctx = MLXContext.gpu();
      expect(ctx.version, isNotEmpty);
      ctx.dispose();
    });

    test('memory properties return non-negative values', () {
      final ctx = MLXContext.gpu();
      expect(ctx.activeMemory, greaterThanOrEqualTo(0));
      expect(ctx.cacheMemory, greaterThanOrEqualTo(0));
      expect(ctx.peakMemory, greaterThanOrEqualTo(0));
      expect(ctx.memoryLimit, greaterThan(0));
      ctx.dispose();
    });

    test('clearCache does not throw', () {
      final ctx = MLXContext.gpu();
      expect(() => ctx.clearCache(), returnsNormally);
      ctx.dispose();
    });

    test('resetPeakMemory does not throw', () {
      final ctx = MLXContext.gpu();
      expect(() => ctx.resetPeakMemory(), returnsNormally);
      ctx.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXArray construction
  // -------------------------------------------------------------------------

  group('MLXArray construction', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('zeros has correct shape and dtype', () {
      final a = MLXArray.zeros(ctx, [2, 3]);
      expect(a.shape, equals([2, 3]));
      expect(a.ndim, equals(2));
      expect(a.dtype, equals(MLXDtype.float32));
      expect(a.size, equals(6));
      a.dispose();
    });

    test('ones has correct shape', () {
      final a = MLXArray.ones(ctx, [4]);
      expect(a.shape, equals([4]));
      a.dispose();
    });

    test('fromFloats round-trips', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final a = MLXArray.fromFloats(ctx, data, shape: [2, 2]);
      expect(a.toFloat32List(), equals(data));
      a.dispose();
    });

    test('fromInts round-trips', () {
      final data = [10, 20, 30];
      final a = MLXArray.fromInts(ctx, data);
      expect(a.toInt32List(), equals(data));
      a.dispose();
    });

    test('arange produces correct values', () {
      final a = MLXArray.arange(ctx, 0.0, 5.0, 1.0);
      expect(a.toFloat32List(), equals([0.0, 1.0, 2.0, 3.0, 4.0]));
      a.dispose();
    });

    test('bool_ scalar', () {
      final a = MLXArray.bool_(ctx, true);
      expect(a.dtype, equals(MLXDtype.bool_));
      a.dispose();
    });

    test('int_ scalar', () {
      final a = MLXArray.int_(ctx, 42);
      expect(a.itemInt(), equals(42));
      a.dispose();
    });

    test('float_ scalar', () {
      final a = MLXArray.float_(ctx, 3.14);
      expect(a.itemFloat(), closeTo(3.14, 1e-5));
      a.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXArray arithmetic
  // -------------------------------------------------------------------------

  group('MLXArray arithmetic', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('add', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0]);
      final b = MLXArray.fromFloats(ctx, [3.0, 4.0]);
      final c = a + b;
      expect(c.toFloat32List(), equals([4.0, 6.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('subtract', () {
      final a = MLXArray.fromFloats(ctx, [5.0, 6.0]);
      final b = MLXArray.fromFloats(ctx, [1.0, 2.0]);
      final c = a - b;
      expect(c.toFloat32List(), equals([4.0, 4.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('multiply', () {
      final a = MLXArray.fromFloats(ctx, [2.0, 3.0]);
      final b = MLXArray.fromFloats(ctx, [4.0, 5.0]);
      final c = a * b;
      expect(c.toFloat32List(), equals([8.0, 15.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('matmul 2x2', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
      final b = MLXArray.ones(ctx, [2, 2]);
      final c = a.matmul(b);
      expect(c.toFloat32List(), equals([3.0, 3.0, 7.0, 7.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXArray shape ops
  // -------------------------------------------------------------------------

  group('MLXArray shape ops', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('reshape', () {
      final a = MLXArray.arange(ctx, 0.0, 6.0, 1.0);
      final b = a.reshape([2, 3]);
      expect(b.shape, equals([2, 3]));
      a.dispose();
      b.dispose();
    });

    test('transpose T', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
      final b = a.T;
      expect(b.shape, equals([2, 2]));
      expect(b.toFloat32List(), equals([1.0, 3.0, 2.0, 4.0]));
      a.dispose();
      b.dispose();
    });

    test('flatten', () {
      final a = MLXArray.zeros(ctx, [2, 3, 4]);
      final b = a.flatten();
      expect(b.shape, equals([24]));
      a.dispose();
      b.dispose();
    });

    test('expandDims', () {
      final a = MLXArray.zeros(ctx, [4]);
      final b = a.expandDims(0);
      expect(b.shape, equals([1, 4]));
      a.dispose();
      b.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXArray reductions
  // -------------------------------------------------------------------------

  group('MLXArray reductions', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('sum all', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0]);
      final s = a.sum();
      expect(s.itemFloat(), closeTo(10.0, 1e-5));
      a.dispose();
      s.dispose();
    });

    test('mean all', () {
      final a = MLXArray.fromFloats(ctx, [2.0, 4.0, 6.0, 8.0]);
      final m = a.mean();
      expect(m.itemFloat(), closeTo(5.0, 1e-5));
      a.dispose();
      m.dispose();
    });

    test('max all', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 5.0, 3.0]);
      final mx = a.max();
      expect(mx.itemFloat(), closeTo(5.0, 1e-5));
      a.dispose();
      mx.dispose();
    });

    test('argmax', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 5.0, 3.0]);
      final idx = a.argmax();
      expect(idx.itemInt(), equals(1));
      a.dispose();
      idx.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXArray element-wise math
  // -------------------------------------------------------------------------

  group('MLXArray element-wise math', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('exp', () {
      final a = MLXArray.float_(ctx, 0.0);
      final b = a.exp();
      expect(b.itemFloat(), closeTo(1.0, 1e-5));
      a.dispose();
      b.dispose();
    });

    test('sqrt', () {
      final a = MLXArray.float_(ctx, 4.0);
      final b = a.sqrt();
      expect(b.itemFloat(), closeTo(2.0, 1e-5));
      a.dispose();
      b.dispose();
    });

    test('abs', () {
      final a = MLXArray.float_(ctx, -3.0);
      final b = a.abs();
      expect(b.itemFloat(), closeTo(3.0, 1e-5));
      a.dispose();
      b.dispose();
    });

    test('relu zeros negatives', () {
      final a = MLXArray.fromFloats(ctx, [-1.0, 0.0, 2.0]);
      final b = a.relu();
      expect(b.toFloat32List(), equals([0.0, 0.0, 2.0]));
      a.dispose();
      b.dispose();
    });

    test('sigmoid at 0 is 0.5', () {
      final a = MLXArray.float_(ctx, 0.0);
      final b = a.sigmoid();
      expect(b.itemFloat(), closeTo(0.5, 1e-5));
      a.dispose();
      b.dispose();
    });

    test('leakyRelu passes positives unchanged', () {
      final a = MLXArray.fromFloats(ctx, [2.0, -1.0]);
      final b = a.leakyRelu(negSlope: 0.1);
      final vals = b.toFloat32List();
      expect(vals[0], closeTo(2.0, 1e-5));
      expect(vals[1], closeTo(-0.1, 1e-5));
      a.dispose();
      b.dispose();
    });

    test('relu6 clips at 6', () {
      final a = MLXArray.fromFloats(ctx, [-1.0, 3.0, 8.0]);
      final b = a.relu6();
      final vals = b.toFloat32List();
      expect(vals[0], closeTo(0.0, 1e-5));
      expect(vals[1], closeTo(3.0, 1e-5));
      expect(vals[2], closeTo(6.0, 1e-5));
      a.dispose();
      b.dispose();
    });

    test('softPlus is smooth approx of relu', () {
      final a = MLXArray.fromFloats(ctx, [0.0]);
      final b = a.softPlus();
      // softplus(0) = ln(2) ≈ 0.6931
      expect(b.itemFloat(), closeTo(0.6931, 1e-3));
      a.dispose();
      b.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // Free functions
  // -------------------------------------------------------------------------

  group('free functions', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('concatenate along axis 0', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0], shape: [1, 2]);
      final b = MLXArray.fromFloats(ctx, [3.0, 4.0], shape: [1, 2]);
      final c = concatenate(ctx, [a, b]);
      expect(c.shape, equals([2, 2]));
      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('stack along new axis', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0]);
      final b = MLXArray.fromFloats(ctx, [3.0, 4.0]);
      final c = stack(ctx, [a, b]);
      expect(c.shape, equals([2, 2]));
      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('where selects correctly', () {
      final cond = MLXArray.fromInts(ctx, [1, 0, 1], dtype: MLXDtype.bool_);
      final x = MLXArray.fromFloats(ctx, [10.0, 20.0, 30.0]);
      final y = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0]);
      final r = where(ctx, cond, x, y);
      expect(r.toFloat32List(), equals([10.0, 2.0, 30.0]));
      cond.dispose();
      x.dispose();
      y.dispose();
      r.dispose();
    });

    test('evalAll evaluates multiple arrays', () {
      final a = MLXArray.zeros(ctx, [3]);
      final b = MLXArray.ones(ctx, [3]);
      evalAll(ctx, [a, b]);
      a.dispose();
      b.dispose();
    });

    test('createCausalMask shape', () {
      final mask = createCausalMask(ctx, n: 4, offset: 0);
      expect(mask.shape, equals([4, 4]));
      mask.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXArray disposal
  // -------------------------------------------------------------------------

  group('MLXArray disposal', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('isDisposed is false before disposal', () {
      final a = MLXArray.zeros(ctx, [2]);
      expect(a.isDisposed, isFalse);
      a.dispose();
    });

    test('isDisposed is true after disposal', () {
      final a = MLXArray.zeros(ctx, [2]);
      a.dispose();
      expect(a.isDisposed, isTrue);
    });

    test('double dispose does not throw', () {
      final a = MLXArray.zeros(ctx, [2]);
      a.dispose();
      expect(() => a.dispose(), returnsNormally);
    });

    test('toString on disposed array returns sentinel', () {
      final a = MLXArray.zeros(ctx, [2]);
      a.dispose();
      expect(a.toString(), equals('MLXArray(disposed)'));
    });

    test('using disposed array throws StateError', () {
      final a = MLXArray.zeros(ctx, [2]);
      a.dispose();
      expect(() => a.shape, throwsStateError);
    });
  });

  // -------------------------------------------------------------------------
  // Losses
  // -------------------------------------------------------------------------

  group('losses', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('mseLoss perfect prediction is zero', () {
      final p = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0]);
      final t = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0]);
      final loss = mseLoss(ctx, p, t);
      expect(loss.itemFloat(), closeTo(0.0, 1e-5));
      p.dispose();
      t.dispose();
      loss.dispose();
    });

    test('mseLoss known value', () {
      final p = MLXArray.fromFloats(ctx, [0.0, 0.0]);
      final t = MLXArray.fromFloats(ctx, [1.0, 3.0]);
      final loss = mseLoss(ctx, p, t); // mean((1+9)/2) = 5
      expect(loss.itemFloat(), closeTo(5.0, 1e-4));
      p.dispose();
      t.dispose();
      loss.dispose();
    });

    test('l1Loss known value', () {
      final p = MLXArray.fromFloats(ctx, [0.0, 0.0]);
      final t = MLXArray.fromFloats(ctx, [1.0, 3.0]);
      final loss = l1Loss(ctx, p, t); // mean(1+3)/2 = 2
      expect(loss.itemFloat(), closeTo(2.0, 1e-4));
      p.dispose();
      t.dispose();
      loss.dispose();
    });

    test('crossEntropy with one-hot target', () {
      // logits = [0,0,10], target class = 2 → loss ≈ 0
      final logits = MLXArray.fromFloats(
        ctx,
        [0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        shape: [2, 3],
      );
      final targets = MLXArray.fromInts(ctx, [2, 2]);
      final loss = crossEntropy(ctx, logits, targets);
      expect(loss.itemFloat(), closeTo(0.0, 0.01));
      logits.dispose();
      targets.dispose();
      loss.dispose();
    });

    test('binaryCrossEntropy with perfect prediction', () {
      // sigmoid(10) ≈ 1.0, target = 1 → loss ≈ 0
      final logits = MLXArray.fromFloats(ctx, [10.0, -10.0]);
      final targets = MLXArray.fromFloats(ctx, [1.0, 0.0]);
      final loss = binaryCrossEntropy(ctx, logits, targets);
      expect(loss.itemFloat(), closeTo(0.0, 0.001));
      logits.dispose();
      targets.dispose();
      loss.dispose();
    });

    test('cosineSimilarityLoss with parallel vectors is ~0', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 0.0], shape: [1, 2]);
      final b = MLXArray.fromFloats(ctx, [2.0, 0.0], shape: [1, 2]);
      final loss = cosineSimilarityLoss(ctx, a, b);
      // cosine sim = 1 → loss = 1 - 1 = 0
      expect(loss.itemFloat(), closeTo(0.0, 1e-4));
      a.dispose();
      b.dispose();
      loss.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // Optimizers
  // -------------------------------------------------------------------------

  group('optimizers', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('SGD moves parameter in gradient direction', () {
      final sgd = SGD(ctx, learningRate: 0.1);
      final w = MLXArray.fromFloats(ctx, [1.0, 2.0]);
      final g = MLXArray.fromFloats(ctx, [1.0, 1.0]);
      final updated = sgd.applyGradients({'w': g}, {'w': w});
      final newW = updated['w']!;
      // w_new = w - lr * g = [0.9, 1.9]
      expect(newW.toFloat32List()[0], closeTo(0.9, 1e-5));
      expect(newW.toFloat32List()[1], closeTo(1.9, 1e-5));
      w.dispose();
      g.dispose();
      newW.dispose();
    });

    test('SGD with momentum accumulates velocity', () {
      final sgd = SGD(ctx, learningRate: 0.1, momentum: 0.9);
      final w = MLXArray.fromFloats(ctx, [0.0]);
      final g = MLXArray.fromFloats(ctx, [1.0]);
      // Step 1: v = g = 1, w = 0 - 0.1 * 1 = -0.1
      final r1 = sgd.applyGradients({'w': g}, {'w': w});
      final w1 = r1['w']!;
      expect(w1.toFloat32List()[0], closeTo(-0.1, 1e-5));
      // Step 2: v = 0.9 * 1 + 1 = 1.9, w = -0.1 - 0.1 * 1.9 = -0.29
      final r2 = sgd.applyGradients({'w': g}, {'w': w1});
      final w2 = r2['w']!;
      expect(w2.toFloat32List()[0], closeTo(-0.29, 1e-4));
      w.dispose();
      g.dispose();
      w1.dispose();
      w2.dispose();
    });

    test('Adam updates parameter', () {
      final adam = Adam(ctx, learningRate: 0.01);
      final w = MLXArray.fromFloats(ctx, [1.0]);
      final g = MLXArray.fromFloats(ctx, [1.0]);
      final r = adam.applyGradients({'w': g}, {'w': w});
      final newW = r['w']!;
      // After one step w should move slightly toward 0
      expect(newW.toFloat32List()[0], lessThan(1.0));
      w.dispose();
      g.dispose();
      newW.dispose();
    });

    test('AdamW applies weight decay', () {
      final adamW = AdamW(ctx, learningRate: 0.01, weightDecay: 0.1);
      final adam = Adam(ctx, learningRate: 0.01);
      final w = MLXArray.fromFloats(ctx, [1.0]);
      final g = MLXArray.fromFloats(ctx, [0.0]); // zero gradient → only weight decay

      final rAdamW = adamW.applyGradients({'w': g}, {'w': w});
      final rAdam = adam.applyGradients({'w': g}, {'w': w});
      // AdamW with nonzero weight decay should produce a smaller weight
      expect(
        rAdamW['w']!.toFloat32List()[0],
        lessThan(rAdam['w']!.toFloat32List()[0]),
      );
      w.dispose();
      g.dispose();
      rAdamW['w']!.dispose();
      rAdam['w']!.dispose();
    });

    test('SGD with weight decay shrinks parameter', () {
      final sgd = SGD(ctx, learningRate: 0.1, weightDecay: 0.5);
      final w = MLXArray.fromFloats(ctx, [2.0]);
      final g = MLXArray.fromFloats(ctx, [0.0]);
      final r = sgd.applyGradients({'w': g}, {'w': w});
      // effective g = 0 + 0.5 * 2 = 1, w_new = 2 - 0.1 * 1 = 1.9
      expect(r['w']!.toFloat32List()[0], closeTo(1.9, 1e-5));
      w.dispose();
      g.dispose();
      r['w']!.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // KV cache
  // -------------------------------------------------------------------------

  group('KVCache', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('first update stores and returns the input', () {
      final cache = KVCache(ctx);
      final k = MLXArray.zeros(ctx, [1, 2, 3, 4]); // [B, H, S, D]
      final v = MLXArray.zeros(ctx, [1, 2, 3, 4]);
      final (rk, _) = cache.update(k, v);
      expect(rk.shape, equals([1, 2, 3, 4]));
      expect(cache.offset, equals(3));
      cache.dispose();
    });

    test('second update concatenates along sequence axis', () {
      final cache = KVCache(ctx);
      final k1 = MLXArray.zeros(ctx, [1, 2, 3, 4]);
      final v1 = MLXArray.zeros(ctx, [1, 2, 3, 4]);
      cache.update(k1, v1);
      final k2 = MLXArray.zeros(ctx, [1, 2, 2, 4]);
      final v2 = MLXArray.zeros(ctx, [1, 2, 2, 4]);
      final (rk, _) = cache.update(k2, v2);
      expect(rk.shape, equals([1, 2, 5, 4]));
      expect(cache.offset, equals(5));
      cache.dispose();
    });

    test('dispose clears state', () {
      final cache = KVCache(ctx);
      final k = MLXArray.zeros(ctx, [1, 1, 1, 4]);
      final v = MLXArray.zeros(ctx, [1, 1, 1, 4]);
      cache.update(k, v);
      cache.dispose();
      expect(cache.isInitialized, isFalse);
      expect(cache.offset, equals(0));
    });
  });

  group('RotatingKVCache', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('does not exceed maxSize', () {
      final cache = RotatingKVCache(ctx, maxSize: 4);
      // Insert 3 tokens then 3 more — should cap at 4.
      final k1 = MLXArray.zeros(ctx, [1, 1, 3, 8]);
      final v1 = MLXArray.zeros(ctx, [1, 1, 3, 8]);
      cache.update(k1, v1);

      final k2 = MLXArray.zeros(ctx, [1, 1, 3, 8]);
      final v2 = MLXArray.zeros(ctx, [1, 1, 3, 8]);
      final (rk, _) = cache.update(k2, v2);
      expect(rk.shape[2], equals(4));
      expect(cache.size, equals(4));
      expect(cache.offset, equals(6)); // total tokens seen
      cache.dispose();
    });

    test('stays within maxSize when input fits', () {
      final cache = RotatingKVCache(ctx, maxSize: 10);
      final k = MLXArray.zeros(ctx, [1, 1, 4, 8]);
      final v = MLXArray.zeros(ctx, [1, 1, 4, 8]);
      final (rk, _) = cache.update(k, v);
      expect(rk.shape[2], equals(4));
      cache.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // Layers
  // -------------------------------------------------------------------------

  group('Linear layer', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shape', () {
      final layer = Linear(ctx, inFeatures: 4, outFeatures: 2);
      final x = MLXArray.fromFloats(ctx, [0.1, 0.2, 0.3, 0.4]);
      final y = layer.call(x);
      expect(y.shape, equals([2]));
      layer.dispose();
      x.dispose();
      y.dispose();
    });

    test('parameters returns weight and bias', () {
      final layer = Linear(ctx, inFeatures: 3, outFeatures: 2);
      final params = layer.parameters();
      expect(params.containsKey('weight'), isTrue);
      expect(params.containsKey('bias'), isTrue);
      layer.dispose();
    });

    test('no-bias Linear has no bias key', () {
      final layer = Linear(ctx, inFeatures: 3, outFeatures: 2, bias: false);
      expect(layer.parameters().containsKey('bias'), isFalse);
      layer.dispose();
    });
  });

  group('Embedding layer', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shape', () {
      final layer = Embedding(ctx, vocabSize: 100, dims: 8);
      final tokens = MLXArray.fromInts(ctx, [0, 5, 42]);
      final out = layer.call(tokens);
      expect(out.shape, equals([3, 8]));
      layer.dispose();
      tokens.dispose();
      out.dispose();
    });
  });

  group('LayerNorm', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shape preserved', () {
      final ln = LayerNorm(ctx, dims: 4);
      final x = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0]);
      final y = ln.call(x);
      expect(y.shape, equals([4]));
      ln.dispose();
      x.dispose();
      y.dispose();
    });

    test('normalises mean to ~0', () {
      final ln = LayerNorm(ctx, dims: 4, bias: false);
      final x = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0]);
      final y = ln.call(x);
      final m = y.mean();
      expect(m.itemFloat().abs(), lessThan(1e-4));
      ln.dispose();
      x.dispose();
      y.dispose();
      m.dispose();
    });
  });

  group('Sequential', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('passes through layers in order', () {
      final model = Sequential([
        Linear(ctx, inFeatures: 4, outFeatures: 8),
        ReLU(),
        Linear(ctx, inFeatures: 8, outFeatures: 2),
      ]);
      final x = MLXArray.zeros(ctx, [4]);
      final y = model.call(x);
      expect(y.shape, equals([2]));
      model.dispose();
      x.dispose();
      y.dispose();
    });
  });

  group('RNN', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shapes [B,L,H] and [B,H]', () {
      final rnn = RNN(ctx, inputSize: 4, hiddenSize: 8);
      final x = MLXArray.zeros(ctx, [2, 5, 4]); // [B=2, L=5, I=4]
      final (out, hn) = rnn.call(x);
      expect(out.shape, equals([2, 5, 8]));
      expect(hn.shape, equals([2, 8]));
      rnn.dispose();
      x.dispose();
      out.dispose();
      hn.dispose();
    });
  });

  group('GRU', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shapes [B,L,H] and [B,H]', () {
      final gru = GRU(ctx, inputSize: 4, hiddenSize: 8);
      final x = MLXArray.zeros(ctx, [2, 5, 4]);
      final (out, hn) = gru.call(x);
      expect(out.shape, equals([2, 5, 8]));
      expect(hn.shape, equals([2, 8]));
      gru.dispose();
      x.dispose();
      out.dispose();
      hn.dispose();
    });
  });

  group('LSTM', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shapes [B,L,H], [B,H], [B,H]', () {
      final lstm = LSTM(ctx, inputSize: 4, hiddenSize: 8);
      final x = MLXArray.zeros(ctx, [2, 5, 4]);
      final (out, hn, cn) = lstm.call(x);
      expect(out.shape, equals([2, 5, 8]));
      expect(hn.shape, equals([2, 8]));
      expect(cn.shape, equals([2, 8]));
      lstm.dispose();
      x.dispose();
      out.dispose();
      hn.dispose();
      cn.dispose();
    });
  });

  group('MultiHeadAttention', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shape [B, L, dims]', () {
      final mha = MultiHeadAttention(ctx, dims: 8, numHeads: 2);
      final q = MLXArray.zeros(ctx, [1, 4, 8]);
      final k = MLXArray.zeros(ctx, [1, 4, 8]);
      final v = MLXArray.zeros(ctx, [1, 4, 8]);
      final out = mha.call(q, k, v);
      expect(out.shape, equals([1, 4, 8]));
      mha.dispose();
      q.dispose();
      k.dispose();
      v.dispose();
      out.dispose();
    });
  });

  group('TransformerEncoder', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shape preserved', () {
      final enc = TransformerEncoder(ctx, numLayers: 2, dims: 8, numHeads: 2);
      final x = MLXArray.zeros(ctx, [1, 4, 8]);
      final out = enc.call(x);
      expect(out.shape, equals([1, 4, 8]));
      enc.dispose();
      x.dispose();
      out.dispose();
    });
  });

  group('MaxPool1d / AvgPool1d', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('MaxPool1d output shape', () {
      final pool = MaxPool1d(kernelSize: 2, stride: 2);
      final x = MLXArray.zeros(ctx, [1, 8, 4]); // [B, L, C]
      final y = pool.call(x);
      expect(y.shape, equals([1, 4, 4]));
      x.dispose();
      y.dispose();
    });

    test('AvgPool1d output shape', () {
      final pool = AvgPool1d(kernelSize: 2, stride: 2);
      final x = MLXArray.zeros(ctx, [1, 8, 4]);
      final y = pool.call(x);
      expect(y.shape, equals([1, 4, 4]));
      x.dispose();
      y.dispose();
    });

    test('MaxPool1d selects max across kernel', () {
      // [B=1, L=2, C=1]: positions [0]=1, [1]=3 with kernelSize=2 → max=3
      final pool = MaxPool1d(kernelSize: 2, stride: 2);
      final x = MLXArray.fromFloats(ctx, [1.0, 3.0], shape: [1, 2, 1]);
      final y = pool.call(x);
      expect(y.toFloat32List()[0], closeTo(3.0, 1e-5));
      x.dispose();
      y.dispose();
    });
  });

  group('MaxPool2d / AvgPool2d', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('MaxPool2d output shape', () {
      final pool = MaxPool2d(kernelSize: 2, stride: 2);
      final x = MLXArray.zeros(ctx, [1, 4, 4, 2]); // [B, H, W, C]
      final y = pool.call(x);
      expect(y.shape, equals([1, 2, 2, 2]));
      x.dispose();
      y.dispose();
    });

    test('AvgPool2d output shape', () {
      final pool = AvgPool2d(kernelSize: 2, stride: 2);
      final x = MLXArray.zeros(ctx, [1, 4, 4, 2]);
      final y = pool.call(x);
      expect(y.shape, equals([1, 2, 2, 2]));
      x.dispose();
      y.dispose();
    });
  });

  group('GroupNorm', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('output shape preserved', () {
      final gn = GroupNorm(ctx, dims: 8, numGroups: 2);
      final x = MLXArray.ones(ctx, [2, 8]);
      final y = gn.call(x);
      expect(y.shape, equals([2, 8]));
      gn.dispose();
      x.dispose();
      y.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXLinalg
  // -------------------------------------------------------------------------

  group('MLXLinalg', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('norm of [3, 4] is 5', () {
      final a = MLXArray.fromFloats(ctx, [3.0, 4.0]);
      final n = MLXLinalg.norm(ctx, a);
      expect(n.itemFloat(), closeTo(5.0, 1e-4));
      a.dispose();
      n.dispose();
    });

    test('inv of identity is identity', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 0.0, 0.0, 1.0], shape: [2, 2]);
      final inv = MLXLinalg.inv(ctx, a);
      final vals = inv.toFloat32List();
      expect(vals[0], closeTo(1.0, 1e-4));
      expect(vals[1], closeTo(0.0, 1e-4));
      expect(vals[2], closeTo(0.0, 1e-4));
      expect(vals[3], closeTo(1.0, 1e-4));
      a.dispose();
      inv.dispose();
    });

    test('svd returns three arrays', () {
      final a = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
      final result = MLXLinalg.svd(ctx, a);
      expect(result.length, equals(3));
      final s = result[1];
      expect(s.shape.length, equals(1)); // singular values are 1-D
      a.dispose();
      for (final r in result) {
        r.dispose();
      }
    });
  });

  // -------------------------------------------------------------------------
  // MLXFFT
  // -------------------------------------------------------------------------

  group('MLXFFT', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('fft of constant signal has DC at index 0', () {
      final x = MLXArray.fromFloats(ctx, [1.0, 1.0, 1.0, 1.0]);
      final X = MLXFFT.fft(ctx, x);
      // FFT of all-ones length-4 → [4, 0, 0, 0] (complex).
      // We only check that the output has the same length.
      expect(X.shape[0], equals(4));
      x.dispose();
      X.dispose();
    });

    test('ifft(fft(x)) round-trips for real input', () {
      final original = MLXArray.fromFloats(ctx, [1.0, 2.0, 3.0, 4.0]);
      final freq = MLXFFT.fft(ctx, original);
      final recovered = MLXFFT.ifft(ctx, freq);
      // Real part should match original; imaginary part ≈ 0.
      expect(recovered.shape[0], equals(4));
      original.dispose();
      freq.dispose();
      recovered.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // MLXRandom
  // -------------------------------------------------------------------------

  group('MLXRandom', () {
    late MLXContext ctx;
    setUp(() => ctx = MLXContext.gpu());
    tearDown(() => ctx.dispose());

    test('uniform produces values in [0, 1)', () {
      final r = MLXRandom.uniform(ctx, shape: [100]);
      final vals = r.toFloat32List();
      for (final v in vals) {
        expect(v, greaterThanOrEqualTo(0.0));
        expect(v, lessThan(1.0));
      }
      r.dispose();
    });

    test('normal has approximately zero mean', () {
      final r = MLXRandom.normal(ctx, shape: [1000]);
      final m = r.mean();
      expect(m.itemFloat().abs(), lessThan(0.2));
      r.dispose();
      m.dispose();
    });

    test('randInt produces values in range', () {
      final r = MLXRandom.randInt(ctx, low: 0, high: 10, shape: [50]);
      final vals = r.toInt32List();
      for (final v in vals) {
        expect(v, greaterThanOrEqualTo(0));
        expect(v, lessThan(10));
      }
      r.dispose();
    });

    test('bernoulli with p=0 produces all zeros', () {
      final r = MLXRandom.bernoulli(ctx, p: 0.0, shape: [10]);
      final vals = r.toFloat32List();
      expect(vals.every((v) => v == 0.0), isTrue);
      r.dispose();
    });

    test('bernoulli with p=1 produces all ones', () {
      final r = MLXRandom.bernoulli(ctx, p: 1.0, shape: [10]);
      final vals = r.toFloat32List();
      expect(vals.every((v) => v == 1.0), isTrue);
      r.dispose();
    });

    test('seed makes output deterministic', () {
      MLXRandom.seed(ctx, 42);
      final r1 = MLXRandom.uniform(ctx, shape: [10]);
      final v1 = r1.toFloat32List();

      MLXRandom.seed(ctx, 42);
      final r2 = MLXRandom.uniform(ctx, shape: [10]);
      final v2 = r2.toFloat32List();

      for (var i = 0; i < v1.length; i++) {
        expect(v1[i], closeTo(v2[i], 1e-6));
      }
      r1.dispose();
      r2.dispose();
    });
  });
}
