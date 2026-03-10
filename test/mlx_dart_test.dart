import 'package:mlx_c_ffi/mlx_c_ffi.dart';
import 'package:mlx_dart/mlx_dart.dart';
import 'package:test/test.dart';

/// Returns true when the native MLX library is available on this machine.
bool _nativeLibAvailable() {
  try {
    MLXContext.gpu();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  // -------------------------------------------------------------------------
  // Pure-Dart tests — no native library required.
  // -------------------------------------------------------------------------

  group('MLXException', () {
    test('toString without message', () {
      const ex = MLXException('mlx_add', 1);
      expect(ex.toString(), equals('MLXException(mlx_add): code=1'));
    });

    test('toString with message', () {
      const ex = MLXException('mlx_matmul', 42, 'shape mismatch');
      expect(
        ex.toString(),
        equals('MLXException(mlx_matmul): code=42, message=shape mismatch'),
      );
    });

    test('fields are accessible', () {
      const ex = MLXException('op', 7, 'msg');
      expect(ex.operation, equals('op'));
      expect(ex.code, equals(7));
      expect(ex.message, equals('msg'));
    });

    test('is an Exception', () {
      const ex = MLXException('op', 1);
      expect(ex, isA<Exception>());
    });
  });

  group('MLXDtype', () {
    test('has 14 variants', () {
      expect(MLXDtype.values.length, equals(14));
    });

    test('fromRaw round-trips for every variant', () {
      for (final dtype in MLXDtype.values) {
        expect(MLXDtype.fromRaw(dtype.raw), equals(dtype));
      }
    });

    test('bool_ maps to MLX_BOOL', () {
      expect(MLXDtype.bool_.raw, equals(mlx_dtype_.MLX_BOOL));
    });

    test('float32 maps to MLX_FLOAT32', () {
      expect(MLXDtype.float32.raw, equals(mlx_dtype_.MLX_FLOAT32));
    });

    test('int32 maps to MLX_INT32', () {
      expect(MLXDtype.int32.raw, equals(mlx_dtype_.MLX_INT32));
    });

    test('bfloat16 maps to MLX_BFLOAT16', () {
      expect(MLXDtype.bfloat16.raw, equals(mlx_dtype_.MLX_BFLOAT16));
    });
  });

  // -------------------------------------------------------------------------
  // Integration tests — skipped when the native library is not present.
  // -------------------------------------------------------------------------

  group('MLXContext', () {
    late bool available;

    setUpAll(() {
      available = _nativeLibAvailable();
      if (!available) {
        // ignore: avoid_print
        print('Native MLX library not found — skipping integration tests.');
      }
    });

    test('gpu() constructs without throwing', () {
      if (!available) return;
      final ctx = MLXContext.gpu();
      expect(ctx, isNotNull);
      ctx.dispose();
    }, skip: !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null);

    test('cpu() constructs without throwing', () {
      if (!available) return;
      final ctx = MLXContext.cpu();
      expect(ctx, isNotNull);
      ctx.dispose();
    }, skip: !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null);

    test('version returns a non-empty string', () {
      if (!available) return;
      final ctx = MLXContext.gpu();
      expect(ctx.version, isNotEmpty);
      ctx.dispose();
    }, skip: !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null);
  });

  group('MLXArray construction', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('zeros has correct shape and dtype', () {
      final a = MLXArray.zeros(ctx!, [2, 3]);
      expect(a.shape, equals([2, 3]));
      expect(a.ndim, equals(2));
      expect(a.dtype, equals(MLXDtype.float32));
      expect(a.size, equals(6));
      a.dispose();
    }, skip: skip);

    test('ones has correct shape', () {
      final a = MLXArray.ones(ctx!, [4]);
      expect(a.shape, equals([4]));
      a.dispose();
    }, skip: skip);

    test('fromFloats round-trips', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final a = MLXArray.fromFloats(ctx!, data, shape: [2, 2]);
      expect(a.toFloat32List(), equals(data));
      a.dispose();
    }, skip: skip);

    test('fromInts round-trips', () {
      final data = [10, 20, 30];
      final a = MLXArray.fromInts(ctx!, data);
      expect(a.toInt32List(), equals(data));
      a.dispose();
    }, skip: skip);

    test('arange produces correct values', () {
      final a = MLXArray.arange(ctx!, 0.0, 5.0, 1.0);
      expect(a.toFloat32List(), equals([0.0, 1.0, 2.0, 3.0, 4.0]));
      a.dispose();
    }, skip: skip);

    test('bool_ scalar', () {
      final a = MLXArray.bool_(ctx!, true);
      expect(a.dtype, equals(MLXDtype.bool_));
      a.dispose();
    }, skip: skip);

    test('int_ scalar', () {
      final a = MLXArray.int_(ctx!, 42);
      expect(a.itemInt(), equals(42));
      a.dispose();
    }, skip: skip);

    test('float_ scalar', () {
      final a = MLXArray.float_(ctx!, 3.14);
      expect(a.itemFloat(), closeTo(3.14, 1e-5));
      a.dispose();
    }, skip: skip);
  });

  group('MLXArray arithmetic', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('add', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 2.0]);
      final b = MLXArray.fromFloats(ctx!, [3.0, 4.0]);
      final c = a + b;
      expect(c.toFloat32List(), equals([4.0, 6.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    }, skip: skip);

    test('subtract', () {
      final a = MLXArray.fromFloats(ctx!, [5.0, 6.0]);
      final b = MLXArray.fromFloats(ctx!, [1.0, 2.0]);
      final c = a - b;
      expect(c.toFloat32List(), equals([4.0, 4.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    }, skip: skip);

    test('multiply', () {
      final a = MLXArray.fromFloats(ctx!, [2.0, 3.0]);
      final b = MLXArray.fromFloats(ctx!, [4.0, 5.0]);
      final c = a * b;
      expect(c.toFloat32List(), equals([8.0, 15.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    }, skip: skip);

    test('matmul 2x2', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
      final b = MLXArray.ones(ctx!, [2, 2]);
      final c = a.matmul(b);
      expect(c.toFloat32List(), equals([3.0, 3.0, 7.0, 7.0]));
      a.dispose();
      b.dispose();
      c.dispose();
    }, skip: skip);
  });

  group('MLXArray shape ops', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('reshape', () {
      final a = MLXArray.arange(ctx!, 0.0, 6.0, 1.0);
      final b = a.reshape([2, 3]);
      expect(b.shape, equals([2, 3]));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('transpose T', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
      final b = a.T;
      expect(b.shape, equals([2, 2]));
      expect(b.toFloat32List(), equals([1.0, 3.0, 2.0, 4.0]));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('flatten', () {
      final a = MLXArray.zeros(ctx!, [2, 3, 4]);
      final b = a.flatten();
      expect(b.shape, equals([24]));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('expandDims', () {
      final a = MLXArray.zeros(ctx!, [4]);
      final b = a.expandDims(0);
      expect(b.shape, equals([1, 4]));
      a.dispose();
      b.dispose();
    }, skip: skip);
  });

  group('MLXArray reductions', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('sum all', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 2.0, 3.0, 4.0]);
      final s = a.sum();
      expect(s.itemFloat(), closeTo(10.0, 1e-5));
      a.dispose();
      s.dispose();
    }, skip: skip);

    test('mean all', () {
      final a = MLXArray.fromFloats(ctx!, [2.0, 4.0, 6.0, 8.0]);
      final m = a.mean();
      expect(m.itemFloat(), closeTo(5.0, 1e-5));
      a.dispose();
      m.dispose();
    }, skip: skip);

    test('max all', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 5.0, 3.0]);
      final mx = a.max();
      expect(mx.itemFloat(), closeTo(5.0, 1e-5));
      a.dispose();
      mx.dispose();
    }, skip: skip);

    test('argmax', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 5.0, 3.0]);
      final idx = a.argmax();
      expect(idx.itemInt(), equals(1));
      a.dispose();
      idx.dispose();
    }, skip: skip);
  });

  group('MLXArray element-wise math', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('exp', () {
      final a = MLXArray.float_(ctx!, 0.0);
      final b = a.exp();
      expect(b.itemFloat(), closeTo(1.0, 1e-5));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('sqrt', () {
      final a = MLXArray.float_(ctx!, 4.0);
      final b = a.sqrt();
      expect(b.itemFloat(), closeTo(2.0, 1e-5));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('abs', () {
      final a = MLXArray.float_(ctx!, -3.0);
      final b = a.abs();
      expect(b.itemFloat(), closeTo(3.0, 1e-5));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('relu zeros negatives', () {
      final a = MLXArray.fromFloats(ctx!, [-1.0, 0.0, 2.0]);
      final b = a.relu();
      expect(b.toFloat32List(), equals([0.0, 0.0, 2.0]));
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('sigmoid at 0 is 0.5', () {
      final a = MLXArray.float_(ctx!, 0.0);
      final b = a.sigmoid();
      expect(b.itemFloat(), closeTo(0.5, 1e-5));
      a.dispose();
      b.dispose();
    }, skip: skip);
  });

  group('free functions', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('concatenate along axis 0', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 2.0], shape: [1, 2]);
      final b = MLXArray.fromFloats(ctx!, [3.0, 4.0], shape: [1, 2]);
      final c = concatenate(ctx!, [a, b]);
      expect(c.shape, equals([2, 2]));
      a.dispose();
      b.dispose();
      c.dispose();
    }, skip: skip);

    test('stack along new axis', () {
      final a = MLXArray.fromFloats(ctx!, [1.0, 2.0]);
      final b = MLXArray.fromFloats(ctx!, [3.0, 4.0]);
      final c = stack(ctx!, [a, b]);
      expect(c.shape, equals([2, 2]));
      a.dispose();
      b.dispose();
      c.dispose();
    }, skip: skip);

    test('where selects correctly', () {
      final cond = MLXArray.fromInts(ctx!, [1, 0, 1], dtype: MLXDtype.bool_);
      final x = MLXArray.fromFloats(ctx!, [10.0, 20.0, 30.0]);
      final y = MLXArray.fromFloats(ctx!, [1.0, 2.0, 3.0]);
      final r = where(ctx!, cond, x, y);
      expect(r.toFloat32List(), equals([10.0, 2.0, 30.0]));
      cond.dispose();
      x.dispose();
      y.dispose();
      r.dispose();
    }, skip: skip);

    test('evalAll evaluates multiple arrays', () {
      final a = MLXArray.zeros(ctx!, [3]);
      final b = MLXArray.ones(ctx!, [3]);
      evalAll(ctx!, [a, b]);
      a.dispose();
      b.dispose();
    }, skip: skip);

    test('createCausalMask shape', () {
      final mask = createCausalMask(ctx!, n: 4, offset: 0);
      expect(mask.shape, equals([4, 4]));
      mask.dispose();
    }, skip: skip);
  });

  group('MLXArray disposal', () {
    MLXContext? ctx;
    final skip = !_nativeLibAvailable() ? 'libmlxc.dylib not found' : null;

    setUpAll(() {
      if (_nativeLibAvailable()) ctx = MLXContext.gpu();
    });
    tearDownAll(() => ctx?.dispose());

    test('isDisposed is false before disposal', () {
      final a = MLXArray.zeros(ctx!, [2]);
      expect(a.isDisposed, isFalse);
      a.dispose();
    }, skip: skip);

    test('isDisposed is true after disposal', () {
      final a = MLXArray.zeros(ctx!, [2]);
      a.dispose();
      expect(a.isDisposed, isTrue);
    }, skip: skip);

    test('double dispose does not throw', () {
      final a = MLXArray.zeros(ctx!, [2]);
      a.dispose();
      expect(() => a.dispose(), returnsNormally);
    }, skip: skip);

    test('toString on disposed array returns sentinel', () {
      final a = MLXArray.zeros(ctx!, [2]);
      a.dispose();
      expect(a.toString(), equals('MLXArray(disposed)'));
    }, skip: skip);

    test('using disposed array throws StateError', () {
      final a = MLXArray.zeros(ctx!, [2]);
      a.dispose();
      expect(() => a.shape, throwsStateError);
    }, skip: skip);
  });
}
