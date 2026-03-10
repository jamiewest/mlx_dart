import 'package:mlx_c_ffi/mlx_c_ffi.dart';
import 'package:mlx_dart/mlx_dart.dart';
import 'package:test/test.dart';

void main() {
  // -------------------------------------------------------------------------
  // Pure-Dart tests — no native library required.
  // Run with: dart test test/mlx_dart_test.dart
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
}
