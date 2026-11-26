#!/usr/bin/env python3
"""Convert original TNY2/TINY.BIN to the integer-only TNI6 format.
No weights are retrained or requantized; only row scales and gains change.
"""
import argparse
import math
import struct
from pathlib import Path


def convert(raw):
    if len(raw) < 12 or raw[:4] != b'TNY2':
        raise ValueError('not a TNY2 model')
    d, l, h, hid = struct.unpack_from('<4H', raw, 4)
    if not (8 <= d <= 128 and 1 <= l <= 8 and 1 <= h <= d and d % h == 0 and 8 <= hid <= 512):
        raise ValueError('unsupported model dimensions')
    out = bytearray(b'TNI6' + raw[4:12])
    out += struct.pack('<I', math.floor(2**44 / (127**2 * math.sqrt(d//h)) + .5))
    off = 12

    def take(n):
        nonlocal off
        if off + n > len(raw):
            raise ValueError('truncated model')
        v = raw[off:off+n]
        off += n
        return v

    def coefficient():
        v, = struct.unpack('<f', take(4))
        if not math.isfinite(v):
            raise ValueError('non-finite model coefficient')
        return v

    def encode(values, zero_exp):
        big = max(map(abs, values))
        exp = math.frexp(big)[1] - 23 if big else zero_exp
        q = [math.floor(math.ldexp(v, -exp)+.5) if v >= 0 else -math.floor(-math.ldexp(v, -exp)+.5) for v in values]
        out.extend(struct.pack('<h', exp))
        return q

    def table(rows, cols):
        scales, weights = [], []
        for _ in range(rows):
            v = coefficient()
            if v < 0:
                raise ValueError('negative row scale')
            scales.append(v)
            weights.append(take(cols))
        for q, w in zip(encode(scales, -23), weights):
            out.extend(struct.pack('<i', q))
            out.extend(w)

    def gain():
        q = encode([coefficient() for _ in range(d)], 0)
        out.extend(struct.pack('<%di' % d, *q))

    table(97, d)
    table(128, d)
    for _ in range(l):
        gain()
        for _ in range(4):
            table(d, d)
        gain()
        table(hid, d)
        table(d, hid)
    gain()
    if off != len(raw):
        raise ValueError('unexpected trailing model data')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', nargs='?', default='TINY.BIN', type=Path)
    p.add_argument('destination', nargs='?', default='TINY.INT', type=Path)
    a = p.parse_args()
    try:
        if a.source.resolve() == a.destination.resolve():
            raise ValueError('source and destination must differ')
        raw = a.source.read_bytes()
        out = convert(raw)
        a.destination.write_bytes(out)
        print(f'{a.source}: {len(raw)} bytes -> {a.destination}: {len(out)} bytes')
    except (OSError, ValueError, struct.error) as e:
        p.exit(1, f'TINY converter: {e}\n')

if __name__ == '__main__':
    main()
