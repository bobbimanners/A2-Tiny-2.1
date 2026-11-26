#!/usr/bin/env python3
import sys
from pathlib import Path
b = Path(sys.argv[1] if len(sys.argv) > 1 else 'TINY.INT').read_bytes()
h = 2166136261
for x in b:
    h = ((h ^ x) * 16777619) & 0xffffffff
print(f'Bytes: {len(b)}\nFNV-1a: {h:08X}')
