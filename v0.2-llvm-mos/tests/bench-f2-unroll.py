"""Replay the standard model's sparse F2 projections on a 65C02.

From v0.2-llvm-mos, run python tests/bench-f2-unroll.py. Set MOSBIN to the
LLVM-MOS SDK bin directory if it is outside PATH. GCC (CC) and cc65 sim65
must also be available. All generated files live in temporary storage.
"""
import os
import re
import runpy
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, directory, env=None):
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{command[0]} exited {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result.stdout + result.stderr


def trace_workload(directory):
    source = (ROOT / 'tiny.c').read_text()
    start = 'static void matvec_prepared(const Table *t,Vec *dst) {'
    read = '        rw_read_at(bank,off,rowbuf,n);'
    assert source.count(start) == source.count(read) == 1
    capture = '''
    static FILE *trace;
    if(t->cols==256){
        uint8_t header[4];
        put16(header,t->data.off);
        put16(header+2,t->data.first);
        if(!trace)trace=fopen("f2.bin","wb");
        if(!trace)exit(2);
        fwrite(header,sizeof(header),1,trace);
        fwrite(projection_q,1,256,trace);
    }
'''
    source = source.replace(start, start + capture)
    source = source.replace(read, read + '\n        if(t->cols==256)fwrite(rowbuf,1,n,trace);')
    (directory / 'host.c').write_text(source)
    converter = runpy.run_path(str(ROOT.parent / 'v0.1-cc65' / 'tiny-convert.py'))
    (directory / 'TINY.INT').write_bytes(converter['convert']((ROOT.parent / 'TINY.BIN').read_bytes()))
    compiler = os.environ.get('CC', 'gcc')
    env = os.environ.copy()
    env['PATH'] = str(Path(shutil.which(compiler) or compiler).parent) + os.pathsep + env['PATH']
    run([compiler, '-O2', '-DTINY_SMALL', '-DTINY_U64', '-DTINY_COMPACT_LUT',
         '-I', str(ROOT), 'host.c', '-o', 'host.exe'], directory, env)
    print(run([directory / 'host.exe', '-q', 'WHAT IS THE CAPITAL OF FRANCE?'], directory, env).strip())
    trace = (directory / 'f2.bin').read_bytes()
    record_size = 4 + 256 + 96 * 260
    assert len(trace) == record_size * 218
    return [trace[i:i + record_size] for i in range(0, len(trace), record_size)]


def assemble(directory):
    mac = (ROOT / 'tiny-mac65-mos.s').read_text()
    start = '.Lrows_some:\n    ldx .Lprepared_first+1'
    end = '    MACROW 0\n    lda tiny_mac65_work+256,x'
    assert mac.count(start) == mac.count(end) == 1
    mac = mac.replace(start, '.Lrows_some:\n    jmp .Lbench_dispatch')
    mac = mac.replace(end, '    MACROW 0\n.Lbench_tail:\n    lda tiny_mac65_work+256,x')
    mac += (ROOT / 'tests' / 'bench-f2-unroll.s').read_text()
    (directory / 'bench.s').write_text(mac)
    (directory / 'bench.ld').write_text('''
MEMORY {
    zp : ORIGIN = 0x80, LENGTH = 0x80
    ram : ORIGIN = 0x2000, LENGTH = 0x2000
}
SECTIONS {
    .text : { *(.text) } >ram
    .zp.bss (NOLOAD) : { *(.zp.bss) } >zp
}
OUTPUT_FORMAT { TRIM(ram) }
''')
    suffix = '.bat' if os.name == 'nt' else ''
    mosbin = os.environ.get('MOSBIN')
    cc = 'mos-apple2-clang' + suffix
    cc = str(Path(mosbin) / cc) if mosbin else cc
    nm = str(Path(mosbin) / 'llvm-nm') if mosbin else 'llvm-nm'
    run([cc, '-mcpu=mos65c02', '-nostdlib', '-T', 'bench.ld', '-Wl,--no-gc-sections',
         '-Wl,-e,bench_start', str(ROOT / 'tiny-qsquare-mos.s'), 'bench.s', '-o', 'bench'], directory)
    symbols = {}
    for line in run([nm, '-n', 'bench.elf'], directory).splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]+)\s+\w\s+(\w+)', line)
        if match:
            symbols[match[2]] = int(match[1], 16)
    binary = (directory / 'bench').read_bytes()
    assert len(binary) <= 8192 and symbols['bench_slots'] + 1021 <= 0x4000
    return binary, symbols


def fixture(binary, symbols, vector):
    off, first = struct.unpack_from('<HH', vector)
    raw = vector[4:260]
    activations = struct.unpack('<256b', raw)
    active = [(i, u + 128) for i, u in enumerate(activations) if u]
    memory = bytearray(0xc000)
    memory[0x2000:0x2000 + len(binary)] = binary
    work = symbols['tiny_mac65_work']
    memory[work:work + 256] = raw
    records = vector[260:]
    for i in range(96):
        row = records[i * 260:(i + 1) * 260]
        address = off + i * 260 if i < first else 0x4000 + (i - first) * 260
        assert address + 260 <= 0xc000
        memory[address:address + 4] = row[:4]
        memory[address + 4:address + 260] = bytes(w ^ 128 for w in row[4:])
        dot = sum(a * w for a, w in zip(activations, struct.unpack('<256b', row[4:])))
        expected = symbols['bench_expected'] + 8 * i
        memory[expected:expected + 8] = row[:4] + struct.pack('<i', dot)

    batches = []
    for address, count in ((off, first), (0x4000, 96 - first)):
        for row in range(0, count, 8):
            batches.append(struct.pack('<HB', address + 260 * row, min(8, count - row)))
    table = symbols['bench_rows']
    data = b''.join(batches)
    memory[table:table + len(data)] = data
    return memory, active


def replay(directory, symbols, memory, active, mode):
    image = memory.copy()
    image[symbols['bench_mode']] = mode
    if mode == 2 and 0 < len(active) <= 60:
        image[symbols['bench_active']] = 1
        slot = symbols['bench_slots']
        for column, u in active:
            image[slot + 1] = column
            image[slot + 3] = u
            image[slot + 9] = 256 - u
            slot += 17
        image[slot] = 0x60
    path = directory / 'fixture.sim'
    header = b'sim65' + bytes([2, 1, 0x20]) + struct.pack('<HH', 0x0800, symbols['bench_start'])
    path.write_bytes(header + image[0x0800:])
    result = run(['sim65', '-c', path], directory)
    return int(re.search(r'(\d+) cycles', result)[1])


def main():
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        workload = trace_workload(directory)
        binary, symbols = assemble(directory)
        # Check termination, first and last columns, and both sides of the
        # code-capacity cutoff using real rows separated by a bank boundary.
        crossing = next(vector for vector in workload if struct.unpack_from('<H', vector, 2)[0] < 96)
        for count in (0, 1, 60, 61, 256):
            edge = bytearray(crossing)
            edge[4:260] = bytes(256 - count) + bytes([127]) * count
            memory, active = fixture(binary, symbols, edge)
            for mode in range(3):
                replay(directory, symbols, memory, active, mode)
        totals = [0, 0, 0]
        specialized = 0
        for vector in workload:
            memory, active = fixture(binary, symbols, vector)
            specialized += 0 < len(active) <= 60
            for mode in range(3):
                totals[mode] += replay(directory, symbols, memory, active, mode)
        print(f'{len(workload)} F2 vectors, {specialized} within 60-column code capacity.')
        print(f'Linked loop: {totals[0]:,} cycles')
        print(f'Unrolled, including target generation: {totals[1]:,} cycles')
        print(f'Unrolled with free generation: {totals[2]:,} cycles')


if __name__ == '__main__':
    main()
