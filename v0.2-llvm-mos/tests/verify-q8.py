#!/usr/bin/env python3
"""Run actual 11N instructions in source interpreter (not native hardware).
Usage from package root: python3 tests/verify-q8.py tiny-wide-mos.s
"""
import importlib.util, pathlib, random, statistics
p=pathlib.Path(__file__).with_name('verify-wide.py')
spec=importlib.util.spec_from_file_location('wide',p)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
cpu=w.CPU(); mem=cpu.mem; rng=random.Random(0x11e); checks=0; vectors=0

def expected(v,m):
    q=(127*abs(v)+m//2)//m
    return (-q if v<0 else q)&255

def run(values,m,src=0x30ff,dst=0x42fe,args=0x54ff):
    global checks,vectors
    assert 1<=len(values)<=256 and all(abs(v)<=m for v in values)
    raw=b''.join(v.to_bytes(2,'little',signed=True) for v in values)
    a=len(values).to_bytes(2,'little')+m.to_bytes(2,'little')
    mem[src-1:src+len(raw)+1]=b'\xab'+raw+b'\xcd'
    mem[dst-1:dst+len(values)+1]=b'\xef' + bytes([0x80])*len(values)+b'\x12'
    mem[args:args+4]=a
    cpu.call('tiny_q8_vector',[src,dst,args]);got=mem[dst:dst+len(values)]
    want=bytes(expected(v,m) for v in values)
    assert got==want,(m,values,got,want)
    assert mem[src-1:src+len(raw)+1]==b'\xab'+raw+b'\xcd'
    assert mem[dst-1]==0xef and mem[dst+len(values)]==0x12
    assert mem[args:args+4]==a
    checks+=len(values);vectors+=1
    return cpu.cycles

for m in list(range(1,65))+[127,128,129,255,256,257,511,512,513,1023,1024,1025,16383,16384,16385,32766,32767]:
    if m<=129:
        vals=list(range(-m,m+1))
    else:
        mags={0,1,m-1,m}
        for q in range(127):
            k=((2*q+1)*m+253)//254
            mags.update(v for v in range(k-1,k+2) if 0<=v<=m)
        vals=sorted(mags|{-v for v in mags})
    for k in range(0,len(vals),256):run(vals[k:k+256],m)
for n in [1,2,7,24,96,127,128,129,255,256]:
    for j in range(10):
        m=rng.randrange(1,32768)
        vals=[rng.randint(-m,m) for _ in range(n)];vals[0]=rng.choice([-m,m])
        run(vals,m,0x3100+j*3,0x42f8+j,0x5500+j)
for j in range(20):
    m=rng.randint(1,32767);run([-m,0,m],m)
    a,b=rng.getrandbits(64),rng.getrandbits(64)
    mem[0x30fd:0x3105]=a.to_bytes(8,'little');mem[0x32fb:0x3303]=b.to_bytes(8,'little')
    cpu.call('wm',[0x34ff,0x30fd,0x32fb])
    assert int.from_bytes(mem[0x34ff:0x3507],'little')==a*b&w.MASK
print(f'PASS: {checks} mantissas in {vectors} vectors, plus 20 interleaved wide multiplications')
for n in [24,96,256]:
    counts=[]
    for _ in range(20):
        m=rng.randint(16384,32767);v=[rng.randint(-m,m) for _ in range(n)];v[0]=m
        counts.append(run(v,m))
    print(f'n={n}: {statistics.mean(counts):.1f} mean modeled cycles/vector; {statistics.mean(counts)/n:.1f} cycles/mantissa')
