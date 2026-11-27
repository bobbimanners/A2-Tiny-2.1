#!/usr/bin/env python3
"""Source-level 65C02 interpreter for this file's instruction subset.
Not an assembler or a replacement for running WIDETEST on the Apple II.
Usage: python3 tests/verify-wide.py [baseline-wide.s] [--full]
Also checks approximate branch displacements using real instruction lengths.
"""
import re, random, pathlib, sys, statistics
root=pathlib.Path(__file__).resolve().parent.parent
MASK=(1<<64)-1
src=pathlib.Path(sys.argv[1]).read_text() if len(sys.argv)>1 and not sys.argv[1].startswith('--') else (root / 'tiny-wide-mos.s').read_text()
# Load the actual shared lookup planes, not independently generated data.
rom=bytearray(65536)
for i,(name,block) in enumerate(re.findall(r'(ql|qh):\s*(.*?)(?=^qh:|\Z)', (root/'tiny-qsquare-mos.s').read_text(),re.M|re.S)):
    values=[int(x,16) for x in re.findall(r'\$([0-9a-fA-F]{2})\b',block)]
    assert len(values)==512, (name,len(values))
    rom[0x8000+i*512:0x8200+i*512]=bytes(values)
syms={"ql":0x8000,"qh":0x8200}; labels={}; code=[]; locs=[]; pc=0x2000; data=0x7000; section=''
branches={'bcc','bcs','beq','bne','bmi','bpl','bra'}
def val(s):
    if '+' in s:
        a,b=s.split('+');return val(a)+val(b)
    if s.startswith('$'):return int(s[1:],16)
    if s.isdigit():return int(s)
    return syms[s]
for line in src.splitlines():
    line=line.split(';')[0].strip()
    if not line:continue
    if line.startswith('.section'):
        section='data' if '.bss' in line else 'code';continue
    if line.startswith('.global'):continue
    if '=' in line:
        a,b=map(str.strip,line.split('='));syms[a]=val(b);continue
    if ':' in line:
        name,line=map(str.strip,line.split(':',1));syms[name]=data if section=='data' else pc
        if section=='code':labels[name]=len(code)
        if not line:continue
    if line.startswith('.space'):
        data+=int(line.split()[1]);continue
    op,*args=line.split(None,1);arg=args[0].strip() if args else ''
    locs.append(pc);code.append((op,arg))
    if op in branches:pc+=2
    elif op in ('jmp','jsr'):pc+=3
    elif not arg or arg=='a':pc+=1
    elif arg.startswith(('#','(')):pc+=2
    else:
        base=arg.split(',')[0]
        pc+=2 if val(base)<256 and not arg.endswith(',y') else 3
for i,(op,arg) in enumerate(code):
    if op in branches:
        disp=syms[arg]-(locs[i]+2)
        if not -128<=disp<=127:raise ValueError(f'branch range {op} {arg} from {locs[i]:04x}: {disp}')
print(f'Estimated assembly code: {pc-0x2000} bytes; scratch: {data-0x7000} bytes; branches in range')
class CPU:
    def __init__(self):self.mem=bytearray(rom);self.cycles=0
    def call(self,name,ptrs,n=0):
        m=self.mem
        for i,p in enumerate(ptrs):m[2+2*i:4+2*i]=p.to_bytes(2,'little')
        a=n&255;x=(n>>8)&255;y=0;c=z=N=0;stack=[];ret=[];ip=labels[name];steps=0;cycles=0
        def nz(v):return int(v==0),int(bool(v&128))
        while True:
            steps+=1
            if steps>100000:raise RuntimeError(name)
            op,arg=code[ip];at=locs[ip];ip+=1
            addr=None;v=None;cross=0;mode="imp"
            if arg and op not in branches|{'jmp','jsr'}:
                if arg=='a':v=a
                elif arg.startswith('#'):v=val(arg[1:]);mode='imm'
                elif arg.startswith('('):
                    p=val(arg[1:arg.index(')')]);base=m[p]|m[p+1]<<8;addr=(base+y)&65535;v=m[addr];mode='indy';cross=int(bool((base^addr)&0xff00))
                else:
                    b,*idx=arg.split(',');base=val(b);addr=(base+(x if idx==['x'] else y if idx==['y'] else 0))&65535;v=m[addr];mode=('zpidx' if base<256 and idx==['x'] else 'absidx' if idx else 'zp' if base<256 else 'abs');cross=int(bool((base^addr)&0xff00))
            if op in branches:
                take={'bcc':not c,'bcs':c,'beq':z,'bne':not z,'bmi':N,'bpl':not N,'bra':True}[op]
                cycles+=2+int(take)*(1+int(bool(((at+2)^syms[arg])&0xff00)))
            elif op in ('jsr','rts'):cycles+=6
            elif op=='jmp':cycles+=3
            elif op in ('pha','php'):cycles+=3
            elif op in ('pla','plp'):cycles+=4
            elif op in ('sta','stx','sty','stz'):cycles+={'zp':3,'zpidx':4,'abs':4,'absidx':5,'indy':6}[mode]
            elif op in ('asl','lsr','rol','ror','inc','dec'):cycles+={'imp':2,'zp':5,'zpidx':6,'abs':6,'absidx':6+cross}[mode]
            else:cycles+={'imp':2,'imm':2,'zp':3,'zpidx':4,'abs':4,'absidx':4+cross,'indy':5+cross}[mode]
            if op in ('lda','ldx','ldy'):
                if op=='lda':a=v
                elif op=='ldx':x=v
                else:y=v
                z,N=nz(v)
            elif op in ('sta','stx','sty','stz'):m[addr]={'sta':a,'stx':x,'sty':y,'stz':0}[op]
            elif op in ('tax','tay','txa','tya'):
                if op=='tax':x=a;v=x
                elif op=='tay':y=a;v=y
                elif op=='txa':a=x;v=a
                else:a=y;v=a
                z,N=nz(v)
            elif op in ('inx','dex','iny','dey'):
                if op in ('inx','dex'):x=(x+(1 if op=='inx' else -1))&255;v=x
                else:y=(y+(1 if op=='iny' else -1))&255;v=y
                z,N=nz(v)
            elif op in ('clc','sec'):c=int(op=='sec')
            elif op in ('adc','sbc'):
                t=a+v+c if op=='adc' else a-v-(1-c);c=int(t>255) if op=='adc' else int(t>=0);a=t&255;z,N=nz(a)
            elif op in ('cmp','cpx','cpy'):
                t={'cmp':a,'cpx':x,'cpy':y}[op]-v;c=int(t>=0);z,N=nz(t&255)
            elif op in ('ora','eor','and'):
                a={'ora':a|v,'eor':a^v,'and':a&v}[op];z,N=nz(a)
            elif op=='bit':z=int((a&v)==0);N=int(bool(v&128))
            elif op in ('asl','lsr','rol','ror','inc','dec'):
                if op=='asl':c=v>>7;v=(v<<1)&255
                elif op=='lsr':c=v&1;v>>=1
                elif op=='rol':old=c;c=v>>7;v=((v<<1)|old)&255
                elif op=='ror':old=c;c=v&1;v=(v>>1)|(old<<7)
                else:v=(v+(1 if op=='inc' else -1))&255
                if arg=='a':a=v
                else:m[addr]=v
                z,N=nz(v)
            elif op in branches:
                if {'bcc':not c,'bcs':c,'beq':z,'bne':not z,'bmi':N,'bpl':not N,'bra':True}[op]:ip=labels[arg]
            elif op=='jmp':ip=labels[arg]
            elif op=='jsr':ret.append(ip);ip=labels[arg]
            elif op=='rts':
                if not ret:
                    assert not stack
                    self.cycles=cycles
                    return a|(x<<8)
                ip=ret.pop()
            elif op=='pha':stack.append(a)
            elif op=='pla':a=stack.pop();z,N=nz(a)
            elif op=='php':stack.append((c,z,N))
            elif op=='plp':c,z,N=stack.pop()
            else:raise ValueError(op)

def signed(v):return v-(1<<64) if v>>63 else v
def cmp(a,b):return (a>b)-(a<b)
def main():
    rng=random.Random(0x6502);cpu=CPU();m=cpu.mem
    # Cross page boundaries deliberately, and exercise exact input aliases.
    a0,b0,r0=0x30fd,0x32fb,0x34ff
    edges=[0,1,2,127,128,255,256,257,65535,65536,0xffffffff,0x100000000,1<<63,(1<<63)-1,MASK,MASK-1]
    cases=[(a,b) for a in edges for b in edges]+[(rng.getrandbits(64),rng.choice([rng.getrandbits(64),rng.randrange(1,256),rng.getrandbits(32)])) for _ in range(150)]
    checks=0
    def put(p,v):m[p:p+8]=(v&MASK).to_bytes(8,'little')
    def get(p):return int.from_bytes(m[p:p+8],'little')
    for i,(a,b) in enumerate(cases):
        n=[-32768,-1,0,1,7,8,9,15,16,31,32,33,63,64,65,256,32767][i%17]
        binary={'wa':a+b,'ws':a-b,'wm':a*b}
        if b:binary.update(tiny_wdiv=a//b,tiny_wrounddiv=a//b+int(a%b>=b-a%b))
        for name,expect in binary.items():
            for r in (r0,a0,b0):
                put(a0,a);put(b0,b);cpu.call(name,[r,a0,b0]);assert get(r)==expect&MASK,(name,hex(a),hex(b),hex(get(r)),hex(expect&MASK));checks+=1
        for name,expect in [('wneg',-a),('wmag',abs(signed(a))),('wl',a if n<=0 else 0 if n>=64 else a<<n),('wr',a if n<=0 else 0 if n>=64 else a>>n)]:
            for r in (r0,a0):
                put(a0,a);cpu.call(name,[r,a0],n);assert get(r)==expect&MASK,(name,hex(a),n,hex(get(r)),hex(expect&MASK));checks+=1
        put(a0,a);put(b0,b)
        for name,args,expect in [('wz',[a0],int(a==0)),('wn',[a0],a>>63),('wc',[a0,b0],cmp(a,b)),('wcs',[a0,b0],cmp(signed(a),signed(b))),('topbit',[a0],max(0,a.bit_length()-1)),('wbit',[a0],(a>>n)&1 if 0<=n<64 else 0)]:
            got=cpu.call(name,args,n);assert got==expect&65535,(name,hex(a),hex(b),n,got,expect);checks+=1
        cpu.call('winc',[a0]);assert get(a0)==(a+1)&MASK;checks+=1
    print(f'PASS: {checks} source-interpreter checks ({len(cases)} operand pairs)')
def multiply_bench():
    rng=random.Random(12345); cpu=CPU();m=cpu.mem; mask=MASK
    for wa,wb in [(8,8),(16,16),(24,24),(32,16),(32,32),(48,16),(64,64)]:
        counts=[]
        for i in range(100):
            a=rng.getrandbits(wa);b=rng.getrandbits(wb)
            if i%3==0:a=(-a)&mask
            if i%5==0:b=(-b)&mask
            m[0x30fd:0x3105]=a.to_bytes(8,'little');m[0x32fb:0x3303]=b.to_bytes(8,'little')
            cpu.call('wm',[0x34ff,0x30fd,0x32fb]);assert int.from_bytes(m[0x34ff:0x3507],'little')==a*b&mask
            counts.append(cpu.cycles)
        print(f'{wa}x{wb} mixed signs: {statistics.mean(counts):.1f} mean modeled cycles')
    # Check all byte products in the actual table, including sum carry.
    q=[rom[0x8100+i]+256*rom[0x8300+i] for i in range(256)]
    for a in range(256):
        for b in range(256):
            t=(a+b)&255
            assert q[t]+((128*t+16384) if a+b>=256 else 0)-q[abs(a-b)]==a*b
    print('PASS: all 65,536 unsigned byte pairs via actual table identity')
    if 'MLA' in syms:
        # Run the actual wide routine for every byte pair, including high-byte carry.
        # The --full option may take a minute or two under Python.
        vals=range(256) if '--full' in sys.argv else sorted(set(range(0,256,4))|{1,127,128,129,254,255})
        n=0
        for a in vals:
            for b in vals:
                m[0x30fd:0x3105]=a.to_bytes(8,'little');m[0x32fb:0x3303]=b.to_bytes(8,'little')
                cpu.call('wm',[0x34ff,0x30fd,0x32fb]);assert int.from_bytes(m[0x34ff:0x3507],'little')==a*b,(a,b)
                n+=1
        print(f'PASS: {n} byte pairs through wm assembly')
if __name__=='__main__':main();multiply_bench()
