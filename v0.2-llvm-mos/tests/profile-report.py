"""Call-stack profile report. Usage: report2.py <dir> [phase]"""
import collections, pathlib, sys

d = pathlib.Path(sys.argv[1])
phase = sys.argv[2] if len(sys.argv) > 2 else "infer"
stacks = []
for line in d.joinpath("stacks.txt").read_text().splitlines():
    n, key = line.split(" ", 1)
    ph, st = key.split("|", 1)
    if ph == phase:
        stacks.append((int(n), st.split(";")))
# The stack walk occasionally misreads one frame (a pushed data byte pairs with a
# return-address byte). Repair the two patterns seen: a wrong name where 'step'
# or 'apple_answer' must be, identified by the frames around it.
STEP_CHILDREN = {"matvec", "attention", "normalize", "residual", "row", "cache_store", "fetch_gain", "rw_read"}
fixed = 0
for i, (n, st) in enumerate(stacks):
    if "step" not in st:
        for k in range(1, len(st)):
            if st[k] in STEP_CHILDREN and st[k - 1] != "step":
                st[k - 1] = "step"; fixed += n; break
    if "step" in st:
        k = st.index("step")
        if k >= 1 and st[k - 1] != "apple_answer" and "apple_answer" not in st:
            st[k - 1] = "apple_answer"; fixed += n
total = sum(n for n, _ in stacks)
print(f"repaired frames in {fixed} samples")
print(f"phase {phase}: {total} samples = {total/60:.0f} s")

self_t, incl = collections.Counter(), collections.Counter()
for n, st in stacks:
    self_t[st[-1]] += n
    for f in set(st):
        incl[f] += n

def pct(n):
    return f"{100*n/total:6.2f}%"

print("\n== inclusive / self by function ==")
for f, n in incl.most_common(60):
    print(f"{pct(n)} incl  {pct(self_t[f])} self  {f}")

def children_of(parent, depth_names=1):
    """Inclusive time by the function directly below `parent` in the stack."""
    c = collections.Counter()
    for n, st in stacks:
        if parent in st:
            i = len(st) - 1 - st[::-1].index(parent)
            c[st[i + 1] if i + 1 < len(st) else "<self>"] += n
    return c

for parent in ("step", "matvec", "attention", "normalize", "residual", "pack", "row", "rshift", "align",
               "delta_q16", "cache_store", "rw_read", "apple_answer"):
    c = children_of(parent)
    if not c:
        continue
    t = sum(c.values())
    print(f"\n== below {parent} ({pct(t)} of total) ==")
    for f, n in c.most_common(14):
        print(f"{pct(n)}  {f}")

def callers_of(leaf):
    c = collections.Counter()
    for n, st in stacks:
        if leaf in st:
            i = st.index(leaf)
            c[st[i - 1] if i > 0 else "<root>"] += n
    return c

for leaf in ("wm", "wr", "wl", "tiny_wrounddiv", "__mulsi3", "__mulhi3", "rw_addr", "tiny_rw65_move", "wneg", "wmag", "wi", "wu", "topbit", "update_hi"):
    c = callers_of(leaf)
    if not c:
        continue
    t = sum(c.values())
    print(f"\n== callers of {leaf} ({pct(t)} inclusive) ==")
    for f, n in c.most_common(8):
        print(f"{pct(n)}  {f}")

# Which top-level activity (direct child of step) owns each sample, by leaf group.
print("\n== unusual stacks (no 'step' in stack) ==")
c = collections.Counter()
for n, st in stacks:
    if "step" not in st:
        c[";".join(st[-3:])] += n
for k, n in c.most_common(8):
    print(f"{pct(n)}  {k}")
s = d.joinpath("series.txt")
if s.exists() and phase == "infer":
    print("\n== per-minute share: attention / matvec ==")
    rows = [list(map(int, l.split())) for l in s.read_text().splitlines()]
    for b, tot, att, mat in rows[:: max(1, len(rows) // 12)]:
        print(f"minute {b:3d}: attention {100*att/tot:5.1f}%  matvec {100*mat/tot:5.1f}%")
