"""Post-process paper/tab_general.tex: within each block except the released checkpoints, bold the best value of
every column (lowest MAE, rho, Delta-MAE; highest Top-1) and tint the EqAlign rows. Idempotent; run after make_general.py."""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else 'paper/tab_general.tex'
L = open(p).read().split('\n')
if any('\\hl{' in l or '\\textbf{' in l for l in L): sys.exit(0)
blocks, cur = [], []
for i, l in enumerate(L):
    if l.startswith('\\midrule') or l.startswith('\\bottomrule'):
        if cur: blocks.append(cur)
        cur = []
    elif l.count('&') == 8 and re.search(r'\d\.\d', l): cur.append(i)
num = lambda s: float(s.replace('\\\\', '').strip()) if re.fullmatch(r'-?\d+\.\d+', s.replace('\\\\', '').strip()) else None
for b in blocks:
    rows = {i: [c for c in L[i].split('&')] for i in b}
    if any('Public' in L[i] for i in b): continue
    for col in range(2, 9):
        vals = {i: num(rows[i][col]) for i in b}
        vs = [v for v in vals.values() if v is not None]
        if len(vs) < 2: continue
        best = max(vs) if col >= 7 else min(vs)
        for i in b:
            if vals[i] == best:
                raw = rows[i][col]; end = ' \\\\' if '\\\\' in raw else ' '
                rows[i][col] = ' \\textbf{' + raw.replace('\\\\', '').strip() + '}' + end
    for i in b:
        if 'cours' in L[i]:
            rows[i] = [rows[i][0]] + [' \\hl' + c if not c.strip().startswith('\\hl') else c for c in rows[i][1:]]
        L[i] = '&'.join(rows[i])
open(p, 'w').write('\n'.join(L))
