"""Post-process the appendix comparison tables: within each block (rows between rules) bold the best value of every
column and tint the EqAlign rows; in the encoder-size table bold the best Top-1 of each row and tint the EqAlign
column. Idempotent; run after the generators and fix_minus.py (cwd: paper/)."""
import re
NUM = re.compile(r'^(?:\\ensuremath\{-\}|-)?\d+\.\d+$')
def num(s):
    s = s.replace('\\\\', '').strip()
    return float(s.replace('\\ensuremath{-}', '-')) if NUM.match(s) else None
def cellsplit(l):
    body = l.rstrip(); end = ''
    if body.endswith('\\\\'): body, end = body[:-2], ' \\\\'
    return body.split('&'), end
def wrap(c, f): return ' ' + f(c.strip()) + ' '
def rows_tab(p, dirs, ours=r'\\ours'):
    """dirs: {column index: +1 (higher is better) or -1}."""
    L = open(p).read().split('\n')
    if any('\\hl' in l for l in L): return
    blocks, cur = [], []
    for i, l in enumerate(L):
        if l.lstrip().startswith(('\\midrule', '\\bottomrule', '\\toprule', '\\cmidrule')):
            if cur: blocks.append(cur)
            cur = []
        elif '&' in l and l.rstrip().endswith('\\\\') and any(num(c) is not None for c in cellsplit(l)[0]): cur.append(i)
    if cur: blocks.append(cur)
    for b in blocks:
        R = {i: cellsplit(L[i]) for i in b}
        for col, d in dirs.items():
            v = {i: num(R[i][0][col]) for i in b if col < len(R[i][0])}
            vs = [x for x in v.values() if x is not None]
            if len(vs) < 2: continue
            best = max(vs) if d > 0 else min(vs)
            for i in b:
                if v.get(i) == best: R[i][0][col] = wrap(R[i][0][col], lambda s: '\\textbf{' + s + '}')
        for i in b:
            c, end = R[i]
            if re.search(ours, L[i]) and ' on $' not in L[i]:
                c = [c[0]] + [' \\hl' + x for x in c[1:]] if c[0].strip() == '' or 'multirow' in c[0] or len(c[0].strip()) <= 3 else [' \\hl ' + c[0].strip() + ' '] + [' \\hl' + x for x in c[1:]]
            L[i] = '&'.join(c).rstrip() + end
    open(p, 'w').write('\n'.join(L))
def scale_tab(p):
    L = open(p).read().split('\n')
    if any('\\hl' in l for l in L): return
    for i, l in enumerate(L):
        c, end = cellsplit(l)
        if len(c) == 8 and num(c[4]) is not None:
            v = [num(c[k]) for k in (4, 5, 6)]; k = 4 + v.index(max(v))
            c[k] = wrap(c[k], lambda s: '\\textbf{' + s + '}'); c[6] = ' \\hl' + c[6]
            L[i] = '&'.join(c).rstrip() + end
    open(p, 'w').write('\n'.join(L))
rows_tab('tab_app_family.tex', {2: -1, 3: -1, 4: -1, 5: +1, 6: +1})
rows_tab('tab_app_arch.tex', {2: -1, 3: -1, 4: -1, 5: +1, 6: +1, 7: +1})
rows_tab('tab_app_datascale.tex', {1: +1, 2: +1, 3: +1, 4: -1, 5: -1, 6: +1, 7: +1, 8: +1, 9: -1, 10: -1})
rows_tab('tab_app_abc.tex', {1: +1, 2: -1, 3: +1, 4: -1, 5: +1, 6: -1, 7: +1, 8: -1, 9: +1, 10: -1})
scale_tab('tab_app_scale.tex')
