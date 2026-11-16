"""Typeset negative numbers in generated tex with a true minus sign (idempotent)."""
import re, glob, os
os.chdir("work")
for f in ['paper/numbers.tex'] + glob.glob('paper/tab_*.tex'):
    s = open(f).read()
    s0 = s
    s = re.sub(r'\\ours (?=[A-Za-z])', r'\\ours\\ ', s)
    t = re.sub(r'(?<=[\s{\[(&$,])-(?=\d)', r'\\ensuremath{-}', s)
    if t != s0:
        open(f, 'w').write(t); print('fixed', f, s.count('-') - t.count('-') + t.count('\\ensuremath{-}') )
