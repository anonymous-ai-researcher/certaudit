"""Check every file of data/ against data/MANIFEST.json (size and MD5) and print what each file holds.
  python scripts/verify_data.py"""
import hashlib, json, os, pickle, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ok = True
for e in json.load(open(os.path.join(root, 'MANIFEST.json'))):
    p = os.path.join(root, e['file'])
    if not os.path.exists(p):
        print(f"[MISSING] {e['file']}"); ok = False; continue
    md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
    good = md5 == e['md5'] and os.path.getsize(p) == e['bytes']
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {e['file']:22s} {e['bytes'] / 1e6:7.1f} MB  {md5}")
if ok and '--summary' in sys.argv:
    D = pickle.load(open(os.path.join(root, 'data.pkl'), 'rb'))
    print(f"data.pkl: {len(D['train'])} training circuits, {len(D['test'])} test circuits, "
          f"{sum(len(v) for v in D['tvars'])} fixed test variants")
print('DATA OK' if ok else 'DATA CHECK FAILED')
sys.exit(0 if ok else 1)
