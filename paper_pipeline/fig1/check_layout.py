"""Geometric verification of a rendered figure: every text box is measured from the PDF and
checked for overlap against every other text box and against the drawn rules, so collisions are
found by computation rather than by eye."""
import sys, itertools, pymupdf

PAD = 0.3          # pt of slack allowed before an overlap is reported
MIN_AREA = 0.8     # pt^2; ignore hairline touching


def boxes(page):
    out = []
    for blk in page.get_text('dict')['blocks']:
        if blk['type'] != 0:
            continue
        for ln in blk['lines']:
            for sp in ln['spans']:
                t = sp['text'].strip()
                if t:
                    out.append((pymupdf.Rect(sp['bbox']), t, round(sp['size'], 1)))
    return out


def inter_area(a, b):
    r = a & b
    return 0.0 if r.is_empty else r.get_area()


ACCENTS = set('\u02c6\u02dc\u00af\u02d9\u02da\u00b4\u0060\u02c7')


def strokes(page):
    """Every stroked segment as a thin rect, inflated by half its line width. A label whose box
    meets one of these is sitting on a drawn line."""
    out = []
    for d in page.get_drawings():
        w = d.get('width') or 0
        if d.get('type') not in ('s', 'fs') or w <= 0:
            continue
        g = w / 2 + 0.25
        for it in d['items']:
            if it[0] == 'l':
                r = pymupdf.Rect(it[1], it[2]).normalize()
            elif it[0] == 're':
                r = pymupdf.Rect(it[1]).normalize()
            elif it[0] == 'c':
                r = pymupdf.Rect(it[1], it[2]).normalize()
                r |= pymupdf.Point(it[3]); r |= pymupdf.Point(it[4])
            else:
                continue
            out.append(pymupdf.Rect(r.x0 - g, r.y0 - g, r.x1 + g, r.y1 + g))
    return out


def main(path):
    d = pymupdf.open(path)
    pg = d[0]
    B = boxes(pg)
    print(f'{path}: {len(B)} text spans, page {pg.rect.width:.1f} x {pg.rect.height:.1f} pt')
    bad = []
    for (ra, ta, sa), (rb, tb, sb) in itertools.combinations(B, 2):
        a = pymupdf.Rect(ra.x0 + PAD, ra.y0 + PAD, ra.x1 - PAD, ra.y1 - PAD)
        b = pymupdf.Rect(rb.x0 + PAD, rb.y0 + PAD, rb.x1 - PAD, rb.y1 - PAD)
        if ta in ACCENTS or tb in ACCENTS:
            continue
        ar = inter_area(a, b)
        if ar > MIN_AREA:
            bad.append((ar, ta, tb, tuple(round(v, 1) for v in ra), tuple(round(v, 1) for v in rb)))
    bad.sort(reverse=True)
    if bad:
        print(f'  {len(bad)} overlapping text pairs:')
        for ar, ta, tb, ra, rb in bad[:25]:
            print(f'   {ar:7.1f}pt^2  {ta[:28]!r:32} x {tb[:28]!r:32} {ra} {rb}')
    else:
        print('  no text-text overlaps')
    # text sitting on top of a drawn line
    S = strokes(pg)
    # a label drawn inside its own frame, or on an opaque filled badge, is not a collision
    SHIELD = []
    for d in pg.get_drawings():
        for it in d['items']:
            if it[0] == 're':
                SHIELD.append(pymupdf.Rect(it[1]).normalize())
        if d.get('type') in ('f', 'fs'):
            try:
                SHIELD.append(pymupdf.Rect(d['rect']).normalize())
            except Exception:
                pass
    hits = []
    for r, t, sz in B:
        rr = pymupdf.Rect(r.x0 + 0.5, r.y0 + 0.5, r.x1 - 0.5, r.y1 - 0.5)
        if rr.is_empty:
            continue
        if any(sh.contains(r) for sh in SHIELD):
            continue
        for q in S:
            if not (rr & q).is_empty:
                hits.append((t, tuple(round(v, 1) for v in r)))
                break
    print(f'  {len(hits)} spans crossing a drawn line ({len(S)} segments)'
          + (': ' + str(hits[:12]) if hits else ''))
    # text escaping the page box
    out = [(t, tuple(round(v, 1) for v in r)) for r, t, s in B
           if r.x0 < -0.5 or r.y0 < -0.5 or r.x1 > pg.rect.width + 0.5 or r.y1 > pg.rect.height + 0.5]
    print(f'  {len(out)} spans outside the canvas' + (': ' + str(out[:6]) if out else ''))
    return len(bad) + len(out)


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1]) == 0 else 1)
