"""Fig. 1 panel C candidates: who sits next to u in embedding space? Data: fig1c_sims.json (exact cosines of
every AND gate of G to u, base_s0 and eqalign_s0). Candidates: ridge (A) and bullseye (B)."""
import os, sys, json, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '../..')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figstyle import C, INK, INK2, MUTED, GRID
D = json.load(open('../../fig1c_sims.json'))
W, H = 2.50, 1.50
from figstyle import OURS_DARK as OURS_INK
S = {k: np.array(v['sims']) for k, v in D.items()}
T = {k: v['idx_true'] for k, v in D.items()}
plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.5, 'mathtext.fontset': 'stix'})


def kde(x, grid, bw):
    return np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1)


def cand_ridge():
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0.03, 0.02, 0.94, 0.96])
    XL = -0.42
    ax.set_xlim(XL, 1.07); ax.set_ylim(-0.62, 2.12); ax.axis('off')
    g = np.linspace(XL, 1.0, 600)
    for k, y0, name in [('base', 1.12, 'Baseline'), ('ours', 0.0, 'EqAlign')]:
        s, t = S[k], T[k]; col = C[k]
        d = kde(s, g, 0.012); d = 0.55 * d / d.max()
        ax.fill_between(g, y0, y0 + d, color=col, alpha=0.16, lw=0)
        ax.plot(g, y0 + d, color=col, lw=0.9)
        ax.plot([XL, 1.0], [y0, y0], color=MUTED, lw=0.5)
        ax.vlines(s, y0 - 0.07, y0, color=col, lw=0.35, alpha=0.6)
        closer = (s > s[t]).sum()
        ax.axvspan(s[t], 1.0, ymin=0, ymax=1, alpha=0)  # placeholder (keeps limits)
        ax.fill_between([s[t], 1.0], y0, y0 + 0.62, color=col if k == 'ours' else INK2, alpha=0.07, lw=0)
        ax.plot(1.0, y0, marker='*', ms=7.5, color=INK, mec='white', mew=0.5, zorder=6, clip_on=False)
        ax.plot(s[t], y0, marker='o', ms=5.6, mfc='white', mec=col, mew=1.3, zorder=6)
        ax.plot(s[t], y0, marker='o', ms=2.2, color=col, zorder=7)
        txt = f'{closer} gates nearer than $v$' if closer else '$v$ is the nearest'
        ax.text(1.0, y0 + 0.66, txt, ha='right', va='bottom', fontsize=6.5,
                color=OURS_INK if k == 'ours' else INK, fontweight='bold')
        ax.text(XL, y0 + 0.10, name, ha='left', va='bottom', fontsize=6.8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.2', fc=col, ec='none'))
    for x in (0, 0.5, 1.0):
        ax.text(x, -0.11, f'{x:g}', ha='center', va='top', fontsize=6.5, color=INK2)
    ax.text(XL, -0.11, 'cosine to $u$', ha='left', va='top', fontsize=6.5, color=INK2)
    # legend line
    yl = -0.50
    ax.plot(XL + 0.03, yl, marker='*', ms=7, color=INK, mec='white', mew=0.5, clip_on=False)
    ax.text(XL + 0.08, yl, 'rewritten gate $u$', ha='left', va='center', fontsize=6.5, color=INK2)
    ax.plot(0.47, yl, marker='o', ms=5, mfc='white', mec=INK2, mew=1.1)
    ax.text(0.52, yl, 'proven equivalent $v$', ha='left', va='center', fontsize=6.5, color=INK2)
    fig.savefig('cA.pdf'); fig.savefig('cA.png', dpi=400); plt.close(fig)


def cand_bull():
    fig = plt.figure(figsize=(W, H))
    R = 1.32
    for j, (k, name) in enumerate([('base', 'Baseline'), ('ours', 'EqAlign')]):
        ax = fig.add_axes([0.02 + j * 0.50, 0.04, 0.46, 0.78]); ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        s, t = S[k], T[k]; col = C[k]; P = np.array(D[k]['pca'])[:-1]
        ang = np.arctan2(P[:, 1], P[:, 0]); r = 1 - s
        for rr, lab in ((0.5, '0.5'), (1.0, '0')):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, ec=GRID, lw=0.6))
        ax.add_patch(plt.Circle((0, 0), r[t], fc=col, alpha=0.10, ec=col, lw=0.7, ls=(0, (2, 1.4))))
        m = np.ones(len(s), bool); m[t] = False
        ax.scatter(r[m] * np.cos(ang[m]), r[m] * np.sin(ang[m]), s=3.2, color=MUTED, lw=0, alpha=0.8, zorder=3)
        vx, vy = r[t] * np.cos(ang[t]), r[t] * np.sin(ang[t])
        ax.plot(vx, vy, marker='o', ms=5.4, mfc='white', mec=col, mew=1.3, zorder=6)
        ax.plot(vx, vy, marker='o', ms=2.1, color=col, zorder=7)
        ax.plot(0, 0, marker='*', ms=7.5, color=INK, mec='white', mew=0.5, zorder=8)
        closer = (s > s[t]).sum()
        ax.text(0, R + 0.05, name, ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.2', fc=col, ec='none'))
        ax.text(0, -R + 0.02, f'rank {closer + 1}', ha='center', va='bottom', fontsize=7,
                color=OURS_INK if k == 'ours' else INK, fontweight='bold')
    fig.savefig('cB.pdf'); fig.savefig('cB.png', dpi=400); plt.close(fig)


cand_ridge(); cand_bull()
