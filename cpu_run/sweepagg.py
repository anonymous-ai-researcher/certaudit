"""One aggregation for the SAT-sweeping probe, shared by the text (make_numbers2) and Fig. 4
(make_figs2), so the two can never disagree again.

For each seed (one sweep file per checkpoint) the reduction is pooled over all design-variant
instances, i.e. total work saved relative to topological order:
    red = 100 * (1 - sum(policy) / sum(topological))
then reported as the mean and standard error over seeds."""
import json
import numpy as np


def _pooled(rows, pol, bits, field, add_emb):
    num = den = 0.0
    for key in sorted(set((x['circ'], x['kind']) for x in rows)):
        sel = [x for x in rows if (x['circ'], x['kind']) == key and x['bits'] == bits]
        topo = [x for x in sel if x['policy'] == 'topo']
        vv = [x for x in sel if x['policy'] == pol]
        if not topo or not vv:
            continue
        den += np.mean([x[field] for x in topo])
        num += np.mean([x[field] + (x['emb_time'] if add_emb else 0.0) for x in vv])
    return 100.0 * (1.0 - num / den) if den > 0 else float('nan')


def reduction(files, pol, bits, field='wasted', add_emb=False):
    """Mean and s.e. over seeds of the pooled reduction; one file per seed."""
    per_seed = [_pooled(json.load(open(f)), pol, bits, field, add_emb) for f in files]
    per_seed = [v for v in per_seed if v == v]
    mu = float(np.mean(per_seed))
    se = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
    return mu, se
