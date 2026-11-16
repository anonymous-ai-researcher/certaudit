"""SAT miter over shared primary inputs (from the paper's sweep.py; CaDiCaL 1.5.3 via PySAT)."""
from pysat.solvers import Cadical153


class Miter:
    """Both circuits Tseitin-encoded in one incremental solver over shared PI variables."""

    def __init__(self, ra, rb):
        self.s = Cadical153()
        self.na, self.nb = ra.n_nodes, rb.n_nodes
        nv = [0]

        def new():
            nv[0] += 1; return nv[0]
        const = new(); self.s.add_clause([-const])
        pis = [new() for _ in range(ra.n_pi)]
        self.va = self._enc(ra, const, pis, new)
        self.vb = self._enc(rb, const, pis, new)
        self.new = new

    def _enc(self, r, const, pis, new):
        v = [const] + pis
        base = 1 + r.n_pi
        for g, (a, b) in enumerate(r.fanins):
            x = v[int(a) >> 1] * (-1 if int(a) & 1 else 1)
            y = v[int(b) >> 1] * (-1 if int(b) & 1 else 1)
            o = new(); v.append(o)
            self.s.add_clause([-o, x]); self.s.add_clause([-o, y]); self.s.add_clause([o, -x, -y])
        return v

    def check(self, i, j, compl):
        """True if orig node i == (compl ? ~var node j : var node j). Returns (equiv, model or None)."""
        a, b = self.va[i], self.vb[j] * (-1 if compl else 1)
        t = self.new()
        # t -> (a xor b)
        self.s.add_clause([-t, a, b]); self.s.add_clause([-t, -a, -b])
        sat = self.s.solve(assumptions=[t])
        if sat:
            m = self.s.get_model()
            self.s.add_clause([-t])
            return False, m
        self.s.add_clause([-t])
        return True, None


