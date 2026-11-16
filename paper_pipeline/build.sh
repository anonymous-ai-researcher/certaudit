# Main text on the GPU results (5 seeds, width 64): the same scripts as build.sh, run in the GPU view
# work_gpu (make_gpu_view.py), with the weights E1a selected on validation.
export LAMSTAR=0.2 TTLAM=0.03 CIRCLAM=0.2 L1LAM=0.0003 L1TTLAM=0.003 HYP_TTLAM=0.1 BILAM=0.1 TFLAM=0.1
G=work_gpu
cd work/paper
python3 $G/make_numbers2.py >/dev/null || exit 1
python3 $G/make_external.py >/dev/null || exit 1
python3 $G/make_general.py >/dev/null || exit 1
python3 work/highlight_general.py tab_general.tex || exit 1
(cd $G && python3 iscas_design_test.py >/dev/null) || exit 1
python3 $G/make_gpu.py >/dev/null || exit 1
python3 $G/make_pi_table.py >/dev/null || exit 1
python3 $G/make_appendix.py >/dev/null || exit 1
python3 work/make_arch_table.py >/dev/null
python3 work/make_scale_table.py >/dev/null
python3 work/make_decisions_tex.py >/dev/null || exit 1
python3 work/make_e4_tex.py >/dev/null || exit 1
python3 work/make_dup.py >/dev/null || exit 1
# E17 (certified ABC rewrites): needs the extracted server package; skipped with a notice if it is not on disk
E17_DIR=${E17_DIR:-${WORK}/dl_e17/x}
if [ -d "$E17_DIR/results/E17" ]; then (cd $G && python3 make_e17.py "$E17_DIR" >/dev/null 2>&1) || exit 1; else echo "E17 package not found: numbers_e17.tex and tab_app_abc.tex left as they are"; fi
python3 $G/fix_minus.py >/dev/null
python3 work/highlight_appendix.py || exit 1
# figures from the same GPU results: Fig. 2 (audit) and Fig. 3 (two levels; layout parameters for the 5-seed scatter)
(cd $G && AU_LY=1.15 AU_H=1.37 python3 make_figs2.py audit >/dev/null 2>&1) || exit 1
(cd $G && TL_H=1.815 TL_HEAD=0 TL_BASE=-5,-7.5 TL_CAP=5,-15,top python3 fig_twolevel.py > /tmp/fig_twolevel.log 2>&1 && grep -q "problems: 0" /tmp/fig_twolevel.log) || { echo "fig_twolevel layout check failed"; exit 1; }
# E14/E5 on GPU: Table II, Fig. 1, Fig. 4 (timing panel from the CPU run), recall@k (extended)
(cd $G && EQ_GRID=0.05,0.1,0.2,0.5,1,2 TT_GRID=0.03,0.1,0.5,1,2 C_GRID=0.05,0.2,0.5,1 python3 make_tables.py >/dev/null 2>&1) || exit 1
(cd $G/fig1cand/c && python3 gen_c.py >/dev/null 2>&1 && cp cA.pdf work/paper/figs/fig1c_cos.pdf) || exit 1
(cd $G/fig1cand && python3 gen_e.py >/dev/null 2>&1) || exit 1
(cd $G && SW_H=1.24 SWEEP_TIME_PREFIX=cpu_sweep2_v_ python3 make_figs2.py sweep retrieval >/dev/null 2>&1) || exit 1
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1; bibtex main >> build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
grep -E "^!|undefined|Overfull" build.log | sort | uniq -c | head -15; pdfinfo main.pdf 2>/dev/null | grep Pages
