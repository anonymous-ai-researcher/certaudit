"""Run job files on all local GPUs, several jobs per GPU, resumable. One process per job; a job that fails is
retried once and then listed in jobs/FAILED.txt (the others keep running).

  python -m tools.run_queue jobs/E1_lr.jsonl [more.jsonl ...] --gpus 0,1,2,3,4,5,6,7 --per-gpu 3
  --gpus cpu runs on the CPU (smoke test without a GPU)."""
import argparse, json, os, subprocess, sys, time
from tools.run_job import is_done

ap = argparse.ArgumentParser()
ap.add_argument('jobs', nargs='+'); ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
ap.add_argument('--per-gpu', type=int, default=3); ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args()
J = [json.loads(l) for f in a.jobs for l in open(f) if l.strip()]
todo = [j for j in J if not is_done(j)]
print(f'{len(J)} jobs, {len(J) - len(todo)} already done, {len(todo)} to run', flush=True)
slots = [g for g in a.gpus.split(',') for _ in range(a.per_gpu)]
running, tries, failed, t0, ndone = {}, {}, [], time.time(), 0
os.makedirs('jobs', exist_ok=True)
ready = lambda j: all(os.path.exists(os.path.join(d, 'model.pt')) and os.path.exists(os.path.join(d, 'train.json')) for d in j.get('deps', [])) \
    and all(os.path.exists(f) for f in j.get('deps_file', []))
while todo or running:
    for slot in [s for s in range(len(slots)) if s not in running]:
        cand = [j for j in todo if ready(j)]
        if not cand:
            if not running and todo:                 # nothing can ever make these ready
                for j in todo:
                    print(f'FAILED {j["id"]}: dependencies never finished: {j.get("deps")}', flush=True); failed.append(j['id'])
                todo = []
            break
        j = cand[0]; todo.remove(j)
        env = dict(os.environ, THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
        if slots[slot] == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            env['CUDA_VISIBLE_DEVICES'] = slots[slot]
        errf = open(os.path.join('jobs', f'.err{os.environ.get("QUEUE_TAG", "")}_{slot}.txt'), 'w')
        p = subprocess.Popen([sys.executable, '-m', 'tools.run_job', json.dumps(j)], env=env,
                             stdout=subprocess.DEVNULL, stderr=errf, text=True)
        p.errpath = errf.name
        running[slot] = (p, j)
    time.sleep(2)
    for slot, (p, j) in list(running.items()):
        if p.poll() is None:
            continue
        del running[slot]
        if p.returncode == 0 and is_done(j):
            ndone += 1
        else:
            tries[j['id']] = tries.get(j['id'], 0) + 1
            err = open(p.errpath).read().strip().splitlines()[-1:]
            if tries[j['id']] < 2:
                print(f'retrying {j["id"]}: {err}', flush=True); todo.append(j)
            else:
                print(f'FAILED {j["id"]}: {err}', flush=True); failed.append(j['id'])
        el = time.time() - t0
        print(f'[{time.strftime("%H:%M:%S")}] done {ndone}, failed {len(failed)}, running {len(running)}, '
              f'queued {len(todo)}, elapsed {el / 60:.1f} min', flush=True)
if failed:
    with open(f'jobs/FAILED{os.environ.get("QUEUE_TAG", "")}.txt', 'a') as f:
        f.write('\n'.join(failed) + '\n')
    print(f'{len(failed)} jobs failed, listed in jobs/FAILED{os.environ.get("QUEUE_TAG", "")}.txt; see <out>/job.log of each', flush=True)
print('queue finished', flush=True)
sys.exit(1 if failed else 0)                       # non-zero: the README's '&&' chain stops here
