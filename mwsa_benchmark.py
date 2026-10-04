"""
Mongol Warfare Strategy Algorithm (MWSA) - standard benchmark suite
===================================================================

Update rule (Eq. 4 of the manuscript), for candidate i at iteration t:

    x_i(t+1) = x_i(t)
             + alpha * exp(-beta * tau) * (x_rand - x_i(t))      # feigned retreat
             + gamma * (1 - exp(-beta * tau)) * (x_best - x_i(t)) # encirclement


Usage
-----
    python mwsa_benchmark.py                      # full benchmark, 30 runs
    python mwsa_benchmark.py --quick              # fast smoke test
    python mwsa_benchmark.py --dim 30 --runs 30 --pop 30 --iters 500
    python mwsa_benchmark.py --funcs F1 F5 F9 --plot

Only numpy is required (matplotlib is optional, for --plot).
"""

import argparse
import time
import numpy as np


# ----------------------------------------------------------------------------
# Benchmark functions: (callable, lower bound, upper bound, known optimum f*)
# ----------------------------------------------------------------------------
def sphere(x):
    return np.sum(x ** 2)


def schwefel_222(x):
    a = np.abs(x)
    return np.sum(a) + np.prod(a)


def schwefel_12(x):
    return np.sum(np.cumsum(x) ** 2)


def schwefel_221(x):
    return np.max(np.abs(x))


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


def step(x):
    return np.sum(np.floor(x + 0.5) ** 2)


def rastrigin(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x) + 10.0)


def ackley(x):
    d = x.size
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20.0 + np.e)


def griewank(x):
    i = np.arange(1, x.size + 1)
    return np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0


def schwefel_226(x):
    # optimum is -418.9829 * D at x_i = 420.9687; returned shifted so f* = 0
    return 418.9829 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy(x):
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    t3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * w[-1]) ** 2)
    return t1 + t2 + t3


BENCHMARKS = {
    #  name   (function,       lb,      ub,     type)
    "F1":  (sphere,        -100.0,  100.0,  "unimodal"),
    "F2":  (schwefel_222,   -10.0,   10.0,  "unimodal"),
    "F3":  (schwefel_12,   -100.0,  100.0,  "unimodal"),
    "F4":  (schwefel_221,  -100.0,  100.0,  "unimodal"),
    "F5":  (rosenbrock,     -30.0,   30.0,  "unimodal"),
    "F6":  (step,          -100.0,  100.0,  "unimodal"),
    "F7":  (rastrigin,       -5.12,   5.12, "multimodal"),
    "F8":  (ackley,         -32.0,   32.0,  "multimodal"),
    "F9":  (griewank,      -600.0,  600.0,  "multimodal"),
    "F10": (schwefel_226,  -500.0,  500.0,  "multimodal"),
    "F11": (levy,           -10.0,   10.0,  "multimodal"),
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _evaluate(func, pop):
    return np.array([func(ind) for ind in pop])


def _init_population(rng, pop_size, dim, lb, ub):
    return lb + rng.random((pop_size, dim)) * (ub - lb)


# ----------------------------------------------------------------------------
# MWSA
# ----------------------------------------------------------------------------
def mwsa(func, dim, lb, ub, pop_size=30, max_iter=500, seed=None,
         alpha=0.5, gamma=0.9, beta=10.0, greedy=True, normalize_time=True):
    """
    Mongol Warfare Strategy Algorithm (minimisation).

    alpha  : feigned-retreat (exploration) weight
    gamma  : encirclement (exploitation) weight
    beta   : time-decay rate governing the tactical transition
    greedy : keep a candidate's new position only if it improves its fitness.
             Set False to reproduce the fluctuating per-iteration curve of the
             paper (the global best is still tracked either way).

    Returns (best_position, best_fitness, convergence_curve).
    """
    rng = np.random.default_rng(seed)
    pop = _init_population(rng, pop_size, dim, lb, ub)
    fit = _evaluate(func, pop)

    b = np.argmin(fit)
    x_best, f_best = pop[b].copy(), fit[b]
    curve = np.empty(max_iter)

    for t in range(max_iter):
        tau = t / max_iter if normalize_time else t
        retreat = alpha * np.exp(-beta * tau)            # strong early
        encircle = gamma * (1.0 - np.exp(-beta * tau))   # strong late

        x_rand = lb + rng.random((pop_size, dim)) * (ub - lb)
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        new_pop = (pop
                   + retreat * r1 * (x_rand - pop)
                   + encircle * r2 * (x_best - pop))
        new_pop = np.clip(new_pop, lb, ub)
        new_fit = _evaluate(func, new_pop)

        if greedy:
            improved = new_fit < fit
            pop[improved] = new_pop[improved]
            fit[improved] = new_fit[improved]
        else:
            pop, fit = new_pop, new_fit

        b = np.argmin(fit)
        if fit[b] < f_best:
            x_best, f_best = pop[b].copy(), fit[b]
        curve[t] = f_best

    return x_best, f_best, curve


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def pso(func, dim, lb, ub, pop_size=30, max_iter=500, seed=None,
        w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    """Standard global-best PSO with linearly decreasing inertia weight."""
    rng = np.random.default_rng(seed)
    pop = _init_population(rng, pop_size, dim, lb, ub)
    vmax = 0.2 * (ub - lb)
    vel = rng.uniform(-vmax, vmax, (pop_size, dim))
    fit = _evaluate(func, pop)

    p_best, p_fit = pop.copy(), fit.copy()
    g = np.argmin(p_fit)
    g_best, g_fit = p_best[g].copy(), p_fit[g]
    curve = np.empty(max_iter)

    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        r1, r2 = rng.random((pop_size, dim)), rng.random((pop_size, dim))
        vel = w * vel + c1 * r1 * (p_best - pop) + c2 * r2 * (g_best - pop)
        vel = np.clip(vel, -vmax, vmax)
        pop = np.clip(pop + vel, lb, ub)
        fit = _evaluate(func, pop)

        better = fit < p_fit
        p_best[better], p_fit[better] = pop[better], fit[better]
        g = np.argmin(p_fit)
        if p_fit[g] < g_fit:
            g_best, g_fit = p_best[g].copy(), p_fit[g]
        curve[t] = g_fit

    return g_best, g_fit, curve


def ga(func, dim, lb, ub, pop_size=30, max_iter=500, seed=None,
       pc=0.9, pm=0.05, elite=2, tournament=3):
    """Real-coded GA: tournament selection, arithmetic crossover, Gaussian
    mutation, elitism."""
    rng = np.random.default_rng(seed)
    pop = _init_population(rng, pop_size, dim, lb, ub)
    fit = _evaluate(func, pop)
    curve = np.empty(max_iter)
    sigma = 0.1 * (ub - lb)

    for t in range(max_iter):
        order = np.argsort(fit)
        elites = pop[order[:elite]].copy()
        elite_fit = fit[order[:elite]].copy()

        # tournament selection
        idx = rng.integers(0, pop_size, (pop_size, tournament))
        winners = idx[np.arange(pop_size), np.argmin(fit[idx], axis=1)]
        parents = pop[winners]

        # arithmetic crossover
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < pc:
                a = rng.random(dim)
                children[i] = a * parents[i] + (1 - a) * parents[i + 1]
                children[i + 1] = a * parents[i + 1] + (1 - a) * parents[i]

        # Gaussian mutation (sigma shrinks over time)
        mask = rng.random((pop_size, dim)) < pm
        scale = sigma * (1.0 - t / max_iter) + 1e-12
        children = np.clip(children + mask * rng.normal(0, scale, (pop_size, dim)), lb, ub)

        child_fit = _evaluate(func, children)
        worst = np.argsort(child_fit)[-elite:]
        children[worst], child_fit[worst] = elites, elite_fit
        pop, fit = children, child_fit
        curve[t] = fit.min()

    b = np.argmin(fit)
    return pop[b].copy(), fit[b], curve


ALGORITHMS = {"MWSA": mwsa, "PSO": pso, "GA": ga}


# ----------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------
def run_benchmark(funcs, algos, dim, runs, pop_size, max_iter, base_seed=0):
    results = {}
    for fname in funcs:
        func, lb, ub, ftype = BENCHMARKS[fname]
        results[fname] = {}
        for aname in algos:
            algo = ALGORITHMS[aname]
            finals, times, curves = [], [], []
            for r in range(runs):
                t0 = time.perf_counter()
                _, f_best, curve = algo(func, dim, lb, ub, pop_size=pop_size,
                                        max_iter=max_iter, seed=base_seed + r)
                times.append(time.perf_counter() - t0)
                finals.append(f_best)
                curves.append(curve)
            results[fname][aname] = {
                "finals": np.array(finals),
                "times": np.array(times),
                "curve": np.mean(curves, axis=0),
            }
            print(f"  {fname:<4} {aname:<5} mean={np.mean(finals):.4e}  "
                  f"std={np.std(finals):.4e}  time={np.mean(times):.2f}s", flush=True)
    return results


def print_table(results, funcs, algos):
    print("\n" + "=" * 98)
    print(f"{'Func':<6}{'Algo':<7}{'Best':>14}{'Worst':>14}{'Mean':>14}{'Std':>14}{'Time (s)':>11}")
    print("=" * 98)
    for fname in funcs:
        means = {a: np.mean(results[fname][a]["finals"]) for a in algos}
        winner = min(means, key=means.get)
        for a in algos:
            f = results[fname][a]["finals"]
            tag = " *" if a == winner else ""
            print(f"{fname:<6}{a:<7}{f.min():>14.4e}{f.max():>14.4e}"
                  f"{f.mean():>14.4e}{f.std():>14.4e}"
                  f"{results[fname][a]['times'].mean():>11.2f}{tag}")
        print("-" * 98)
    print("* = lowest mean fitness for that function")

    # average rank across functions
    ranks = {a: [] for a in algos}
    for fname in funcs:
        m = np.array([np.mean(results[fname][a]["finals"]) for a in algos])
        order = m.argsort().argsort() + 1
        for a, rk in zip(algos, order):
            ranks[a].append(rk)
    print("\nAverage rank (lower is better):")
    for a in sorted(algos, key=lambda k: np.mean(ranks[k])):
        print(f"  {a:<6}{np.mean(ranks[a]):.2f}")


def save_csv(results, funcs, algos, path):
    with open(path, "w") as fh:
        fh.write("function,algorithm,best,worst,mean,std,mean_time_s\n")
        for fname in funcs:
            for a in algos:
                f = results[fname][a]["finals"]
                fh.write(f"{fname},{a},{f.min():.6e},{f.max():.6e},"
                         f"{f.mean():.6e},{f.std():.6e},"
                         f"{results[fname][a]['times'].mean():.4f}\n")
    print(f"\nSaved summary table to {path}")


def plot_convergence(results, funcs, algos, path):
    import matplotlib.pyplot as plt
    n = len(funcs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, fname in zip(axes.ravel(), funcs):
        for a in algos:
            ax.semilogy(np.maximum(results[fname][a]["curve"], 1e-300), label=a)
        ax.set_title(fname)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved convergence plot to {path}")


def main():
    p = argparse.ArgumentParser(description="MWSA vs PSO vs GA on standard benchmarks")
    p.add_argument("--funcs", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    p.add_argument("--algos", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    p.add_argument("--dim", type=int, default=30)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default="mwsa_benchmark_results.csv")
    p.add_argument("--plot", action="store_true", help="save convergence curves (needs matplotlib)")
    p.add_argument("--quick", action="store_true", help="smoke test: 3 runs, 100 iterations")
    args = p.parse_args()

    if args.quick:
        args.runs, args.iters = 3, 100

    print(f"Benchmark: D={args.dim}, runs={args.runs}, pop={args.pop}, iters={args.iters}")
    results = run_benchmark(args.funcs, args.algos, args.dim, args.runs,
                            args.pop, args.iters, args.seed)
    print_table(results, args.funcs, args.algos)
    save_csv(results, args.funcs, args.algos, args.csv)
    if args.plot:
        plot_convergence(results, args.funcs, args.algos, "mwsa_convergence.png")


if __name__ == "__main__":
    main()
