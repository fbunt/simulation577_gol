import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import correlate


def random_init(shape, p=0.2):
    """Populate the state randomly using probability p """
    return (np.random.random(shape) < p).astype(np.uint8)


# https://www.conwaylife.com/wiki/Glider
PATTERN_GLIDER = """
010
001
111
"""
# https://www.conwaylife.com/wiki/Pentadecathlon
PATTERN_PENTADECATHLON = """
11111111
10111101
11111111
"""
# https://www.conwaylife.com/wiki/Acorn
PATTERN_ACORN = """
0100000
0001000
1100111
"""
# https://www.conwaylife.com/wiki/24827M
PATTERN_24827M = """
00000010
00010100
10000100
01100000
00101000
00010000
00000111
"""
PATTERN_MAP = {
    "glider": PATTERN_GLIDER,
    "acorn": PATTERN_ACORN,
    "pent": PATTERN_PENTADECATHLON,
    "meth": PATTERN_24827M,
}


def pattern_to_array(pat):
    pat = pat.strip()
    arr = []
    for line in pat.split():
        row = []
        for c in line:
            row.append(int(c))
        arr.append(row)
    arr = np.array(arr, dtype=np.uint8)
    return arr


def init_from_pattern(shape, pat):
    state = np.zeros(shape, dtype=np.uint8)
    pat = pattern_to_array(pat)
    i, j = [x // 2 for x in shape]
    n, m = pat.shape
    istart = i - (n // 2)
    jstart = j - (m // 2)
    state[istart : istart + n, jstart : jstart + m] = pat
    return state


def density_calc(state):
    return state.sum() / state.size


NEIGHBOR_KERN = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


class GOLSimulation:
    def __init__(self, initial_state):
        self.state = initial_state
        self.count = np.zeros_like(self.state)
        self.next_state = np.zeros_like(self.state)
        self.density = [initial_state.sum() / initial_state.size]
        self.density_t = []
        self.density_tp1 = []

    def step(self):
        try:
            self._step()
            self.density_t.append(density_calc(self.state))
            self.density_tp1.append(density_calc(self.next_state))
            # Swap to update current state
            self.state, self.next_state = self.next_state, self.state
            self.density.append(density_calc(self.state))
        except KeyboardInterrupt as e:
            # Trim in case the simulation was stopped between appends
            if len(self.sim.density_t) > len(self.sim.density_tp1):
                self.sim.density_t.pop()
            elif len(self.sim.density_tp1) > len(self.sim.density_t):
                self.sim.density_tp1.pop()
            raise e
        return self.state

    def _step(self):
        self.next_state[:] = 0
        correlate(self.state, NEIGHBOR_KERN, output=self.count, mode="wrap")
        dead = self.state == 0
        living = self.state == 1
        count3 = self.count == 3
        # Rules 2 & 3
        rule = living & ((self.count == 2) | count3)
        self.next_state[rule] = 1
        # Rule 4
        rule = dead & count3
        self.next_state[rule] = 1


class SimulationRunner:
    def __init__(self, sim, max_iter):
        self.sim = sim
        self.max_iter = max_iter

    def run(self):
        try:
            i = 1
            while i < self.max_iter:
                self.sim.step()
                i += 1
        except KeyboardInterrupt:
            pass


class SimulationAnimation:
    def __init__(self, sim, interval, max_iter):
        self.sim = sim
        self.fig = plt.figure()
        self.im = None
        self.ani = None
        self.interval = interval
        self.max_iter = max_iter
        self.iter = 1
        self.paused = False

    def init(self):
        self.im = plt.imshow(
            self.sim.state, interpolation="none", animated=True, cmap="gray"
        )
        return (self.im,)

    def update(self, *args):
        self.sim.step()
        self.im.set_data(self.sim.state)
        self.iter += 1
        if self.iter > self.max_iter:
            self.ani.event_source.stop()
            plt.close()
        return (self.im,)

    def on_click(self, event):
        if event.key != " ":
            return
        if self.paused:
            self.ani.event_source.start()
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.paused = True

    def run(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_click)
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            interval=self.interval,
            blit=True,
        )
        plt.show()


def density_vs_plot(sim):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot(sim.density_tp1, sim.density_t)
    plt.title("Density(t+1) vs Density(t)")
    plt.xlabel("Density(t+1)")
    plt.ylabel("Density(t)")
    plt.show()


def density_over_time_plot(sim):
    sns.set_style("whitegrid")
    plt.figure()
    plt.plot(sim.density)
    plt.title("Density Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Density")
    plt.show()


def validate_pattern(pat):
    if pat in PATTERN_MAP:
        return pat
    raise KeyError("Invalid pattern name. Use -h/--help to view valid options")


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-p",
        "--prob",
        type=float,
        default=0.5,
        help="Probability of cells starting alive. Default: 0.5",
    )
    p.add_argument(
        "-n",
        "--grid_size",
        type=int,
        default=400,
        help="Grid size. Default: 400",
    )
    p.add_argument(
        "-i",
        "--interval",
        type=int,
        default=1,
        help="Minimum animation frame interval in ms. Default: 1ms",
    )
    p.add_argument(
        "-s",
        "--headless",
        action="store_true",
        help="Run simulation without animation",
    )
    p.add_argument(
        "-m",
        "--max_iter",
        type=float,
        default=np.inf,
        help="Max number of iterations. Default: inf",
    )
    opts = f"{[k for k in PATTERN_MAP]}".strip("[]")
    p.add_argument(
        "--pat",
        type=validate_pattern,
        default=None,
        help=f"Starting pattern to use instead of random. Options are: {opts}",
    )
    return p


def run(
    grid_size=400,
    prob=0.5,
    pat=None,
    max_iter=np.inf,
    interval=1,
    headless=True,
):
    n = grid_size
    shape = (n, n)
    if pat is not None:
        state = init_from_pattern(shape, PATTERN_MAP[pat])
    else:
        state = random_init(shape, p=prob)
    sim = GOLSimulation(state)
    runner = None
    if headless:
        runner = SimulationRunner(sim, max_iter)
    else:
        runner = SimulationAnimation(sim, interval, max_iter)
    runner.run()
    return sim


def main(args):
    sim = run(**vars(args))
    density_over_time_plot(sim)
    density_vs_plot(sim)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
