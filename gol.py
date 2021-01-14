import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import correlate


def random_init(shape, p=0.2):
    """Populate the state randomly using probability p """
    return (np.random.random(shape) < p).astype(np.uint8)


def block_init(shape):
    """Initialize state with the block formation

    https://www.conwaylife.com/wiki/Block
    """
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i : i + 2, j : j + 2] = 1
    return state


def init_beehive(shape):
    """Initialize state with the beehive formation

    https://www.conwaylife.com/wiki/Beehive
    """
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i, j : j + 2] = 1
    state[i + 2, j : j + 2] = 1
    state[i + 1, j - 1] = 1
    state[i + 1, j + 2] = 1
    return state


def init_glider(shape):
    """Initialize state with the glider formation

    https://www.conwaylife.com/wiki/Glider
    """
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i - 1, j] = 1
    state[i, j + 1] = 1
    state[i + 1, j - 1 : j + 2] = 1
    return state


def init_pentadecathlon(shape):
    """Initialize state with the pentadecathlon formation

    https://www.conwaylife.com/wiki/Pentadecathlon
    """
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i - 1 : i + 2, j - 3 : j + 5] = 1
    state[i, j - 2] = 0
    state[i, j + 3] = 0
    return state


def init_acorn(shape):
    """Initialize state with the acorn formation

    https://www.conwaylife.com/wiki/Acorn
    """
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i, j] = 1
    state[i - 1, j - 2] = 1
    state[i + 1, j - 3 : j + 4] = 1
    state[i + 1, j - 1 : j + 1] = 0
    return state


def init_24827M(shape):
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i - 3, j + 2] = 1
    state[i - 2, j - 1] = 1
    state[i - 2, j + 1] = 1
    state[i - 1, j - 4] = 1
    state[i - 1, j + 1] = 1
    state[i, j - 3 : j - 1] = 1
    state[i + 1, j - 2] = 1
    state[i + 1, j] = 1
    state[i + 2, j - 1] = 1
    state[i + 3, j + 1 : j + 4] = 1
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


class MaxIterException(Exception):
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


def density_plot(sim):
    sns.set_style("whitegrid")
    plt.figure()
    plt.plot(sim.density)
    plt.title("Density Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Density")

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot(sim.density_tp1, sim.density_t)
    plt.title("Density(t+1) vs Density(t)")
    plt.xlabel("Density(t+1)")
    plt.ylabel("Density(t)")

    plt.show()


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
    return p


def main(args):
    n = args.grid_size
    shape = (n, n)
    p = args.prob
    state = random_init(shape, p=p)
    # state = init_block(shape)
    # state = init_beehive(shape)
    # state = init_glider(shape)
    # state = init_pentadecathlon(shape)
    # state = init_acorn(shape)
    # state = init_24827M(shape)
    sim = GOLSimulation(state)
    runner = None
    if args.headless:
        runner = SimulationRunner(sim, args.max_iter)
    else:
        runner = SimulationAnimation(sim, args.interval, args.max_iter)
    runner.run()
    # density_plot(sim)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
