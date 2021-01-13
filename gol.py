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


NEIGHBOR_KERN = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


class GOLSimulation:
    def __init__(self, initial_state):
        self.state = initial_state
        self.count = np.zeros_like(self.state)
        self.next_state = np.zeros_like(self.state)
        self.density = [initial_state.sum() / initial_state.size]

    def step(self):
        self._step()
        self.state, self.next_state = self.next_state, self.state
        self.density.append(self.state.sum() / self.state.size)
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


class GOLAnimation:
    def __init__(self, sim, interval):
        self.sim = sim
        self.fig = plt.figure()
        self.im = None
        self.ani = None
        self.interval = interval

    def init(self):
        self.im = plt.imshow(
            self.sim.state, interpolation="none", animated=True, cmap="gray"
        )
        return (self.im,)

    def update(self, *args):
        self.sim.step()
        self.im.set_data(self.sim.state)
        return (self.im,)

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            interval=self.interval,
            blit=True,
        )
        plt.show()


def density_plot(sim):
    sns.set_theme()
    plt.figure()
    plt.plot(sim.density)
    plt.title("Density Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Density")
    sns.despine()
    plt.show()


def main():
    n = 400
    shape = (n, n)
    p = 0.1
    state = random_init(shape, p=p)
    # state = init_block(shape)
    # state = init_beehive(shape)
    # state = init_glider(shape)
    # state = init_pentadecathlon(shape)
    # state = init_acorn(shape)
    # state = init_24827M(shape)
    sim = GOLSimulation(state)
    animation = GOLAnimation(sim, 1)
    animation.run()
    density_plot(sim)


if __name__ == "__main__":
    main()
