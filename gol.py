import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.ndimage import correlate


def random_init(shape, p=0.2):
    return (np.random.random(shape) < p).astype(np.uint8)


def block_init(shape):
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i : i + 2, j : j + 2] = 1
    return state


def init_beehive(shape):
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i, j : j + 2] = 1
    state[i + 2, j : j + 2] = 1
    state[i + 1, j - 1] = 1
    state[i + 1, j + 2] = 1
    return state


def init_glider(shape):
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i - 1, j] = 1
    state[i, j + 1] = 1
    state[i + 1, j - 1 : j + 2] = 1
    return state


def init_pentadecathlon(shape):
    state = np.zeros(shape, dtype=np.uint8)
    i = shape[0] // 2
    j = shape[1] // 2
    state[i - 1 : i + 2, j - 3 : j + 5] = 1
    state[i, j - 2] = 0
    state[i, j + 3] = 0
    return state


NEIGHBOR_KERN = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def step(state, count, next_state):
    next_state[:] = 0
    correlate(state, NEIGHBOR_KERN, output=count, mode="wrap")
    dead = state == 0
    living = state == 1
    count3 = count == 3
    # Rule 1
    rule = living & ((count == 2) | count3)
    next_state[rule] = 1
    # Rule 2
    rule = dead & count3
    next_state[rule] = 1
    return next_state


class GOLSimulation:
    def __init__(self, initial_state):
        self.state = initial_state
        self.count = np.zeros_like(self.state)
        self.next_state = np.zeros_like(self.state)

    def step(self):
        step(self.state, self.count, self.next_state)
        self.state, self.next_state = self.next_state, self.state
        return self.state


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
        state = self.sim.step()
        # Swap
        self.im.set_data(state)
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


def main():
    n = 400
    shape = (n, n)
    p = 0.1
    state = random_init(shape, p=p)
    # state = init_block(shape)
    # state = init_beehive(shape)
    # state = init_glider(shape)
    # state = init_pentadecathlon(shape)
    sim = GOLSimulation(state)
    animation = GOLAnimation(sim, 1)
    animation.run()


if __name__ == "__main__":
    main()