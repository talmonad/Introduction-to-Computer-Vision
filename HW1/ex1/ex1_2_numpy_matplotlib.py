# %% [markdown]
# # EX 1.2- Numpy & Matplotlib: Build pyramid
# Correct the mistakes below to reveal the pyramid drawing
# This ex. is meant to enhance your debugging skills.
# %%

import matplotlib.pyplot as plt
import numpy as np

# %%


def half_pyramid(pyramid_size):
    # building one half of the pyramid
    # THIS FUNCTION IS CORRECT! DO NOT TOUCH
    # CLUE: print this half pyramid in a plot. The answer would be similar up to mirroring
    y = np.concatenate((np.zeros(pyramid_size), np.arange(pyramid_size))).reshape(1, -1)
    x = np.arange(y.shape[1]).reshape(1, -1)
    print(f"x={x}, y={y}")
    return np.concatenate((x, y), axis=0)


# %%


def build_pyramid(pyramid_size):
    xy_first_half = half_pyramid(pyramid_size)
    #plt.plot(xy_first_half[0, :], xy_first_half[1, :])
    #plt.show()
    # TODO: in the 3 lines of code below 3 mistakes were made.
    #  correct the mistakes to reveal the correct plot.
    #  try to keep as much from the original code as-is, so don't re-write the entire 3 lines
    xy_other_half = np.array([xy_first_half[0, :], xy_first_half[1, :][::-1]])
    xy_full = np.concatenate((xy_first_half, xy_other_half), axis=1)
    plt.plot(xy_full[0, :], xy_full[1, :], marker='o')


# %%
pyramid_sizes = [10, 50, 90, 140]

plt.figure()
for sz in pyramid_sizes:
    build_pyramid(sz)
plt.title("my pyramids")
plt.show()


# %%
