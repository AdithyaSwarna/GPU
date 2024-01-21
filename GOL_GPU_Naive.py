from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

img_plot = None
answer = []

@cuda.jit
def update_game_state_gpu(matrix, new_matrix):
    i, j = cuda.grid(2)

    if i < matrix.shape[0] and j < matrix.shape[1]:
        neighbors = (
            matrix[(i - 1) % matrix.shape[0], (j - 1) % matrix.shape[1]] +
            matrix[(i - 1) % matrix.shape[0], j] +
            matrix[(i - 1) % matrix.shape[0], (j + 1) % matrix.shape[1]] +
            matrix[i, (j - 1) % matrix.shape[1]] +
            matrix[i, (j + 1) % matrix.shape[1]] +
            matrix[(i + 1) % matrix.shape[0], (j - 1) % matrix.shape[1]] +
            matrix[(i + 1) % matrix.shape[0], j] +
            matrix[(i + 1) % matrix.shape[0], (j + 1) % matrix.shape[1]]
        )

        if((neighbors == 3) or ((neighbors == 2) and matrix[i, j])):
            new_matrix[i, j] = 1
        else:
            new_matrix[i, j] = 0

def conways_game_of_life_gpu(matrix, generations=50):
    matrix = np.array(matrix)
    new_matrix = np.zeros_like(matrix)
    global answer
    answer.append(np.copy(matrix))

    matrix_global_mem = cuda.to_device(matrix)
    new_matrix_global_mem = cuda.to_device(new_matrix)

    threads_per_block = (4, 4)
    blocks_per_grid_x = int(np.ceil(matrix.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(matrix.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for _ in range(generations):
        update_game_state_gpu[blocks_per_grid, threads_per_block](matrix_global_mem, new_matrix_global_mem)
        new_matrix_global_mem.copy_to_host(new_matrix)
        matrix_global_mem.copy_to_host(matrix)
        matrix_global_mem = cuda.to_device(new_matrix)
        answer.append(np.copy(new_matrix))

    return matrix

def period_60_glider_gun(size):
    grid = np.zeros((size, size), dtype=int)

    gun = [
        (0, 5), (0, 6), (1, 5), (1, 6),
        (10, 5), (10, 6), (10, 7), (11, 4), (11, 8), (12, 3), (12, 9),
        (13, 3), (13, 9), (14, 6), (15, 4), (15, 8), (16, 5), (16, 6), (16, 7),
        (17, 6), (20, 3), (20, 4), (20, 5), (21, 3), (21, 4), (21, 5),
        (22, 2), (22, 6), (24, 1), (24, 2), (24, 6), (24, 7),
        (34, 3), (34, 4), (35, 3), (35, 4)
    ]

    for i, j in gun:
        grid[i, j] = 1

    return grid


def update_plot(frame, matrices, img_plot):
    new_matrix = np.zeros_like(answer[frame])
    new_matrix[:] = answer[frame]
    img_plot.set_data(new_matrix)
    return [img_plot]

def plot_matrices_as_gif(matrices, generations, interval=100, output_file='GOL_GPU.gif'):
    fig, ax = plt.subplots(figsize=(15,15))
    img_plot = ax.imshow(matrices[0], cmap='binary', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    ani = animation.FuncAnimation(fig, update_plot, frames=generations,fargs=(matrices, img_plot), interval=interval, blit=True)
    plt.tight_layout()
    ani.save(output_file, writer='pillow')

start_time = time.time()

initial_state = period_60_glider_gun(100)
gen = 500
result = conways_game_of_life_gpu(initial_state, generations=gen)

end_time = time.time()
print("Time taken = ",end_time - start_time)

plot_matrices_as_gif(answer, generations=gen)

end_time = time.time()
print("Time taken after plotting= ",end_time - start_time)
