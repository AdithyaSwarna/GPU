import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import numpy as np
import time

matrix, matrix_size, img_plot = None, None, None

def initialize(size):
    #matrix = np.random.choice([0, 1], size * size, p=[0.8, 0.2]).reshape(size, size)
    matrix = np.zeros((size, size), dtype=int)

    gun = [
        (0, 5), (0, 6), (1, 5), (1, 6),
        (10, 5), (10, 6), (10, 7), (11, 4), (11, 8), (12, 3), (12, 9),
        (13, 3), (13, 9), (14, 6), (15, 4), (15, 8), (16, 5), (16, 6), (16, 7),
        (17, 6), (20, 3), (20, 4), (20, 5), (21, 3), (21, 4), (21, 5),
        (22, 2), (22, 6), (24, 1), (24, 2), (24, 6), (24, 7),
        (34, 3), (34, 4), (35, 3), (35, 4)
    ]

    for i, j in gun:
        matrix[i, j] = 1
        
    return matrix

def update_step(frame):
    global matrix, matrix_size, img_plot
    new_matrix = np.zeros_like(matrix)
    rows,cols = matrix.shape
    for i in range(matrix_size):
        for j in range(matrix_size):
            neighbors = (matrix[(i - 1) % matrix.shape[0], (j - 1) % matrix.shape[1]] +
            matrix[(i - 1) % matrix.shape[0], j] +
            matrix[(i - 1) % matrix.shape[0], (j + 1) % matrix.shape[1]] +
            matrix[i, (j - 1) % matrix.shape[1]] +
            matrix[i, (j + 1) % matrix.shape[1]] +
            matrix[(i + 1) % matrix.shape[0], (j - 1) % matrix.shape[1]] +
            matrix[(i + 1) % matrix.shape[0], j] +
            matrix[(i + 1) % matrix.shape[0], (j + 1) % matrix.shape[1]])
            if matrix[i, j] == 1 and (neighbors in (2,3)):
                new_matrix[i, j] = 1
            elif matrix[i, j] == 0 and neighbors == 3:
                new_matrix[i, j] = 1
            else:
                new_matrix[i, j] = 0
    matrix = new_matrix
    img_plot.set_data(new_matrix)
    return img_plot

def game_of_life(random=True, size=100):
    global matrix, matrix_size, img_plot
    matrix_size = size
    matrix = initialize(size)
    fig, ax = plt.subplots(figsize=(10, 10))
    img_plot = ax.imshow(matrix, interpolation='nearest', cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    ani = animation.FuncAnimation(fig, frames=500, func=update_step, interval=100)
    plt.tight_layout()
    ani.save('GOL_SERIAL.gif')
    #plt.show()
    return ani


start_time = time.time()
game_of_life(size=int(input("Enter matrix size: ")))
end_time = time.time()
print("Time taken = ",end_time - start_time)
