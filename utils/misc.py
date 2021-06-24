import torch
import numpy as np


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


# combine block batches to whole images
def combine(data, shape, block_size=72):
    union_data = np.zeros(shape, dtype=np.float32)

    for i in range(shape[0] // block_size + 1):
        for j in range(shape[1] // block_size + 1):
            for k in range(shape[2] // block_size + 1):
                if i == shape[0] // block_size:
                    x = shape[0] - block_size // 2
                else:
                    x = i * block_size + block_size // 2

                if j == shape[1] // block_size:
                    y = shape[1] - block_size // 2
                else:
                    y = j * block_size + block_size // 2

                if k == shape[2] // block_size:
                    z = shape[2] - block_size // 2
                else:
                    z = k * block_size + block_size // 2

                union_data[x - block_size // 2: x + block_size // 2,
                y - block_size // 2: y + block_size // 2,
                z - block_size // 2: z + block_size // 2] = data[
                    i * (shape[1] // block_size + 1) * (shape[2] // block_size + 1) + j * (
                                shape[2] // block_size + 1) + k]

    return union_data