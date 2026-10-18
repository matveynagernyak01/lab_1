"""Решения задач на NumPy."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    return np.matmul(matrices, vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_statistics_histograms(stats: MatrixStatistics) -> None:
    """Гистограммы значений по строкам и по столбцам (matplotlib.pyplot.hist)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in stats.matrix:
        axes[0].hist(row, bins=20, alpha=0.4)
    axes[0].set_title("Гистограммы по строкам")
    for column in stats.matrix.T:
        axes[1].hist(column, bins=20, alpha=0.4)
    axes[1].set_title("Гистограммы по столбцам")
    fig.tight_layout()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    bottom = min(top + height, image_height)
    right = min(left + width, image_width)
    image[top:bottom, left:right] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    y, x = np.indices((image_height, image_width))
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2
    inside = (
        (x - x0) ** 2 * semi_axis_y ** 2 + (y - y0) ** 2 * semi_axis_x ** 2
        <= (semi_axis_x * semi_axis_y) ** 2
    )
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("Размер окна должен быть положительным")

    middle = values[1:-1]
    left, right = values[:-2], values[2:]
    maxima = np.flatnonzero((middle > left) & (middle > right)) + 1
    minima = np.flatnonzero((middle < left) & (middle < right)) + 1

    if window > len(values):
        moving_average = np.empty(0)
    else:
        moving_average = sliding_window_view(values, window).mean(axis=1)

    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if labels.size and labels.max() >= class_count:
        raise ValueError("Метка класса больше или равна class_count")
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result