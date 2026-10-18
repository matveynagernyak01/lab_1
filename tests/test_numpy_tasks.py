import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, sum_prod, unique_columns, unique_rows,
)


def test_sum_prod():
    rng = np.random.default_rng(0)
    p, n = 4, 3
    matrices = rng.integers(-5, 5, size=(p, n, n))
    vectors = rng.integers(-5, 5, size=(p, n, 1))
    expected = sum(matrices[i] @ vectors[i] for i in range(p))
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert result.shape == (n, 1)
    assert np.array_equal(result, expected)


def test_sum_prod_single_and_scalar():
    result = sum_prod(MatrixVectorBatchInput(matrices=np.array([[[2]]]), vectors=np.array([[[3]]])))
    assert np.array_equal(result, [[6]])
    eye = np.stack([np.eye(2)] * 3)
    ones = np.ones((3, 2, 1))
    assert np.array_equal(sum_prod(MatrixVectorBatchInput(matrices=eye, vectors=ones)), [[3], [3]])


def test_binarize():
    m = np.array([[1, 5, 3], [3, 0, -1]])
    result = binarize(BinarizeInput(matrix=m, threshold=3))
    assert np.array_equal(result, [[0, 1, 0], [0, 0, 0]])
    assert np.array_equal(m, [[1, 5, 3], [3, 0, -1]])
    assert result is not m


def test_binarize_edge():
    assert np.array_equal(binarize(BinarizeInput(matrix=np.array([[0.5]]), threshold=0.5)), [[0]])
    assert np.array_equal(binarize(BinarizeInput(matrix=np.array([[-1, -2]]), threshold=-3)), [[1, 1]])


def test_unique_rows_and_columns():
    m = np.array([[1, 1, 2], [3, 3, 3], [4, 5, 4]])
    assert unique_rows(MatrixInput(matrix=m)) == [[1, 2], [3], [4, 5]]
    assert unique_columns(MatrixInput(matrix=m)) == [[1, 3, 4], [1, 3, 5], [2, 3, 4]]


def test_unique_single_element():
    assert unique_rows(MatrixInput(matrix=np.array([[7]]))) == [[7]]
    assert unique_columns(MatrixInput(matrix=np.array([[1, 2, 1]]))) == [[1], [2], [1]]


def test_matrix_statistics():
    data = RandomMatrixInput(rows=5, columns=7, mean=3.0, std=2.0, seed=42)
    s = matrix_statistics(data)
    assert s.matrix.shape == (5, 7)
    assert np.allclose(s.row_means, s.matrix.mean(axis=1))
    assert np.allclose(s.column_means, s.matrix.mean(axis=0))
    assert np.allclose(s.row_variances, s.matrix.var(axis=1))
    assert np.allclose(s.column_variances, s.matrix.var(axis=0))
    assert s.row_means.shape == (5,) and s.column_means.shape == (7,)
    assert np.array_equal(matrix_statistics(data).matrix, s.matrix)
    other = matrix_statistics(RandomMatrixInput(rows=5, columns=7, mean=3.0, std=2.0, seed=1))
    assert not np.array_equal(other.matrix, s.matrix)


def test_matrix_statistics_distribution():
    s = matrix_statistics(RandomMatrixInput(rows=300, columns=300, mean=5.0, std=2.0, seed=0))
    assert s.matrix.mean() == pytest.approx(5.0, abs=0.05)
    assert s.matrix.std() == pytest.approx(2.0, abs=0.05)


def test_chess():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    assert np.array_equal(result, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(chess(ChessInput(rows=1, columns=1, first=7, second=9)), [[7]])
    assert np.array_equal(chess(ChessInput(rows=2, columns=1, first=1, second=2)), [[1], [2]])
    assert chess(ChessInput(rows=4, columns=6, first=2, second=3)).shape == (4, 6)


def test_draw_rectangle():
    img = draw_rectangle(RectangleInput(
        width=2, height=2, image_height=4, image_width=4,
        shape_color=(255, 0, 0), background_color=(0, 0, 0)))
    assert img.shape == (4, 4, 3)
    expected = np.zeros((4, 4, 3), dtype=np.uint8)
    expected[1:3, 1:3] = (255, 0, 0)
    assert np.array_equal(img, expected)


def test_draw_rectangle_non_square_and_full():
    img = draw_rectangle(RectangleInput(
        width=3, height=1, image_height=3, image_width=5,
        shape_color=(1, 2, 3), background_color=(9, 9, 9)))
    assert img.shape == (3, 5, 3)
    assert (img[1, 1:4] == (1, 2, 3)).all()
    assert (img == (1, 2, 3)).all(axis=2).sum() == 3
    full = draw_rectangle(RectangleInput(
        width=4, height=4, image_height=4, image_width=4,
        shape_color=(1, 1, 1), background_color=(0, 0, 0)))
    assert (full == 1).all()


def test_draw_ellipse():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=1, image_height=3, image_width=5,
        shape_color=(255, 255, 255), background_color=(0, 0, 0)))
    assert img.shape == (3, 5, 3)
    mask = (img == 255).all(axis=2)
    assert np.array_equal(mask, [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]])


def test_draw_ellipse_circle_symmetry():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=5, semi_axis_y=5, image_height=11, image_width=11,
        shape_color=(1, 2, 3), background_color=(0, 0, 0)))
    mask = (img == (1, 2, 3)).all(axis=2)
    assert mask[5, 5] and not mask[0, 0]
    assert np.array_equal(mask, mask.T) and np.array_equal(mask, mask[::-1, ::-1])


def test_analyze_time_series():
    values = np.array([1, 3, 2, 5, 4, 4, 6, 1], dtype=float)
    s = analyze_time_series(TimeSeriesInput(values=values, window=3))
    assert s.mean == pytest.approx(values.mean())
    assert s.variance == pytest.approx(values.var())
    assert s.std == pytest.approx(values.std())
    assert list(s.local_maxima_indices) == [1, 3, 6]
    assert list(s.local_minima_indices) == [2]
    assert len(s.moving_average) == len(values) - 3 + 1
    assert np.allclose(s.moving_average, [2, 10 / 3, 11 / 3, 13 / 3, 14 / 3, 11 / 3])


def test_analyze_time_series_edges():
    s = analyze_time_series(TimeSeriesInput(values=np.array([1.0, 1.0, 1.0, 1.0]), window=1))
    assert list(s.local_maxima_indices) == [] and list(s.local_minima_indices) == []  # плато — не экстремум
    assert np.allclose(s.moving_average, [1, 1, 1, 1])
    s = analyze_time_series(TimeSeriesInput(values=np.array([5.0, 1.0, 5.0]), window=3))
    assert list(s.local_minima_indices) == [1] and list(s.local_maxima_indices) == []
    assert np.allclose(s.moving_average, [11 / 3])
    s = analyze_time_series(TimeSeriesInput(values=np.array([1.0, 2.0]), window=2))
    assert list(s.local_maxima_indices) == [] and list(s.local_minima_indices) == []

def test_one_hot():
    result = one_hot(OneHotInput(labels=np.array([0, 2, 3, 0]), class_count=None))
    assert np.array_equal(result, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])


def test_one_hot_class_count():
    result = one_hot(OneHotInput(labels=np.array([0, 1]), class_count=5))
    assert result.shape == (2, 5)
    assert np.array_equal(result, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    assert one_hot(OneHotInput(labels=np.array([0, 0]), class_count=None)).tolist() == [[1], [1]]
    assert (result.sum(axis=1) == 1).all()


def test_one_hot_invalid():
    with pytest.raises(ValueError):
        one_hot(OneHotInput(labels=np.array([0, 4]), class_count=3))