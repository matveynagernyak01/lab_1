import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)


def num(n):
    return PositiveIntegerInput(value=n)


def text(s):
    return TextInput(value=s)


@pytest.mark.parametrize("s, expected", [
    ("", 0), ("bcdfg", 0), ("aeiou", 5), ("AEIOU", 5), ("Hello World", 3), ("rhythm", 0),
])
def test_count_vowels(s, expected):
    assert count_vowels(text(s)) == expected


@pytest.mark.parametrize("s, expected", [
    ("", True), ("a", True), ("abc", True), ("aba", False), ("abcdea", False), ("aA", True),
])
def test_has_unique_characters(s, expected):
    assert has_unique_characters(text(s)) is expected


@pytest.mark.parametrize("n, expected", [
    (1, 1), (2, 1), (7, 3), (8, 1), (255, 8), (1023, 10), (2 ** 100, 1), (2 ** 100 - 1, 100),
])
def test_count_one_bits(n, expected):
    assert count_one_bits(num(n)) == expected


@pytest.mark.parametrize("n, expected", [
    (4, 0), (9, 0), (10, 1), (39, 3), (999, 4), (25, 2), (77, 4), (679, 5),
])
def test_multiplicative_persistence(n, expected):
    assert multiplicative_persistence(num(n)) == expected


def test_mse():
    assert mse(VectorPairInput(predicted=[1, 2, 3], expected=[1, 2, 3])) == 0
    assert mse(VectorPairInput(predicted=[1, 2, 3], expected=[2, 4, 6])) == pytest.approx(14 / 3)
    assert mse(VectorPairInput(predicted=[5], expected=[2])) == 9
    assert mse(VectorPairInput(predicted=[-1, 1], expected=[1, -1])) == 4


def test_mse_invalid():
    with pytest.raises(ValueError):
        mse(VectorPairInput(predicted=[1, 2], expected=[1]))
    with pytest.raises(ValueError):
        mse(VectorPairInput(predicted=[], expected=[]))


@pytest.mark.parametrize("n, expected", [
    (1, ""), (2, "(2)"), (4, "(2**2)"), (12, "(2**2)(3)"), (97, "(97)"),
    (86240, "(2**5)(5)(7**2)(11)"), (1024, "(2**10)"), (9973 * 2, "(2)(9973)"),
])
def test_prime_factorization(n, expected):
    assert prime_factorization(num(n)) == expected


@pytest.mark.parametrize("n, expected", [
    (1, 1), (5, 2), (14, 3), (30, 4), (55, 5), (91, 6), (4, "It is impossible"),
    (2, "It is impossible"), (15, "It is impossible"), (100, "It is impossible"),
])
def test_pyramid(n, expected):
    assert pyramid(num(n)) == expected


@pytest.mark.parametrize("n, expected", [
    (1, True), (7, True), (13, True), (99, True), (432, False), (424, True),
    (959, True), (1234006, True), (123456, False), (1221, True), (1232, False),
    (4, True), (12321, True), (1024, False),
])
def test_is_balanced_number(n, expected):
    assert is_balanced_number(num(n)) is expected