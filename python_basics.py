"""Решения задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

VOWELS = frozenset("aeiou")


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(1 for char in text.lower() if char in VOWELS)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы должны быть одинаковой длины")
    if len(predicted) == 0:
        raise ValueError("Векторы не должны быть пустыми")
    squared_sum = sum((x - y) ** 2 for x, y in zip(predicted, expected))
    return squared_sum / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts = []
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            power = 0
            while number % divisor == 0:
                number //= divisor
                power += 1
            parts.append(f"({divisor}**{power})" if power > 1 else f"({divisor})")
        divisor += 1 if divisor == 2 else 2
    if number > 1:
        parts.append(f"({number})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    return k if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(char) for char in str(number)]
    length = len(digits)
    left = digits[: (length - 1) // 2]
    right = digits[length // 2 + 1 :]
    return sum(left) == sum(right)