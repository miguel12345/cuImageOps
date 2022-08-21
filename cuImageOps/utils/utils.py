import math


def gaussian(x: float, sigma: float):
    return (1 / ((math.sqrt(2 * math.pi)) * sigma)) * math.exp(
        -((x**2)) / (2 * (sigma**2))
    )
