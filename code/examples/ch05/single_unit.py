"""Chapter 5.2: a single unit is an adjustable scoring rule.

The task: score how urgent a harbor message is, from three yes/no features.
Here a person sets the adjustable numbers by hand. In Chapter 6, training sets them.

Run from `code/`:  python examples/ch05/single_unit.py
"""

FEATURES = ["mentions_storm", "mentions_injury", "mentions_market"]


def unit(inputs: list[float], weights: list[float], bias: float) -> float:
    """Multiply each input by its weight, add the results, then add the bias."""
    total = bias
    for value, weight in zip(inputs, weights):
        total += value * weight
    return total


messages = {
    "Storm coming in tonight":            [1.0, 0.0, 0.0],
    "Fisher hurt on the pier":            [0.0, 1.0, 0.0],
    "Market opens late tomorrow":         [0.0, 0.0, 1.0],
    "Storm damage, two fishers injured":  [1.0, 1.0, 0.0],
}

settings = {
    "storm and injury matter":      ([2.0, 3.0, -1.0], -0.5),
    "only the market matters":      ([0.0, 0.0, 2.0], 0.0),
    "everything slightly negative": ([-1.0, -1.0, -1.0], 0.0),
}

for label, (weights, bias) in settings.items():
    print(f"weights={weights} bias={bias}  ({label})")
    for text, features in messages.items():
        print(f"   {unit(features, weights, bias):6.2f}  {text}")
