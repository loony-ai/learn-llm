"""Chapter 4.6: a hidden source of nondeterminism: the iteration order of sets of strings.

Python randomizes string hashing in every new process (a security measure), so
`list(set(words))` can come out in a different order each time a program runs.
This script runs the same one-liner in three fresh Python processes.

Run from `code/`:  python examples/ch04/hash_order.py
"""

import os
import subprocess
import sys

snippet = "print(list({'keeper', 'lamp', 'boats', 'fog', 'gulls', 'pier'}))"

print("Three fresh processes, default settings:")
for _ in range(3):
    print("  ", subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True).stdout.strip())

print("Three fresh processes with PYTHONHASHSEED=0:")
fixed = dict(os.environ, PYTHONHASHSEED="0")
for _ in range(3):
    print("  ", subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=fixed).stdout.strip())

print("Sorting removes the problem entirely:")
print("  ", sorted({"keeper", "lamp", "boats", "fog", "gulls", "pier"}))
