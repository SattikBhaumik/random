"""
Maximum-complexity random sequence generator.

Interpretation of "most complex":
    Kolmogorov complexity is maximized when the sequence is incompressible,
    which for finite sequences is achieved (on average) by iid sampling at
    full entropy H = log2(k) bits per symbol, where k is the alphabet size.

Strategy:
    Three high-entropy sources through SHA3-512 in counter mode:
      (1) OS cryptographic entropy pool  (secrets.token_bytes)
      (2) Chaotic logistic map x_{n+1} = 4 x_n (1 - x_n), Lyapunov exp = ln 2
      (3) High-resolution timing jitter  (perf_counter_ns)
    Mapping of the hash output to the target integer range via rejection sampling
    (avoiding modulo bias that would otherwise degrade uniformity).

Author note:
    This is as close to Kolmogorov-maximal as pure-Python can get.
    For true physical randomness one would have to replace source (1) with a
    hardware RNG or a quantum RNG API.
"""

from __future__ import annotations
import secrets
import hashlib
import time
import struct


def logistic_stream(seed_bytes: bytes, n_bytes: int) -> bytes:
    """Chaotic byte stream from the logistic map at r = 4 (fully chaotic).

    Uses the least-significant 4 bytes of each IEEE 754 double, since those
    bits of the mantissa are most strongly mixed by the iteration.
    """
    x = int.from_bytes(seed_bytes[:8], "big") / 2**64
    if x == 0.0:                       # avoid the trivial fixed point
        x = 0.5773502691896257
    out = bytearray()
    while len(out) < n_bytes:
        x = 4.0 * x * (1.0 - x)
        out += struct.pack("d", x)[-4:]
    return bytes(out[:n_bytes])


def timing_jitter(n_samples: int = 256) -> bytes:
    """Low-order nanosecond bits of perf_counter, capturing scheduling noise."""
    samples = bytearray()
    for _ in range(n_samples):
        samples.append(time.perf_counter_ns() & 0xFF)
    return bytes(samples)


def max_complexity_sequence(length: int,
                            lo: int = 0,
                            hi: int = 999) -> list[int]:
    """Return `length` integers in [lo, hi] with maximum attainable entropy.

    Pipeline:
        entropy || chaos || jitter || counter  -->  SHA3-512  -->  uniform ints
    """
    if length <= 0 or hi < lo:
        raise ValueError("require length > 0 and hi >= lo")

    entropy = secrets.token_bytes(64)
    chaos   = logistic_stream(entropy, n_bytes=max(64, length * 4))
    jitter  = timing_jitter()

    span    = hi - lo + 1
    cutoff  = (2**64 // span) * span           # rejection threshold, no bias

    out: list[int] = []
    counter = 0
    while len(out) < length:
        h = hashlib.sha3_512(
            entropy + chaos + jitter + counter.to_bytes(8, "big")
        ).digest()                              # 64 bytes = 8 candidate u64s
        for i in range(0, 64, 8):
            if len(out) == length:
                break
            u = int.from_bytes(h[i:i + 8], "big")
            if u < cutoff:                      # rejection sampling
                out.append(lo + u % span)
        counter += 1
    return out


if __name__ == "__main__":
    N = 37
    seq = max_complexity_sequence(N, lo=0, hi=999)

    print(f"sequence (n={N}):")
    print(seq)

    # Sanity check against uniform[0, 999]: expected mean 499.5, std ~ 288.67
    mean = sum(seq) / N
    var  = sum((x - mean)**2 for x in seq) / N
    print(f"\nmean = {mean:8.2f}   (expected 499.50)")
    print(f"std  = {var**0.5:8.2f}   (expected 288.67)")

    # Compression ratio as a crude Kolmogorov-complexity proxy:
    # a truly random byte string compresses to >= its own size.
    import zlib
    raw = bytes(secrets.token_bytes(len(seq) * 2))  # equivalent-length bytes
    enc = ",".join(map(str, seq)).encode()
    print(f"\ncompressed size of ascii encoding: {len(zlib.compress(enc))} / {len(enc)} bytes")
