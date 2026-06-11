"""
How many barcode bits do we need to confidently match candidates?

Question this answers:
    Given O(N_candidates) candidates to pick from (e.g. ~10^4), how many bits of
    barcode do we need so that each match is correct with target confidence
    (e.g. 99%)?

Core idea (the "birthday" / collision math, see barcode_birthday.py):
    With b bits there are N = 2^b possible barcodes. If k candidates each draw a
    barcode uniformly at random, a match to a given candidate is only *correct*
    when no OTHER candidate happens to share that same barcode. So:

        P(a specific match is correct) = P(barcode is unique among the k)
                                       = (1 - 1/N)^(k - 1)

    This is exactly `calculate_fraction_unique` from barcode_birthday.py, with
    N = 2^bits. We want to find the smallest b such that this probability meets
    a target (e.g. 0.99).

Rule of thumb (from the exp approximation (1-1/N)^(k-1) ~ exp(-(k-1)/N)):
    N >= (k - 1) / (-ln(target))   ->   for 99% confidence, N ~ 100 * k
    i.e. you need roughly 100x more barcode space than candidates per "nine".
"""

from math import comb
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Poisson infection model: mean MOI + number of marker types -> effective bits
# ---------------------------------------------------------------------------
#
# Model (equal marker abundance, perfect detection):
#   - B distinct marker types. A cell is hit by a Poisson(lambda) number of
#     particles; each particle carries one of the B types uniformly.
#   - Each type lands ~Poisson(lambda / B), so a given type is PRESENT with
#     probability  p = 1 - e^(-lambda / B), independently across types.
#   - A cell's barcode is the SET of present types: a length-B binary presence
#     vector. Multiplicity is ignored (getting type i twice == once), and an
#     extra OFF position adds no identity ("110" and "11" are the same code).
#
# Effective space (Renyi-2 / collision entropy), which is what the birthday /
# uniqueness math actually needs because the distribution is non-uniform:
#   Per-pair collision prob = sum_x P(x)^2. Because bits are independent this
#   factorizes exactly:
#       sum_x P(x)^2 = (p^2 + (1-p)^2)^B
#   so   N_eff = 1 / (p^2 + (1-p)^2)^B
#   and  effective_bits = log2(N_eff) = B * ( -log2(p^2 + (1-p)^2) ).
#
# The per-type term -log2(p^2 + (1-p)^2) is the Renyi-2 entropy of a single
# Bernoulli(p) bit: it maxes at 1 bit when p = 0.5, i.e. lambda = B * ln 2.


def per_type_presence(mean_moi: float, n_types: int) -> float:
    """Probability a given marker type is present: p = 1 - e^(-lambda / B)."""
    return 1.0 - np.exp(-mean_moi / n_types)


def effective_bits(mean_moi: float, n_types: int) -> float:
    """
    Effective barcode bits via collision (Renyi-2) entropy, for `n_types`
    independent markers each present with prob p = 1 - e^(-moi / n_types).

        effective_bits = B * ( -log2(p^2 + (1-p)^2) )

    This is the bit count to feed `confidence_per_match` so that the uniqueness
    estimate correctly accounts for the non-uniform (Hamming-weight) barcode
    distribution. Equals B exactly at the optimum p = 0.5 (moi = B * ln 2).
    """
    p = per_type_presence(mean_moi, n_types)
    collision_per_bit = p ** 2 + (1.0 - p) ** 2
    return n_types * (-np.log2(collision_per_bit))


def optimal_moi(n_types: int) -> float:
    """MOI that maximizes effective bits (p = 0.5): lambda = B * ln 2."""
    return n_types * np.log(2.0)


def confidence_from_moi(
    mean_moi: float, n_types: int, num_candidates: int
) -> float:
    """End-to-end: P(each match correct) given infection MOI and #marker types."""
    return confidence_per_match(effective_bits(mean_moi, n_types), num_candidates)


def confidence_per_match(bits: float, num_candidates: int) -> float:
    """
    Probability that a single match is correct (i.e. the candidate's barcode is
    unique among all candidates), given `bits` of barcode and `num_candidates`
    candidates drawing barcodes uniformly at random.

    P = (1 - 1/N)^(k-1),  N = 2^bits,  k = num_candidates
    """
    k = num_candidates
    if k <= 1:
        return 1.0
    N = 2.0 ** bits
    return (1.0 - 1.0 / N) ** (k - 1)


def bits_needed(num_candidates: int, target_confidence: float = 0.99) -> float:
    """
    Smallest (real-valued) number of bits so that each match is correct with at
    least `target_confidence`.

    Solve (1 - 1/N)^(k-1) >= target for N, then b = log2(N).
    Using the large-N approximation (1-1/N)^(k-1) ~ exp(-(k-1)/N):
        N >= (k - 1) / (-ln(target))
    """
    k = num_candidates
    if k <= 1:
        return 0.0
    N_required = (k - 1) / (-np.log(target_confidence))
    return np.log2(N_required)


def plot_bits_for_confidence(
    candidate_counts: list[int] = [1_000, 10_000, 100_000],
    target_confidence: float = 0.99,
    bit_range: tuple[int, int] = (4, 32),
    num_points: int = 400,
) -> Figure:
    """
    Plot per-match confidence vs. number of bits, for several candidate-pool
    sizes, and annotate the bits required to reach `target_confidence`.

    Args:
        candidate_counts: pool sizes to plot (e.g. ~10^4 candidates)
        target_confidence: desired confidence each match is correct (e.g. 0.99)
        bit_range: (min_bits, max_bits) for the x-axis
        num_points: resolution of each curve
    """
    bits = np.linspace(bit_range[0], bit_range[1], num_points)

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in candidate_counts:
        conf = [confidence_per_match(b, k) for b in bits]
        line, = ax.plot(bits, conf, "-", label=f"{k:,} candidates")

        # Required bits (round UP to a whole bit, since bits are discrete).
        b_req = bits_needed(k, target_confidence)
        b_req_int = int(np.ceil(b_req))
        conf_at_int = confidence_per_match(b_req_int, k)
        ax.plot(
            [b_req_int], [conf_at_int],
            "o", color=line.get_color(), markersize=8,
        )
        ax.annotate(
            f"{b_req_int} bits\n(N={2 ** b_req_int:,})",
            xy=(b_req_int, conf_at_int),
            xytext=(8, -28),
            textcoords="offset points",
            fontsize=9,
            color=line.get_color(),
            arrowprops=dict(arrowstyle="->", color=line.get_color(), alpha=0.6),
        )

    ax.axhline(
        y=target_confidence, color="gray", linestyle="--", alpha=0.6,
        label=f"{target_confidence:.0%} target",
    )

    ax.set_xlabel("Number of barcode bits")
    ax.set_ylabel("P(each match is correct)")
    ax.set_title(
        "Bits needed for confident matching\n"
        "P(unique barcode) = (1 - 1/2^bits)^(k-1)"
    )
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fixed Hamming-depth combinatorial scheme: choose w of B markers per code
# ---------------------------------------------------------------------------
#
# Every cell carries EXACTLY w markers ON out of B types ("Hamming depth" = w),
# as in combinatorial_decoding.ipynb (C(channels, bit_depth)). The barcode space
# is then the number of weight-w binary vectors:
#       N = C(B, w)
# These codes are equiprobable (one weight shell), so N_eff = C(B, w) exactly --
# no non-uniformity penalty -- and fraction unique = (1 - 1/N)^(k-1).
# C(B, w) is maximized at w = B/2, so uniqueness peaks at half-depth.


def combinatorial_space(n_types: int, depth: int) -> int:
    """Number of distinct codes with exactly `depth` of `n_types` markers ON."""
    return comb(n_types, depth)


def fraction_unique_by_depth(
    n_types: int, depth: int, num_candidates: int
) -> float:
    """
    Fraction of codes that are unique (P each match correct) for a fixed
    Hamming-depth combinatorial scheme: N = C(n_types, depth) codes.
    """
    N = combinatorial_space(n_types, depth)
    if N <= 0:
        return 0.0
    return confidence_per_match(np.log2(float(N)), num_candidates)


def plot_fraction_unique_by_depth(
    n_types_list: list[int] = [20, 25, 30],
    num_candidates: int = 30_000,
    target_confidence: float = 0.99,
) -> Figure:
    """
    Fraction unique vs Hamming depth w (bits ON per code), for several
    marker-type counts B, with a fixed candidate pool.

    Args:
        n_types_list: marker-type counts B to compare
        num_candidates: candidate-pool size k (default 30,000)
        target_confidence: confidence line to draw
    """
    fig, (ax_frac, ax_space) = plt.subplots(1, 2, figsize=(14, 6))

    for B in n_types_list:
        depths = np.arange(0, B + 1)
        frac = [fraction_unique_by_depth(B, int(w), num_candidates) for w in depths]
        space = [combinatorial_space(B, int(w)) for w in depths]
        line, = ax_frac.plot(depths, frac, "-o", markersize=3, label=f"B = {B}")
        ax_space.plot(depths, space, "-o", markersize=3, color=line.get_color(),
                      label=f"B = {B}")

        # Mark half-depth (the optimum).
        w_opt = B // 2
        f_opt = fraction_unique_by_depth(B, w_opt, num_candidates)
        ax_frac.plot([w_opt], [f_opt], "o", color=line.get_color(), markersize=9)
        ax_frac.annotate(
            f"w={w_opt}\nN={combinatorial_space(B, w_opt):,}\n{f_opt:.1%}",
            xy=(w_opt, f_opt), xytext=(6, -34), textcoords="offset points",
            fontsize=8, color=line.get_color(),
        )

    ax_frac.axhline(
        y=target_confidence, color="gray", linestyle="--", alpha=0.6,
        label=f"{target_confidence:.0%} target",
    )
    ax_frac.set_xlabel("Hamming depth w (bits ON per code)")
    ax_frac.set_ylabel("Fraction unique = P(each match correct)")
    ax_frac.set_title(
        f"Fraction unique vs Hamming depth\n"
        f"({num_candidates:,} candidates, N = C(B, w))"
    )
    ax_frac.set_ylim(0, 1.02)
    ax_frac.grid(True, alpha=0.3)
    ax_frac.legend(loc="lower center")

    ax_space.axhline(
        y=num_candidates, color="gray", linestyle="--", alpha=0.6,
        label=f"{num_candidates:,} candidates",
    )
    ax_space.set_xlabel("Hamming depth w (bits ON per code)")
    ax_space.set_ylabel("Code space N = C(B, w)")
    ax_space.set_title("Combinatorial code space vs depth")
    ax_space.set_yscale("log")
    ax_space.grid(True, alpha=0.3, which="both")
    ax_space.legend(loc="lower center")

    fig.tight_layout()
    return fig


def plot_moi_to_confidence(
    n_types_list: list[int] = [20, 30, 40],
    num_candidates: int = 10_000,
    target_confidence: float = 0.99,
    moi_max: Optional[float] = None,
    num_points: int = 400,
) -> Figure:
    """
    Two-panel view of the Poisson infection model:
      (left)  effective bits vs mean MOI, optimum (p=0.5) marked
      (right) P(each match correct) vs mean MOI for `num_candidates`

    Args:
        n_types_list: marker-type counts B to compare
        num_candidates: candidate-pool size k (e.g. ~10^4)
        target_confidence: confidence line to draw
        moi_max: max mean MOI on x-axis (default: ~2x the largest optimum)
        num_points: curve resolution
    """
    if moi_max is None:
        moi_max = 2.0 * optimal_moi(max(n_types_list))
    moi = np.linspace(0.01, moi_max, num_points)

    fig, (ax_bits, ax_conf) = plt.subplots(1, 2, figsize=(14, 6))

    for B in n_types_list:
        eff = np.array([effective_bits(m, B) for m in moi])
        conf = np.array([confidence_per_match(b, num_candidates) for b in eff])
        line, = ax_bits.plot(moi, eff, "-", label=f"B = {B} bits")
        ax_conf.plot(moi, conf, "-", color=line.get_color(), label=f"B = {B} bits")

        # Mark the optimum (p = 0.5, effective bits == B).
        m_opt = optimal_moi(B)
        ax_bits.plot([m_opt], [B], "o", color=line.get_color(), markersize=7)
        ax_bits.annotate(
            f"MOI*={m_opt:.1f}\n{B} bits",
            xy=(m_opt, B), xytext=(6, -6), textcoords="offset points",
            fontsize=8, color=line.get_color(),
        )

    ax_bits.set_xlabel("Mean infections per cell (MOI λ)")
    ax_bits.set_ylabel("Effective bits (Renyi-2)")
    ax_bits.set_title("Effective bits vs MOI\nB · (−log₂(p²+(1−p)²)), p = 1−e^(−λ/B)")
    ax_bits.grid(True, alpha=0.3)
    ax_bits.legend()

    ax_conf.axhline(
        y=target_confidence, color="gray", linestyle="--", alpha=0.6,
        label=f"{target_confidence:.0%} target",
    )
    ax_conf.set_xlabel("Mean infections per cell (MOI λ)")
    ax_conf.set_ylabel("P(each match is correct)")
    ax_conf.set_title(f"Match confidence vs MOI\n({num_candidates:,} candidates)")
    ax_conf.set_ylim(0, 1.02)
    ax_conf.grid(True, alpha=0.3)
    ax_conf.legend(loc="lower right")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Headline answer to the motivating question.
    for k in [1_000, 10_000, 100_000]:
        b = bits_needed(k, 0.99)
        b_int = int(np.ceil(b))
        print(
            f"{k:>8,} candidates -> need {b:5.2f} bits "
            f"(use {b_int} bits, N=2^{b_int}={2 ** b_int:,}); "
            f"confidence at {b_int} bits = {confidence_per_match(b_int, k):.4f}"
        )

    # Plot 1: confidence vs raw bits (clean barcodes).
    fig1 = plot_bits_for_confidence(
        candidate_counts=[1_000, 10_000, 100_000],
        target_confidence=0.99,
    )

    # Plot 2: Poisson infection model -- effective bits & confidence vs MOI.
    fig2 = plot_moi_to_confidence(
        n_types_list=[20, 30, 40],
        num_candidates=10_000,
        target_confidence=0.99,
    )

    # Plot 3: fixed Hamming-depth combinatorial scheme, 30k candidates.
    fig3 = plot_fraction_unique_by_depth(
        n_types_list=[20, 25, 30],
        num_candidates=30_000,
        target_confidence=0.99,
    )

    for fig, name in [
        (fig1, "bits_for_confidence.png"),
        (fig2, "moi_to_confidence.png"),
        (fig3, "fraction_unique_by_depth.png"),
    ]:
        fig.savefig(name, dpi=120, bbox_inches="tight")
        print(f"saved {name}")

    plt.show()
