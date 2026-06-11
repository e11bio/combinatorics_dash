import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def calculate_collision_probability(
    num_barcodes: int, barcode_space_size: int
) -> float:
    """
    Calculate the theoretical probability of at least one collision in a set of barcodes.
    Uses the complement of the probability of no collisions.

    Args:
        num_barcodes: Number of barcodes being used
        barcode_space_size: Total possible unique barcodes in the space

    Returns:
        Probability of at least one collision (between 0 and 1)
    """
    # Calculate probability of no collisions and take complement
    no_collision_prob = 1.0
    for i in range(num_barcodes):
        no_collision_prob *= (barcode_space_size - i) / barcode_space_size

    return 1 - no_collision_prob


def calculate_fraction_unique(num_barcodes: int, barcode_space_size: int) -> float:
    """
    Calculate the fraction of barcodes that are expected to be unique (have no collisions).

    This calculation assumes a uniform distribution of barcodes in the virus library.
    For k infected neurons and N possible barcodes:
    - For any single neuron A, P(A) = 1/N is the probability of getting a specific barcode
    - For k-1 other neurons, (1-1/N)^(k-1) is the probability that none share A's barcode
    - Therefore, the fraction of unique neurons F = (1-1/N)^(k-1)

    Args:
        num_barcodes: Number of infected neurons (k)
        barcode_space_size: Total possible unique barcodes in the space (N)

    Returns:
        Expected fraction of uniquely labeled neurons (between 0 and 1)
    """
    k = num_barcodes
    N = barcode_space_size

    # Handle edge cases
    if k <= 1:
        return 1.0
    if N <= 0:
        return 0.0

    # Calculate fraction of unique neurons: F = (1-1/N)^(k-1)
    return (1 - 1 / N) ** (k - 1)


def simulate_collisions(
    num_barcodes: int, barcode_space_size: int, num_trials: int = 10000
) -> float:
    """
    Simulate barcode assignments to estimate collision probability.

    Args:
        num_barcodes: Number of barcodes being used
        barcode_space_size: Total possible unique barcodes in the space
        num_trials: Number of simulation trials to run

    Returns:
        Estimated probability of at least one collision
    """
    collisions = 0

    for _ in range(num_trials):
        # Randomly select barcodes
        selected = np.random.choice(barcode_space_size, size=num_barcodes, replace=True)
        # Check if there are any duplicates
        if len(np.unique(selected)) < num_barcodes:
            collisions += 1

    return collisions / num_trials


def plot_collision_probabilities(
    max_barcodes: int,
    barcode_space_size: int,
    num_points: int = 100,
    include_simulation: bool = True,
    num_trials: int = 1000,
) -> None:
    """
    Plot collision probabilities as the number of barcodes increases.

    Args:
        max_barcodes: Maximum number of barcodes to plot up to
        barcode_space_size: Total possible unique barcodes in the space
        num_points: Number of points to plot (default 100)
        include_simulation: Whether to include simulation results
        num_trials: Number of trials for each simulation point
    """
    # Create array of barcode counts to evaluate
    ns = np.linspace(1, max_barcodes, num_points)

    # Calculate theoretical probabilities
    theory_probs = [
        calculate_collision_probability(int(n), barcode_space_size) for n in ns
    ]

    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(ns, theory_probs, "b-", label="Theoretical")

    if include_simulation:
        # Calculate simulation probabilities for fewer points to save time
        sim_ns = np.linspace(1, max_barcodes, 20)  # fewer points for simulation
        sim_probs = [
            simulate_collisions(int(n), barcode_space_size, num_trials) for n in sim_ns
        ]
        plt.plot(sim_ns, sim_probs, "ro", label="Simulated", alpha=0.6)

    # Add reference lines
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    plt.axhline(y=0.99, color="gray", linestyle="--", alpha=0.3)

    # Customize the plot
    plt.xlabel("Number of Barcodes")
    plt.ylabel("Probability of Collision")
    plt.title(f"Barcode Collision Probability\n(Space Size: {barcode_space_size:,})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text annotations for key probabilities
    n_50_percent = next((i for i, p in enumerate(theory_probs) if p > 0.5), None)
    if n_50_percent is not None:
        plt.annotate(
            f"50% collision probability\nat ~{int(ns[n_50_percent])} cells",
            xy=(ns[n_50_percent], 0.5),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

    plt.tight_layout()
    return fig


def plot_unique_fractions(
    max_barcodes: int, barcode_space_size: int, num_points: int = 100
) -> plt.Figure:
    """
    Plot the expected fraction of unique barcodes as the number of barcodes increases.

    Args:
        max_barcodes: Maximum number of barcodes to plot up to
        barcode_space_size: Total possible unique barcodes in the space
        num_points: Number of points to plot
    """
    # Create array of barcode counts to evaluate
    ns = np.linspace(1, max_barcodes, num_points)

    # Calculate fractions
    fractions = [calculate_fraction_unique(int(n), barcode_space_size) for n in ns]

    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(ns, fractions, "g-", label="Theoretical")

    # Add reference lines
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    plt.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)

    # Customize the plot
    plt.xlabel("Number of Labeled Neurons")
    plt.ylabel("Fraction of Unique Labels")
    plt.title(
        f"Expected Fraction of Unique Barcodes\n(Space Size: {barcode_space_size:,})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text annotations for key fractions
    n_90_percent = next((i for i, f in enumerate(fractions) if f < 0.9), None)
    if n_90_percent is not None:
        plt.annotate(
            f"90% unique labels\nat ~{int(ns[n_90_percent])} cells",
            xy=(ns[n_90_percent], 0.9),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

    plt.tight_layout()
    return fig


def plot_unique_vs_ratio(
    max_ratio: float = 1.0,
    barcode_space_sizes: list[int] = [1000, 10000, 100000],
    num_points: int = 100,
) -> plt.Figure:
    """
    Plot the fraction of unique barcodes as a function of cells/total barcodes ratio.

    Args:
        max_ratio: Maximum ratio of cells to total barcodes to plot (default 1.0 = 100%)
        barcode_space_sizes: List of different barcode space sizes to plot
        num_points: Number of points to plot for each curve
    """
    # Create array of ratios to evaluate
    ratios = np.linspace(0, max_ratio, num_points)

    # Create the plot
    fig = plt.figure(figsize=(10, 6))

    # Plot for each barcode space size
    for N in barcode_space_sizes:
        # Calculate number of cells for each ratio
        cells = (ratios * N).astype(int)
        # Calculate fractions
        fractions = [calculate_fraction_unique(k, N) if k > 0 else 1.0 for k in cells]
        plt.plot(ratios, fractions, "-", label=f"N = {N:,}")

    # Add reference lines
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    plt.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)

    # Customize the plot
    plt.xlabel("Ratio (Cells / Total Barcodes)")
    plt.ylabel("Fraction of Unique Labels")
    plt.title("Expected Fraction of Unique Barcodes\nvs Relative Space Saturation")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add percentage on top x-axis
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel("Percent of Barcode Space Used")
    ax2.set_xticks(np.linspace(0, max_ratio, 6))
    ax2.set_xticklabels([f"{x * 100:.0f}%" for x in np.linspace(0, max_ratio, 6)])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage with all plots
    space_size = 10000

    # Create and display the collision probability plot
    plot_collision_probabilities(
        max_barcodes=200, barcode_space_size=space_size, include_simulation=True
    )
    plt.figure()  # Create new figure

    # Create and display the unique fraction plot
    plot_unique_fractions(max_barcodes=200, barcode_space_size=space_size)
    plt.figure()  # Create new figure

    # Create and display the ratio plot
    plot_unique_vs_ratio(
        max_ratio=0.5,  # Plot up to 50% saturation
        barcode_space_sizes=[1000, 10000, 100000],
    )
    plt.show()

    # Print some example scenarios
    test_cases = [
        (10, 1000),
        (50, 1000),
        (100, 1000),
        (20, 10000),
        (100, 10000),
    ]

    print("\nExample scenarios:")
    for n, space in test_cases:
        unique_frac = calculate_fraction_unique(n, space)
        ratio = n / space
        print(
            f"{n} neurons in space of {space} (ratio: {ratio:.1%}): {unique_frac:.4f} fraction unique"
        )
