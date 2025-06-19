import itertools
import math
from typing import Optional

import pandas as pd


class BrainbowConfig:
    """
    Configuration class for Brainbow cassette generation and analysis.
    Centralizes marker and tag definitions to avoid variable tunneling.
    """

    def __init__(
        self,
        base_markers: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        max_tags_per_marker: int = 2,
    ):
        """
        Initialize Brainbow configuration.

        Args:
            base_markers: List of base markers (fluorescent proteins)
            tags: List of available tags
            max_tags_per_marker: Maximum number of tags per marker
        """
        self.base_markers = base_markers or [
            "mNeonGreen",
            "hfYFP",
            "mScarlet3",
            "mBaojin",
        ]
        self.tags = tags or ["HSV", "TD", "ALFA", "NWS"]
        self.max_tags_per_marker = max_tags_per_marker

        # Create sets for efficient lookup
        self.base_markers_set = set(self.base_markers)
        self.tags_set = set(self.tags)

    def get_all_markers(self) -> list[str]:
        """Get all available markers (base markers + tags)."""
        return self.base_markers + self.tags

    def is_base_marker(self, marker: str) -> bool:
        """Check if a marker is a base marker."""
        return marker in self.base_markers_set

    def is_tag(self, marker: str) -> bool:
        """Check if a marker is a tag."""
        return marker in self.tags_set


# Global default configuration
DEFAULT_CONFIG = BrainbowConfig()


def create_cassette(
    null_marker: str = "Null",
    bit1_marker: str = "hfYFP",
    bit2_marker: str = "mNeonGreen",
    bit3_marker: str = "mScarlet",
) -> list[tuple[str, str]]:
    """
    Create a cassette with configurable markers for each state.

    Args:
        null_marker: Marker for Null state (00)
        bit1_marker: Marker for Bit1 state (10)
        bit2_marker: Marker for Bit2 state (01)
        bit3_marker: Marker for Bit3 state (11)

    Returns:
        List of (state_label, protein) tuples
    """
    return [
        ("Null", null_marker),  # 00
        ("Bit1", bit1_marker),  # 10
        ("Bit2", bit2_marker),  # 01
        ("Bit3", bit3_marker),  # 11
    ]


# --------- 2. Utility for "linear" readout (presence / absence of each marker) --
def linear_markers(protein: str) -> set[str]:
    """
    Extract unique markers from a protein string by splitting on '-' and normalizing case.
    Example: "mScarlet-OLLAS" -> {"mScarlet", "OLLAS"}
    """
    if not protein or protein.lower() == "null":
        return set()

    # Split on '-' and normalize case
    return {marker.strip() for marker in protein.split("-")}


# --------- 3. Enumerate all 16 joint outcomes ---------------------------
def outcome_table(
    p_cre: float = 0.8,
    cassette_a: Optional[list[tuple[str, str]]] = None,
    cassette_b: Optional[list[tuple[str, str]]] = None,
    pos1_bias: float = 1,  # 2.3, additional reports report no bias...who knows.
    pos2_bias: float = 1,  # 0.36,
    pos3_bias: float = 1,  # 0.32,
) -> pd.DataFrame:
    """
    Generate outcome table for two cassettes with positional recombination efficiency biases.

    Args:
        p_cre: fraction of cells that see Cre (same for both cassettes)
        cassette_a: List of (state_label, protein) tuples for cassette A
        cassette_b: List of (state_label, protein) tuples for cassette B
        pos1_bias: recombination efficiency bias for position 1 (Bit1) relative to uniform
        pos2_bias: recombination efficiency bias for position 2 (Bit2) relative to uniform
        pos3_bias: recombination efficiency bias for position 3 (Bit3) relative to uniform

    Returns:
        DataFrame with all possible outcomes and their probabilities
    """
    # Use default cassettes if none provided
    if cassette_a is None:
        cassette_a = create_cassette()
    if cassette_b is None:
        cassette_b = create_cassette()

    # Calculate normalized probabilities for each position
    # The biases are relative to uniform probability, so we need to normalize
    total_bias = pos1_bias + pos2_bias + pos3_bias
    p_bit1 = (p_cre * pos1_bias) / total_bias
    p_bit2 = (p_cre * pos2_bias) / total_bias
    p_bit3 = (p_cre * pos3_bias) / total_bias
    p_null = 1 - p_cre

    rows = []

    for (lab_a, prot_a), (lab_b, prot_b) in itertools.product(cassette_a, cassette_b):
        # Calculate probability for each cassette based on state
        if lab_a == "Null":
            p_a = p_null
        elif lab_a == "Bit1":
            p_a = p_bit1
        elif lab_a == "Bit2":
            p_a = p_bit2
        elif lab_a == "Bit3":
            p_a = p_bit3
        else:
            p_a = 0  # fallback

        if lab_b == "Null":
            p_b = p_null
        elif lab_b == "Bit1":
            p_b = p_bit1
        elif lab_b == "Bit2":
            p_b = p_bit2
        elif lab_b == "Bit3":
            p_b = p_bit3
        else:
            p_b = 0  # fallback

        rows.append(
            {
                "CassetteA_state": lab_a,
                "CassetteB_state": lab_b,
                "ProteinA": prot_a,
                "ProteinB": prot_b,
                # ---- linear readout: unique marker set --------------------
                "Linear_readout": tuple(
                    sorted(linear_markers(prot_a) | linear_markers(prot_b))
                ),
                # ---- combinatorial readout: keep proteins distinct --------
                "Combinatorial_readout": tuple([prot_a, prot_b]),
                # ---- overall probability ---------------------------------
                "Probability": round(p_a * p_b, 6),
                "Num_channels": len(linear_markers(prot_a) | linear_markers(prot_b)),
            }
        )

    return pd.DataFrame(rows)


# --------- 4. Panel entropy ----------------------------------------------------
# flatten to get the full marker list
def entropy(sig_probs):
    """sig_probs = dict {signature: total_probability}"""
    return -sum(p * math.log2(p) for p in sig_probs.values())


def find_best_panels(df, num_markers, print_output=True, readout_col="Linear_readout"):
    all_markers = sorted({m for tup in df[readout_col] for m in tup})
    # make a test that num markers is not greater than the number of markers in all_markers
    if num_markers > len(all_markers):
        raise ValueError(
            f"Number of markers is greater than the number of markers in all_markers: {num_markers} > {len(all_markers)}"
        )
    best_score = -1
    best_panels = []

    panels = itertools.combinations(all_markers, num_markers)
    for panel in panels:
        panel = set(panel)
        df["filtered_sig"] = df[readout_col].apply(
            lambda tup: tuple(sorted(set(tup) & panel))  # noqa: B023
        )
        sig_probs = df.groupby("filtered_sig")["Probability"].sum().to_dict()
        score = entropy(sig_probs)  # or distinct-count

        if score > best_score:
            best_score = score
            best_panels = [panel]
        elif score == best_score:
            best_panels.append(panel)
        if print_output:
            print(f"Best entropy = {best_score:.3f} bits")
            for p in best_panels:
                print("Panel:", sorted(p))
        return best_score, best_panels


def calculate_panel_entropy(df, panel):
    """Calculate the entropy of a specific panel configuration."""
    panel = set(panel)

    # filter each row down to the chosen markers
    filt = df["Linear_readout"].apply(lambda tup: tuple(sorted(set(tup) & panel)))

    # get probabilities for each unique signature
    sig_probs = df.groupby(filt)["Probability"].sum()

    # calculate entropy - fix the .values() issue
    panel_entropy = -sum(p * math.log2(p) for p in sig_probs.values)
    return panel_entropy


def calculate_per_channel_entropy(df, panel):
    """Calculate the per-channel entropy of a specific panel configuration.
    Returns a dictionary with channel names as keys and entropy values as values."""
    panel = set(panel)
    channel_entropy = {}
    # filter each row down to the chosen markers
    panel_entropy = calculate_panel_entropy(df, panel)
    for channel in panel:
        reduced_panel = panel - {channel}
        reduced_entropy = calculate_panel_entropy(df, reduced_panel)
        channel_entropy[channel] = panel_entropy - reduced_entropy
    return channel_entropy


def get_best_panel_entropy(
    df: pd.DataFrame, num_channels: int, readout_col: str = "Linear_readout"
) -> float:
    """
    Wrapper function that finds the best panel for a specified number of channels
    and returns its entropy.

    Args:
        df: DataFrame with outcome data (from outcome_table)
        num_channels: Number of channels/markers to use in the panel
        readout_col: Column name to use for readout analysis (default: "Linear_readout")

    Returns:
        float: Entropy of the best panel in bits

    Raises:
        ValueError: If num_channels is greater than available markers
    """
    all_markers = sorted({m for tup in df[readout_col] for m in tup})

    if num_channels > len(all_markers):
        raise ValueError(
            f"Number of channels ({num_channels}) is greater than available markers ({len(all_markers)})"
        )

    best_score = -1

    for panel in itertools.combinations(all_markers, num_channels):
        panel = set(panel)

        # Filter each row to the chosen markers
        df_temp = df.copy()
        df_temp["filtered_sig"] = df_temp[readout_col].apply(
            lambda tup: tuple(sorted(set(tup) & panel))  # noqa: B023
        )

        # Aggregate probabilities for identical signatures
        sig_probs = df_temp.groupby("filtered_sig")["Probability"].sum().to_dict()

        # Calculate entropy
        score = entropy(sig_probs)

        if score > best_score:
            best_score = score
            best_panel = panel

    return best_score, best_panel


def get_best_panel_info(
    df: pd.DataFrame,
    num_channels: int,
    readout_col: str = "Linear_readout",
    base_marker_weight: float = 1.0,
    tag_weight: float = 0.5,
    config: Optional[BrainbowConfig] = None,
) -> dict:
    """
    Wrapper function that finds the best panel for a specified number of channels
    and returns comprehensive information about it.

    Args:
        df: DataFrame with outcome data (from outcome_table)
        num_channels: Number of channels/markers to use in the panel
        readout_col: Column name to use for readout analysis (default: "Linear_readout")
        base_marker_weight: Weight multiplier for base markers in entropy calculation (default: 1.0)
        tag_weight: Weight multiplier for tags in entropy calculation (default: 0.5)
        config: BrainbowConfig object defining available markers and tags

    Returns:
        dict: Dictionary containing:
            - 'entropy': float, entropy of the best panel in bits
            - 'best_panels': list of sets, all panels that achieve the best entropy
            - 'num_channels': int, number of channels used
            - 'entropy_per_channel': float, entropy divided by number of channels
            - 'available_markers': list, all available markers in the dataset
            - 'weights_used': dict, the weights used for marker selection
    """
    all_markers = sorted({m for tup in df[readout_col] for m in tup})

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    if num_channels > len(all_markers):
        raise ValueError(
            f"Number of channels ({num_channels}) is greater than available markers ({len(all_markers)})"
        )

    best_score = -1
    best_panels = []

    for panel in itertools.combinations(all_markers, num_channels):
        panel = set(panel)

        # Filter each row to the chosen markers
        df_temp = df.copy()
        df_temp["filtered_sig"] = df_temp[readout_col].apply(
            lambda tup: tuple(sorted(set(tup) & panel))  # noqa: B023
        )

        # Aggregate probabilities for identical signatures
        sig_probs = df_temp.groupby("filtered_sig")["Probability"].sum().to_dict()

        # Calculate weighted entropy based on marker types
        weighted_entropy = 0
        for signature, prob in sig_probs.items():
            # Count base markers and tags in this signature
            base_marker_count = sum(
                1 for m in signature if m in config.base_markers_set
            )
            tag_count = len(signature) - base_marker_count

            # Apply weights to the probability
            weighted_prob = (
                prob * (base_marker_weight**base_marker_count) * (tag_weight**tag_count)
            )
            weighted_entropy -= weighted_prob * math.log2(weighted_prob)

        if weighted_entropy > best_score:
            best_score = weighted_entropy
            best_panels = [panel]
        elif weighted_entropy == best_score:
            best_panels.append(panel)

    return {
        "entropy": best_score,
        "best_panels": best_panels,
        "num_channels": num_channels,
        "entropy_per_channel": best_score / num_channels,
        "available_markers": all_markers,
        "weights_used": {
            "base_marker_weight": base_marker_weight,
            "tag_weight": tag_weight,
        },
    }


def generate_shuffled_cassettes(
    num_configurations: int = 1000,
    seed: Optional[int] = None,
    config: Optional[BrainbowConfig] = None,
) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """
    Generate shuffled cassette configurations following the specified rules.

    Rules:
    - Each element must have one and only one base marker from the config
    - Each element can have 0-max_tags_per_marker additional tags from the config

    Args:
        num_configurations: Number of random configurations to generate
        seed: Random seed for reproducibility
        config: BrainbowConfig object defining available markers and tags

    Returns:
        List of tuples, each containing (cassette_a, cassette_b) configurations
    """
    import random

    if seed is not None:
        random.seed(seed)

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    # Sort markers to ensure order-independent results with same seed
    sorted_base_markers = sorted(config.base_markers)
    sorted_tags = sorted(config.tags)

    configurations = []

    for _ in range(num_configurations):
        # Generate cassette A
        cassette_a = []
        for state in ["Null", "Bit1", "Bit2", "Bit3"]:
            # Choose one base marker from sorted list
            base_marker = random.choice(sorted_base_markers)

            # Choose 0 to max_tags_per_marker additional tags from sorted list
            num_tags = random.randint(0, config.max_tags_per_marker)
            selected_tags = random.sample(sorted_tags, num_tags)

            # Combine base marker with tags
            if selected_tags:
                marker = base_marker + "-" + "-".join(selected_tags)
            else:
                marker = base_marker

            cassette_a.append((state, marker))

        # Generate cassette B
        cassette_b = []
        for state in ["Null", "Bit1", "Bit2", "Bit3"]:
            # Choose one base marker from sorted list
            base_marker = random.choice(sorted_base_markers)

            # Choose 0 to max_tags_per_marker additional tags from sorted list
            num_tags = random.randint(0, config.max_tags_per_marker)
            selected_tags = random.sample(sorted_tags, num_tags)

            # Combine base marker with tags
            if selected_tags:
                marker = base_marker + "-" + "-".join(selected_tags)
            else:
                marker = base_marker

            cassette_b.append((state, marker))

        configurations.append((cassette_a, cassette_b))

    return configurations


def find_optimal_cassette_configuration(
    num_configurations: int = 1000,
    p_cre: float = 0.5,
    target_channels: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    base_marker_weight: float = 1.0,
    tag_weight: float = 0.5,
    min_base_markers: int = 2,
    config: Optional[BrainbowConfig] = None,
) -> dict:
    """
    Find the cassette configuration that maximizes entropy.

    Args:
        num_configurations: Number of random configurations to test
        p_cre: Probability of Cre recombination
        target_channels: If specified, optimize for this number of channels
        seed: Random seed for reproducibility
        verbose: Whether to print progress updates
        base_marker_weight: Weight multiplier for base markers in entropy calculation (default: 1.0)
        tag_weight: Weight multiplier for tags in entropy calculation (default: 0.5)
        min_base_markers: Minimum number of base markers to include in the panel (default: 2)
        config: BrainbowConfig object defining available markers and tags

    Returns:
        Dictionary containing:
            - 'best_entropy': float, highest entropy found
            - 'best_cassette_a': list, optimal cassette A configuration
            - 'best_cassette_b': list, optimal cassette B configuration
            - 'best_df': DataFrame, outcome table for best configuration
            - 'best_panel_info': dict, panel information if target_channels specified
            - 'configurations_tested': int, number of configurations tested
            - 'weights_used': dict, the weights used for marker selection
            - 'config_used': BrainbowConfig, the configuration used
    """
    import random

    if seed is not None:
        random.seed(seed)

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    configurations = generate_shuffled_cassettes(
        num_configurations=num_configurations, seed=seed, config=config
    )

    best_entropy = -1
    best_cassette_a = None
    best_cassette_b = None
    best_df = None
    best_panel_info = None

    for i, (cassette_a, cassette_b) in enumerate(configurations):
        if verbose and (i + 1) % 100 == 0:
            print(f"Tested {i + 1}/{num_configurations} configurations...")

        try:
            # Generate outcome table
            df = outcome_table(
                p_cre=p_cre, cassette_a=cassette_a, cassette_b=cassette_b
            )

            if target_channels is not None:
                # Optimize for specific number of channels
                try:
                    panel_info = get_best_panel_info_two_step(
                        df,
                        target_channels,
                        base_marker_weight=base_marker_weight,
                        tag_weight=tag_weight,
                        min_base_markers=min_base_markers,
                        config=config,
                    )
                    current_entropy = panel_info["entropy"]
                except ValueError:
                    # Not enough unique markers for target channels
                    continue
            else:
                # Use all available markers
                all_markers = sorted({m for tup in df["Linear_readout"] for m in tup})
                if len(all_markers) == 0:
                    continue

                panel_info = get_best_panel_info_two_step(
                    df,
                    len(all_markers),
                    base_marker_weight=base_marker_weight,
                    tag_weight=tag_weight,
                    min_base_markers=min_base_markers,
                    config=config,
                )
                current_entropy = panel_info["entropy"]

            if current_entropy > best_entropy:
                best_entropy = current_entropy
                best_cassette_a = cassette_a
                best_cassette_b = cassette_b
                best_df = df
                best_panel_info = panel_info

                if verbose:
                    print(
                        f"New best entropy: {best_entropy:.3f} bits at configuration {i + 1}"
                    )
                    if target_channels:
                        print(
                            f"  Best panel: {sorted(list(best_panel_info['best_panels'][0]))}"
                        )
                        print(
                            f"  Base markers: {sorted(list(best_panel_info['base_markers_used']))}"
                        )
                        print(f"  Tags: {sorted(list(best_panel_info['tags_used']))}")

        except Exception as e:
            if verbose:
                print(f"Error in configuration {i + 1}: {e}")
            continue

    return {
        "best_entropy": best_entropy,
        "best_cassette_a": best_cassette_a,
        "best_cassette_b": best_cassette_b,
        "best_df": best_df,
        "best_panel_info": best_panel_info,
        "configurations_tested": num_configurations,
        "weights_used": {
            "base_marker_weight": base_marker_weight,
            "tag_weight": tag_weight,
        },
        "config_used": config,
    }


def analyze_cell_labeling(
    df: pd.DataFrame, panel: set, readout_col: str = "Linear_readout"
) -> dict:
    """
    Analyze what percentage of cells will be labeled with a given panel.

    Args:
        df: DataFrame with outcome data (from outcome_table)
        panel: Set of markers/channels to analyze
        readout_col: Column name to use for readout analysis

    Returns:
        dict: Dictionary containing:
            - 'total_labeled_percent': float, percentage of cells with any labeling
            - 'unlabeled_percent': float, percentage of cells with no labeling
            - 'combination_distribution': DataFrame, distribution of marker combinations
            - 'num_combinations': int, number of unique combinations
    """
    # Filter each row to the chosen markers
    df_temp = df.copy()
    df_temp["filtered_sig"] = df_temp[readout_col].apply(
        lambda tup: tuple(sorted(set(tup) & panel))
    )

    # Calculate probabilities for each combination
    combination_probs = (
        df_temp.groupby("filtered_sig")["Probability"].sum().reset_index()
    )
    combination_probs.columns = ["Marker_Combination", "Probability"]
    combination_probs["Percentage"] = combination_probs["Probability"] * 100
    combination_probs = combination_probs.sort_values("Percentage", ascending=False)

    # Calculate labeled vs unlabeled
    unlabeled_prob = combination_probs[combination_probs["Marker_Combination"] == ()][
        "Probability"
    ].sum()
    labeled_prob = 1 - unlabeled_prob

    return {
        "total_labeled_percent": labeled_prob * 100,
        "unlabeled_percent": unlabeled_prob * 100,
        "combination_distribution": combination_probs,
        "num_combinations": len(combination_probs),
    }


def get_best_panel_info_two_step(
    df: pd.DataFrame,
    num_channels: int,
    readout_col: str = "Linear_readout",
    base_marker_weight: float = 1.0,
    tag_weight: float = 0.5,
    min_base_markers: int = 2,
    config: Optional[BrainbowConfig] = None,
) -> dict:
    """
    Exhaustively search for the highest-entropy panel made of `num_channels`
    markers, subject to ≥`min_base_markers` "base" markers.  Marker importance
    is introduced through multiplicative weights and *renormalised* before the
    Shannon-entropy calculation.

    Returns a dictionary with the best score and all equally good panels.
    """

    # ──────────────────────────  guard clauses & basic sets  ────────────────────
    all_markers = sorted({m for tup in df[readout_col] for m in tup})

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    available_base_markers = config.base_markers_set & set(all_markers)
    available_tags = set(all_markers) - config.base_markers_set

    if num_channels > len(all_markers):
        raise ValueError(
            f"num_channels ({num_channels}) exceeds markers available ({len(all_markers)})."
        )
    if min_base_markers > len(available_base_markers):
        raise ValueError(
            f"min_base_markers ({min_base_markers}) exceeds available base markers ({len(available_base_markers)})."
        )

    # ────────────────────────────  pre-compute per-row sets  ───────────────────
    df = df.copy()  # avoid mutating caller's df
    df["marker_set"] = df[readout_col].apply(set)

    # ────────────────────────────  exhaustive search loop  ─────────────────────
    best_score = -float("inf")
    best_panels: list[set[str]] = []
    best_base_markers: Optional[set[str]] = None
    best_tags: Optional[set[str]] = None
    # EPS = 1e-12  # tolerance for float equality

    # Step 1: Try different numbers of base markers (from minimum to maximum possible)
    for num_base in range(
        min_base_markers, min(len(available_base_markers), num_channels) + 1
    ):
        # Step 2: Try every possible combination of base markers for this count
        for base_panel in map(
            set, itertools.combinations(available_base_markers, num_base)
        ):
            # Step 3: Calculate how many channels are left for tags
            remaining = num_channels - len(base_panel)
            # Step 4: Try every possible combination of tags to fill remaining channels
            for tag_panel in map(
                set, itertools.combinations(available_tags, remaining)
            ):
                # Step 5: Combine base markers and tags into final panel
                panel = base_panel | tag_panel

                # Step 6: Calculate the entropy for this panel configuration
                # First, filter each row to only include markers that are in our panel
                df_temp = df.copy()
                df_temp["filtered_sig"] = df_temp[readout_col].apply(
                    lambda tup: tuple(sorted(set(tup) & panel))  # noqa: B023
                )

                # Step 7: Group identical signatures and sum their probabilities
                # This gives us the probability of each unique cell labeling pattern
                sig_probs = (
                    df_temp.groupby("filtered_sig")["Probability"].sum().to_dict()
                )

                # Step 8: Calculate weighted entropy based on marker types
                weighted_entropy = 0
                for signature, prob in sig_probs.items():
                    # Step 9: Count how many base markers vs tags are in this signature
                    base_marker_count = sum(
                        1 for m in signature if m in config.base_markers_set
                    )
                    tag_count = len(signature) - base_marker_count

                    # Step 10: Apply weights to the probability
                    # Base markers get multiplied by base_marker_weight^count
                    # Tags get multiplied by tag_weight^count
                    weighted_prob = (
                        prob
                        * (base_marker_weight**base_marker_count)
                        * (tag_weight**tag_count)
                    )

                    # Step 11: Add this signature's contribution to total entropy
                    # Using Shannon entropy formula: -p * log2(p)
                    weighted_entropy -= weighted_prob * math.log2(weighted_prob)

                # Step 12: Check if this panel gives better entropy than our current best
                if weighted_entropy > best_score:
                    best_score = weighted_entropy
                    best_panels = [panel]
                    best_base_markers = base_panel
                    best_tags = set(tag_panel)
                elif weighted_entropy == best_score:
                    # Step 13: If tied, keep track of all panels that achieve this score
                    best_panels.append(panel)
                    if base_panel != best_base_markers:
                        best_base_markers = base_panel
                        best_tags = set(tag_panel)

    # ────────────────────────────  package the result  ─────────────────────────
    return {
        "entropy": best_score,
        "best_panels": best_panels,
        "num_channels": num_channels,
        "entropy_per_channel": best_score / num_channels if best_score > 0 else 0.0,
        "available_markers": all_markers,
        "base_markers_used": best_base_markers,
        "tags_used": best_tags,
        "weights_used": {
            "base_marker_weight": base_marker_weight,
            "tag_weight": tag_weight,
        },
    }


def calculate_all_channels_entropy(
    df: pd.DataFrame,
    readout_col: str = "Linear_readout",
    base_marker_weight: float = 1.0,
    tag_weight: float = 0.5,
    config: Optional[BrainbowConfig] = None,
) -> dict:
    """
    Calculate the entropy when all available channels are imaged.

    Args:
        df: DataFrame with outcome data (from outcome_table)
        readout_col: Column name to use for readout analysis (default: "Linear_readout")
        base_marker_weight: Weight multiplier for base markers in entropy calculation (default: 1.0)
        tag_weight: Weight multiplier for tags in entropy calculation (default: 0.5)
        config: BrainbowConfig object defining available markers and tags

    Returns:
        dict: Dictionary containing:
            - 'entropy': float, entropy when all channels are imaged in bits
            - 'num_channels': int, total number of channels used
            - 'entropy_per_channel': float, entropy divided by number of channels
            - 'all_markers': list, all markers that were used
            - 'base_markers': set, base markers present in the dataset
            - 'tags': set, tags present in the dataset
            - 'weights_used': dict, the weights used for marker selection
    """
    all_markers = sorted({m for tup in df[readout_col] for m in tup})

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    if not all_markers:
        return {
            "entropy": 0.0,
            "num_channels": 0,
            "entropy_per_channel": 0.0,
            "all_markers": [],
            "base_markers": set(),
            "tags": set(),
            "weights_used": {
                "base_marker_weight": base_marker_weight,
                "tag_weight": tag_weight,
            },
        }

    # Use all available markers
    panel = set(all_markers)

    # Filter each row to the chosen markers
    df_temp = df.copy()
    df_temp["filtered_sig"] = df_temp[readout_col].apply(
        lambda tup: tuple(sorted(set(tup) & panel))
    )

    # Aggregate probabilities for identical signatures
    sig_probs = df_temp.groupby("filtered_sig")["Probability"].sum().to_dict()

    # Calculate weighted entropy based on marker types
    weighted_entropy = 0
    for signature, prob in sig_probs.items():
        # Count base markers and tags in this signature
        base_marker_count = sum(1 for m in signature if m in config.base_markers_set)
        tag_count = len(signature) - base_marker_count

        # Apply weights to the probability
        weighted_prob = (
            prob * (base_marker_weight**base_marker_count) * (tag_weight**tag_count)
        )
        weighted_entropy -= weighted_prob * math.log2(weighted_prob)

    # Identify which markers are base markers vs tags
    base_markers_present = set(all_markers) & config.base_markers_set
    tags_present = set(all_markers) - config.base_markers_set

    return {
        "entropy": weighted_entropy,
        "num_channels": len(all_markers),
        "entropy_per_channel": weighted_entropy / len(all_markers)
        if all_markers
        else 0.0,
        "all_markers": all_markers,
        "base_markers": base_markers_present,
        "tags": tags_present,
        "weights_used": {
            "base_marker_weight": base_marker_weight,
            "tag_weight": tag_weight,
        },
    }


def calculate_base_marker_expression_percentages(
    df: pd.DataFrame,
    readout_col: str = "Linear_readout",
    config: Optional[BrainbowConfig] = None,
) -> dict:
    """
    Calculate what percentage of cells express each base marker individually.

    Args:
        df: DataFrame with outcome data (from outcome_table)
        readout_col: Column name to use for readout analysis (default: "Linear_readout")
        config: BrainbowConfig object defining available markers and tags

    Returns:
        dict: Dictionary containing:
            - 'expression_percentages': dict, mapping base marker names to their expression percentages
            - 'total_cells': float, total probability (should be 1.0)
            - 'base_markers_found': list, base markers that were found in the dataset
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    # Get all unique markers in the dataset
    all_markers = {m for tup in df[readout_col] for m in tup}
    base_markers_found = sorted(config.base_markers_set & all_markers)

    expression_percentages = {}

    for marker in base_markers_found:
        # Calculate probability that this marker appears in any cell
        marker_probability = 0.0

        for _, row in df.iterrows():
            markers_in_cell = set(row[readout_col])
            if marker in markers_in_cell:
                marker_probability += row["Probability"]

        expression_percentages[marker] = marker_probability * 100

    return {
        "expression_percentages": expression_percentages,
        "total_cells": df["Probability"].sum(),
        "base_markers_found": base_markers_found,
    }


def print_optimal_configuration_report(
    labeling_analysis: dict,
    variables: dict,
    best_panel: set,
    optimal_config: Optional[dict] = None,
    base_markers: Optional[set] = None,
) -> None:
    """
    Print a comprehensive report of the optimal cassette configuration analysis.

    Args:
        labeling_analysis: Dictionary returned by analyze_cell_labeling
        variables: Dictionary containing analysis parameters (e.g., p_cre)
        best_panel: Set of markers in the optimal panel
        optimal_config: Optional dictionary returned by find_optimal_cassette_configuration
    """
    print("=== CELL LABELING ANALYSIS ===")
    print(f"Panel: {sorted(list(best_panel))}")
    print(f"Percent of Cre-labeled cells: {variables['p_cre'] * 100}%")
    print(
        f"Total cells labeled in panel: {labeling_analysis['total_labeled_percent']:.1f}%"
    )
    print(f"Cells not labeled in panel: {labeling_analysis['unlabeled_percent']:.1f}%")
    print(f"Number of unique combinations: {labeling_analysis['num_combinations']}")

    print("\n=== MARKER COMBINATION DISTRIBUTION ===")
    combo_df = labeling_analysis["combination_distribution"]
    for _, row in combo_df.iterrows():
        if len(row["Marker_Combination"]) == 0:
            combo_name = "No markers (unlabeled)"
        else:
            combo_name = " + ".join(sorted(row["Marker_Combination"]))
        print(f"{combo_name}: {row['Percentage']:.1f}%")

    # Only print optimal configuration details if provided
    if optimal_config is not None:
        print("\n=== OPTIMAL CONFIGURATION DETAILS ===")
        print(f"Best entropy: {optimal_config['best_entropy']:.3f} bits")
        if "best_panel_info" in optimal_config:
            print(
                f"Entropy per channel: {optimal_config['best_panel_info']['entropy_per_channel']:.3f} bits"
            )

        print("\nOptimal Cassette A:")
        for state, marker in optimal_config["best_cassette_a"]:
            print(f"  {state}: {marker}")

        print("\nOptimal Cassette B:")
        for state, marker in optimal_config["best_cassette_b"]:
            print(f"  {state}: {marker}")

        print("\n=== JUST BASE MARKERS ===")
        if base_markers is not None:
            panel = base_markers
        else:
            panel = set(optimal_config["config_used"].base_markers)
        df = outcome_table(
            0.8,
            optimal_config["best_cassette_a"],
            optimal_config["best_cassette_b"],
            1,
            1,
            1,
        )
        base_labeling_analysis = analyze_cell_labeling(df, panel)
        print(
            f"Total cells labeled with base markers: {base_labeling_analysis['total_labeled_percent']:.1f}%"
        )
        print(
            f"Cells not labeled with base markers: {base_labeling_analysis['unlabeled_percent']:.1f}%"
        )
        print(
            f"Number of unique base marker combinations: {base_labeling_analysis['num_combinations']}"
        )
        for _, row in base_labeling_analysis["combination_distribution"].iterrows():
            if len(row["Marker_Combination"]) == 0:
                combo_name = "No markers (unlabeled)"
            else:
                combo_name = " + ".join(sorted(row["Marker_Combination"]))
            print(f"{combo_name}: {row['Percentage']:.1f}%")


# --------- 4. Example ----------------------------------------------------
if __name__ == "__main__":
    # Example with default cassettes
    df_default = outcome_table(p_cre=0.6)
    print("Default cassettes:")
    print(df_default)

    # Example with custom cassettes
    custom_cassette_a = create_cassette(
        null_marker="Null",
        bit1_marker="hfYFP-ALFA",
        bit2_marker="mNeonGreen-TD",
        bit3_marker="mScarlet-TD-ALFA",
    )
    custom_cassette_b = create_cassette(
        null_marker="Null",
        bit1_marker="hfYFP-ALFA",
        bit2_marker="mNeonGreen-TD",
        bit3_marker="mScarlet-ALFA-TD",
    )

    df_custom = outcome_table(
        p_cre=0.6, cassette_a=custom_cassette_a, cassette_b=custom_cassette_b
    )
    print("\nCustom cassettes:")
    print(df_custom)

    # Example with custom configuration
    print("\n--- Custom Configuration Example ---")

    # Create a custom configuration with different markers and tags
    custom_config = BrainbowConfig(
        base_markers=["mCherry", "eGFP", "mTurquoise", "mOrange"],
        tags=["HA", "FLAG", "Myc", "V5"],
        max_tags_per_marker=1,
    )

    # Find optimal configuration with custom markers
    result = find_optimal_cassette_configuration(
        num_configurations=100,  # Small number for demo
        p_cre=0.7,
        target_channels=4,
        verbose=True,
        config=custom_config,
    )

    print(f"Best entropy found: {result['best_entropy']:.3f} bits")
    print(
        f"Config used: {result['config_used'].base_markers} base markers, {result['config_used'].tags} tags"
    )

    # Example with default configuration
    print("\n--- Default Configuration Example ---")
    default_result = find_optimal_cassette_configuration(
        num_configurations=100,  # Small number for demo
        p_cre=0.7,
        target_channels=4,
        verbose=True,
        # Uses DEFAULT_CONFIG automatically
    )

    print(f"Best entropy found: {default_result['best_entropy']:.3f} bits")
    print(
        f"Config used: {default_result['config_used'].base_markers} base markers, {default_result['config_used'].tags} tags"
    )

    # # Full report with optimal configuration
    # print_optimal_configuration_report(labeling_analysis, variables, best_panel, result)

    # # Just panel labeling information (no optimal config needed)
    # print_optimal_configuration_report(labeling_analysis, variables, best_panel)

    # # For any arbitrary panel analysis
    # arbitrary_panel = {"mScarlet3", "hfYFP", "ALFA"}
    # arbitrary_labeling = analyze_cell_labeling(df_custom, arbitrary_panel)
    # print_optimal_configuration_report(arbitrary_labeling, variables, arbitrary_panel)
