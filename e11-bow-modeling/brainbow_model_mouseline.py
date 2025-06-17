import itertools
import math
from typing import Union, Optional

import pandas as pd


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
) -> pd.DataFrame:
    """
    Generate outcome table for two cassettes.

    Args:
        p_cre: fraction of cells that see Cre (same for both cassettes)
              -> P(null)   = 1-p_cre
              -> P(bit1/2/3) each = p_cre / 3
        cassette_a: List of (state_label, protein) tuples for cassette A
        cassette_b: List of (state_label, protein) tuples for cassette B

    Returns:
        DataFrame with all possible outcomes and their probabilities
    """
    # Use default cassettes if none provided
    if cassette_a is None:
        cassette_a = create_cassette()
    if cassette_b is None:
        cassette_b = create_cassette()

    rows = []

    for (lab_a, prot_a), (lab_b, prot_b) in itertools.product(cassette_a, cassette_b):
        #   probability for each cassette
        p_a = (1 - p_cre) if lab_a == "Null" else (p_cre / 3)
        p_b = (1 - p_cre) if lab_b == "Null" else (p_cre / 3)

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
                "Probability": round(p_a * p_b, 4),
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
            lambda tup: tuple(sorted(set(tup) & panel))
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
            lambda tup: tuple(sorted(set(tup) & panel))
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
    base_markers = {"mNeonGreen", "hfYFP", "mScarlet3", "mBaojin"}

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
            lambda tup: tuple(sorted(set(tup) & panel))
        )

        # Aggregate probabilities for identical signatures
        sig_probs = df_temp.groupby("filtered_sig")["Probability"].sum().to_dict()

        # Calculate weighted entropy based on marker types
        weighted_entropy = 0
        for signature, prob in sig_probs.items():
            # Count base markers and tags in this signature
            base_marker_count = sum(1 for m in signature if m in base_markers)
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
    num_configurations: int = 1000, seed: Optional[int] = None
) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """
    Generate shuffled cassette configurations following the specified rules.

    Rules:
    - Each element must have one and only one base marker: ["mNeonGreen", "hfYFP", "mScarlet3", "mBaojin"]
    - Each element can have 0-2 additional tags from: ["HSV", "TD", "ALFA", "NWS"]

    Args:
        num_configurations: Number of random configurations to generate
        seed: Random seed for reproducibility

    Returns:
        List of tuples, each containing (cassette_a, cassette_b) configurations
    """
    import random

    if seed is not None:
        random.seed(seed)

    base_markers = ["mNeonGreen", "hfYFP", "mScarlet3", "mBaojin"]
    tags = ["HSV", "TD", "ALFA", "NWS"]

    configurations = []

    for _ in range(num_configurations):
        # Generate cassette A
        cassette_a = []
        for state in ["Null", "Bit1", "Bit2", "Bit3"]:
            # Choose one base marker
            base_marker = random.choice(base_markers)

            # Choose 0-2 additional tags
            num_tags = random.randint(0, 2)
            selected_tags = random.sample(tags, num_tags)

            # Combine base marker with tags
            if selected_tags:
                marker = base_marker + "-" + "-".join(selected_tags)
            else:
                marker = base_marker

            cassette_a.append((state, marker))

        # Generate cassette B
        cassette_b = []
        for state in ["Null", "Bit1", "Bit2", "Bit3"]:
            # Choose one base marker
            base_marker = random.choice(base_markers)

            # Choose 0-2 additional tags
            num_tags = random.randint(0, 2)
            selected_tags = random.sample(tags, num_tags)

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

    Returns:
        Dictionary containing:
            - 'best_entropy': float, highest entropy found
            - 'best_cassette_a': list, optimal cassette A configuration
            - 'best_cassette_b': list, optimal cassette B configuration
            - 'best_df': DataFrame, outcome table for best configuration
            - 'best_panel_info': dict, panel information if target_channels specified
            - 'configurations_tested': int, number of configurations tested
            - 'weights_used': dict, the weights used for marker selection
    """
    import random

    if seed is not None:
        random.seed(seed)

    configurations = generate_shuffled_cassettes(
        num_configurations=num_configurations, seed=seed
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
    min_base_markers: int = 2,  # Minimum number of base markers to include
) -> dict:
    """
    Find the best panel using a two-step process:
    1. First select optimal base markers
    2. Then add tags on top of those base markers

    Args:
        df: DataFrame with outcome data (from outcome_table)
        num_channels: Total number of channels/markers to use in the panel
        readout_col: Column name to use for readout analysis
        base_marker_weight: Weight multiplier for base markers in entropy calculation
        tag_weight: Weight multiplier for tags in entropy calculation
        min_base_markers: Minimum number of base markers to include in the panel

    Returns:
        dict: Dictionary containing:
            - 'entropy': float, entropy of the best panel in bits
            - 'best_panels': list of sets, all panels that achieve the best entropy
            - 'num_channels': int, number of channels used
            - 'entropy_per_channel': float, entropy divided by number of channels
            - 'available_markers': list, all available markers in the dataset
            - 'base_markers_used': set, the base markers selected in step 1
            - 'tags_used': set, the tags added in step 2
            - 'weights_used': dict, the weights used for marker selection
    """
    all_markers = sorted({m for tup in df[readout_col] for m in tup})
    base_markers = {"mNeonGreen", "hfYFP", "mScarlet3", "mBaojin"}
    available_base_markers = base_markers & set(all_markers)
    available_tags = set(all_markers) - base_markers

    if num_channels > len(all_markers):
        raise ValueError(
            f"Number of channels ({num_channels}) is greater than available markers ({len(all_markers)})"
        )

    if min_base_markers > len(available_base_markers):
        raise ValueError(
            f"Minimum base markers ({min_base_markers}) is greater than available base markers ({len(available_base_markers)})"
        )

    best_score = -1
    best_panels = []
    best_base_markers = None
    best_tags = None

    # Step 1: Find optimal base markers
    for num_base in range(
        min_base_markers, min(len(available_base_markers), num_channels) + 1
    ):
        for base_panel in itertools.combinations(available_base_markers, num_base):
            base_panel = set(base_panel)

            # Step 2: Add tags to complete the panel
            remaining_channels = num_channels - len(base_panel)
            if remaining_channels > 0:
                for tag_panel in itertools.combinations(
                    available_tags, remaining_channels
                ):
                    panel = base_panel | set(tag_panel)

                    # Filter each row to the chosen markers
                    df_temp = df.copy()
                    df_temp["filtered_sig"] = df_temp[readout_col].apply(
                        lambda tup: tuple(sorted(set(tup) & panel))
                    )

                    # Aggregate probabilities for identical signatures
                    sig_probs = (
                        df_temp.groupby("filtered_sig")["Probability"].sum().to_dict()
                    )

                    # Calculate weighted entropy based on marker types
                    weighted_entropy = 0
                    for signature, prob in sig_probs.items():
                        # Count base markers and tags in this signature
                        base_marker_count = sum(
                            1 for m in signature if m in base_markers
                        )
                        tag_count = len(signature) - base_marker_count

                        # Apply weights to the probability
                        weighted_prob = (
                            prob
                            * (base_marker_weight**base_marker_count)
                            * (tag_weight**tag_count)
                        )
                        weighted_entropy -= weighted_prob * math.log2(weighted_prob)

                    if weighted_entropy > best_score:
                        best_score = weighted_entropy
                        best_panels = [panel]
                        best_base_markers = base_panel
                        best_tags = set(tag_panel)
                    elif weighted_entropy == best_score:
                        best_panels.append(panel)
                        if base_panel != best_base_markers:
                            best_base_markers = base_panel
                            best_tags = set(tag_panel)

    return {
        "entropy": best_score,
        "best_panels": best_panels,
        "num_channels": num_channels,
        "entropy_per_channel": best_score / num_channels,
        "available_markers": all_markers,
        "base_markers_used": best_base_markers,
        "tags_used": best_tags,
        "weights_used": {
            "base_marker_weight": base_marker_weight,
            "tag_weight": tag_weight,
        },
    }


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
