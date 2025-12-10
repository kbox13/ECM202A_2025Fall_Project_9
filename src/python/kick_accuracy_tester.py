#!/usr/bin/env python3
"""
Kick Timing Accuracy Tester

Evaluates the accuracy of kick timing hits from hits_predictions log and MQTT commands log
against ground truth data. Uses a shared alignment offset calculated from hits_predictions.
"""

import json
import argparse
import statistics
import sys
import csv
from typing import List, Tuple, Optional


def extract_ground_truth_timings(json_path: str) -> List[float]:
    """
    Load and extract kick timings array from ground truth JSON.
    
    Args:
        json_path: Path to ground truth JSON file
        
    Returns:
        Sorted list of timestamps
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract kick array from nested structure
        # Structure: {"song_name": {"kick": [timestamps]}}
        if not isinstance(data, dict):
            raise ValueError("Ground truth JSON must be a dictionary")
        
        # Find the first key that contains a "kick" key
        kick_timings = None
        for song_name, song_data in data.items():
            if isinstance(song_data, dict) and "kick" in song_data:
                kick_timings = song_data["kick"]
                break
        
        if kick_timings is None:
            raise ValueError("Could not find 'kick' array in ground truth JSON")
        
        if not isinstance(kick_timings, list):
            raise ValueError("'kick' must be an array")
        
        # Convert to floats and sort
        timings = sorted([float(t) for t in kick_timings])
        
        if len(timings) == 0:
            print(f"Warning: Ground truth contains no kick timings", file=sys.stderr)
        
        return timings
    
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in ground truth file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading ground truth: {e}", file=sys.stderr)
        sys.exit(1)


def extract_hits_predictions_timings(log_path: str) -> List[float]:
    """
    Parse hits_predictions log and extract kick hit timings.
    
    Args:
        log_path: Path to hits_predictions log file
        
    Returns:
        Sorted list of audio_time values for kick hits
    """
    try:
        timings = []
        
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                try:
                    entry = json.loads(line)
                    # Filter for kick hits
                    if (entry.get("type") == "hit" and 
                        entry.get("instrument") == "kick" and
                        "audio_time" in entry):
                        timings.append(float(entry["audio_time"]))
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
        
        # Sort by audio_time
        timings.sort()
        
        if len(timings) == 0:
            print(f"Warning: No kick hits found in hits_predictions log", file=sys.stderr)
        
        return timings
    
    except FileNotFoundError:
        print(f"Error: hits_predictions log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading hits_predictions log: {e}", file=sys.stderr)
        sys.exit(1)


def extract_mqtt_commands_timings(log_path: str) -> List[float]:
    """
    Parse MQTT commands log and extract kick tPredSec array.
    
    Args:
        log_path: Path to MQTT commands log file
        
    Returns:
        Sorted list of tPredSec values for kick instrument
    """
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                try:
                    entry = json.loads(line)
                    # Find kick instrument line
                    if (entry.get("instrument") == "kick" and 
                        "tPredSec" in entry):
                        t_pred_sec = entry["tPredSec"]
                        if not isinstance(t_pred_sec, list):
                            raise ValueError("tPredSec must be an array")
                        
                        # Convert to floats and sort
                        timings = sorted([float(t) for t in t_pred_sec])
                        
                        if len(timings) == 0:
                            print(f"Warning: MQTT commands log contains no kick timings", file=sys.stderr)
                        
                        return timings
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
        
        # If we get here, no kick line was found
        print(f"Warning: No kick instrument found in MQTT commands log", file=sys.stderr)
        return []
    
    except FileNotFoundError:
        print(f"Error: MQTT commands log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading MQTT commands log: {e}", file=sys.stderr)
        sys.exit(1)


def calculate_alignment_offset(times: List[float], reference_times: List[float]) -> float:
    """
    Calculate alignment offset to align times to reference_times by first event.
    
    Args:
        times: Timestamp list to be aligned
        reference_times: Reference timestamp list
        
    Returns:
        Offset value such that times[0] + offset = reference_times[0]
    """
    if len(times) == 0 or len(reference_times) == 0:
        raise ValueError("Cannot calculate alignment offset for empty timestamp lists")
    
    return reference_times[0] - times[0]


def align_timestamps(times1: List[float], times2: List[float]) -> Tuple[List[float], float]:
    """
    Align two timestamp lists by first event.
    
    Args:
        times1: First timestamp list (will be aligned)
        times2: Second timestamp list (reference)
        
    Returns:
        Tuple of (aligned_times1, offset) where aligned_times1 = [t + offset for t in times1]
    """
    offset = calculate_alignment_offset(times1, times2)
    aligned_times1 = apply_alignment_offset(times1, offset)
    return aligned_times1, offset


def apply_alignment_offset(times: List[float], offset: float) -> List[float]:
    """
    Apply a pre-calculated offset to a timestamp list.
    
    Args:
        times: Timestamp list
        offset: Offset to apply
        
    Returns:
        Aligned timestamps = [t + offset for t in times]
    """
    return [t + offset for t in times]


def _match_timestamps(
    aligned_times1: List[float],
    original_times1: List[float],
    aligned_times2: List[float],
    original_times2: List[float],
    tolerance: float
) -> Tuple[List[Tuple[float, float, float]], List[int], List[int]]:
    """
    Core matching algorithm: find nearest-neighbor matches between two aligned timestamp lists.
    
    Args:
        aligned_times1: First timestamp list (aligned)
        original_times1: Original timestamps for first list (for return values)
        aligned_times2: Second timestamp list (aligned)
        original_times2: Original timestamps for second list (for return values)
        tolerance: Maximum distance for a match (in seconds)
        
    Returns:
        Tuple of (matched_pairs, unmatched_indices1, unmatched_indices2)
        matched_pairs: List of (original_time1, original_time2, error) tuples where error = aligned_time1 - aligned_time2
        unmatched_indices1: List of indices from times1 that didn't match
        unmatched_indices2: List of indices from times2 that didn't match
    """
    if len(aligned_times1) == 0 or len(aligned_times2) == 0:
        unmatched_indices1 = list(range(len(original_times1)))
        unmatched_indices2 = list(range(len(original_times2)))
        return [], unmatched_indices1, unmatched_indices2
    
    # Sort both lists for efficient matching
    aligned_times1_sorted = sorted(enumerate(aligned_times1), key=lambda x: x[1])
    aligned_times2_sorted = sorted(enumerate(aligned_times2), key=lambda x: x[1])
    
    matched_pairs = []
    matched_indices1 = set()
    matched_indices2 = set()
    
    # For each time1, find nearest time2
    for idx1, aligned_time1 in aligned_times1_sorted:
        best_idx2 = None
        best_distance = float('inf')
        
        for idx2, aligned_time2 in aligned_times2_sorted:
            # Skip if already matched
            if idx2 in matched_indices2:
                continue
            
            distance = abs(aligned_time1 - aligned_time2)
            
            # If we're getting further away, we can stop (lists are sorted)
            if distance > best_distance:
                break
            
            if distance < best_distance:
                best_distance = distance
                best_idx2 = idx2
        
        # If within tolerance, create match
        if best_idx2 is not None and best_distance <= tolerance:
            original_time1 = original_times1[idx1]
            original_time2 = original_times2[best_idx2]
            aligned_time2 = aligned_times2[best_idx2]
            error = aligned_time1 - aligned_time2
            matched_pairs.append((original_time1, original_time2, error))
            matched_indices1.add(idx1)
            matched_indices2.add(best_idx2)
    
    # Find unmatched indices
    unmatched_indices1 = [i for i in range(len(original_times1)) if i not in matched_indices1]
    unmatched_indices2 = [i for i in range(len(original_times2)) if i not in matched_indices2]
    
    return matched_pairs, unmatched_indices1, unmatched_indices2


def compare_timestamp_lists(
    times1: List[float],
    times2: List[float],
    tolerance: float,
    times1_offset: Optional[float] = None,
    times2_offset: Optional[float] = None,
    reference_times: Optional[List[float]] = None
) -> Tuple[List[Tuple[float, float, float]], List[float], List[float], float]:
    """
    Unified function to compare two timestamp lists with optional alignment offsets.
    
    This function replaces both compare_timestamps() and compare_test_sets_direct().
    
    Args:
        times1: First timestamp list
        times2: Second timestamp list (reference for alignment if reference_times is None)
        tolerance: Maximum distance for a match (in seconds)
        times1_offset: Optional pre-calculated alignment offset for times1
        times2_offset: Optional pre-calculated alignment offset for times2
        reference_times: Optional reference timestamps for alignment calculation.
                        If provided, both times1 and times2 are aligned to this reference.
                        If None, times1 is aligned to times2.
        
    Returns:
        Tuple of (matched_pairs, unmatched_times1, unmatched_times2, alignment_offset)
        matched_pairs: List of (time1, time2, error) tuples where error = aligned_time1 - aligned_time2
        unmatched_times1: List of times1 timestamps that didn't match
        unmatched_times2: List of times2 timestamps that didn't match
        alignment_offset: The offset used for times1 alignment (or times2_offset if both offsets provided)
    """
    if len(times1) == 0 or len(times2) == 0:
        return [], times1.copy(), times2.copy(), 0.0
    
    # Determine alignment strategy
    if reference_times is not None and len(reference_times) > 0:
        # Both times1 and times2 should be aligned to reference_times
        if times1_offset is None:
            times1_offset = calculate_alignment_offset(times1, reference_times)
        if times2_offset is None:
            times2_offset = calculate_alignment_offset(times2, reference_times)
        aligned_times1 = apply_alignment_offset(times1, times1_offset)
        aligned_times2 = apply_alignment_offset(times2, times2_offset)
        offset = times1_offset  # Return times1 offset for consistency
    elif times1_offset is not None and times2_offset is not None:
        # Both offsets provided - use them directly (matches compare_test_sets_direct behavior)
        aligned_times1 = apply_alignment_offset(times1, times1_offset)
        aligned_times2 = apply_alignment_offset(times2, times2_offset)
        offset = times2_offset  # Return times2 offset to match original behavior
    elif times1_offset is not None:
        # Only times1 offset provided - align times2 to aligned times1
        aligned_times1 = apply_alignment_offset(times1, times1_offset)
        aligned_times2, _ = align_timestamps(times2, aligned_times1)
        offset = times1_offset
    elif times2_offset is not None:
        # Only times2 offset provided - align times1 to aligned times2
        aligned_times2 = apply_alignment_offset(times2, times2_offset)
        aligned_times1, offset = align_timestamps(times1, aligned_times2)
    else:
        # No offsets provided - align times1 to times2 (default behavior)
        aligned_times1, offset = align_timestamps(times1, times2)
        aligned_times2 = times2
    
    # Use core matching algorithm
    matched_pairs, unmatched_indices1, unmatched_indices2 = _match_timestamps(
        aligned_times1, times1, aligned_times2, times2, tolerance
    )
    
    # Convert unmatched indices to timestamps
    unmatched_times1 = [times1[i] for i in unmatched_indices1]
    unmatched_times2 = [times2[i] for i in unmatched_indices2]
    
    return matched_pairs, unmatched_times1, unmatched_times2, offset


# Backward compatibility aliases
def compare_timestamps(
    test_set_times: List[float], 
    ground_truth_times: List[float], 
    tolerance: float,
    alignment_offset: Optional[float] = None
) -> Tuple[List[Tuple[float, float, float]], List[float], List[float], float]:
    """
    Compare test set timestamps to ground truth timestamps.
    
    This is a compatibility wrapper around compare_timestamp_lists().
    
    Args:
        test_set_times: List of test set timestamps
        ground_truth_times: List of ground truth timestamps
        tolerance: Maximum distance for a match (in seconds)
        alignment_offset: Optional pre-calculated alignment offset. If None, calculates from first events.
        
    Returns:
        Tuple of (matched_pairs, unmatched_test_set, unmatched_ground_truth, alignment_offset)
        matched_pairs: List of (test_set, ground_truth, error) tuples
        unmatched_test_set: List of test set timestamps that didn't match
        unmatched_ground_truth: List of ground truth timestamps that didn't match
        alignment_offset: The offset used for alignment
    """
    return compare_timestamp_lists(
        test_set_times, ground_truth_times, tolerance,
        times1_offset=alignment_offset
    )


def compare_test_sets_direct(
    test_set1_times: List[float],
    test_set2_times: List[float],
    tolerance: float,
    test_set1_alignment_offset: Optional[float] = None,
    test_set2_alignment_offset: Optional[float] = None
) -> Tuple[List[Tuple[float, float, float]], List[float], List[float], float]:
    """
    Compare two test sets directly (no ground truth).
    
    This is a compatibility wrapper around compare_timestamp_lists().
    
    Args:
        test_set1_times: First test set timestamps
        test_set2_times: Second test set timestamps
        tolerance: Maximum distance for a match (in seconds)
        test_set1_alignment_offset: Optional alignment offset for test_set1 (aligns to ground truth)
        test_set2_alignment_offset: Optional alignment offset for test_set2 (aligns to ground truth)
        
    Returns:
        Tuple of (matched_pairs, unmatched_test_set1, unmatched_test_set2, alignment_offset)
        matched_pairs: List of (test_set1, test_set2, error) tuples where error = test_set1_aligned - test_set2_aligned
        unmatched_test_set1: List of test_set1 timestamps that didn't match
        unmatched_test_set2: List of test_set2 timestamps that didn't match
        alignment_offset: The offset used (returns test_set1_alignment_offset if provided, else calculated)
    """
    return compare_timestamp_lists(
        test_set1_times, test_set2_times, tolerance,
        times1_offset=test_set1_alignment_offset,
        times2_offset=test_set2_alignment_offset
    )


def compute_metrics(
    matched_pairs: List[Tuple[float, float, float]],
    unmatched_test_set: List[float],
    unmatched_reference: List[float],
    tolerance: float
) -> dict:
    """
    Compute accuracy metrics from comparison results.
    
    Args:
        matched_pairs: List of (test_set, reference, error) tuples
        unmatched_test_set: List of unmatched test set timestamps
        unmatched_reference: List of unmatched reference timestamps
        tolerance: Tolerance used for matching
        
    Returns:
        Dictionary of metrics
    """
    total_test_set = len(matched_pairs) + len(unmatched_test_set)
    total_reference = len(matched_pairs) + len(unmatched_reference)
    matched_count = len(matched_pairs)
    
    # Calculate precision, recall, F1
    precision = matched_count / total_test_set if total_test_set > 0 else 0.0
    recall = matched_count / total_reference if total_reference > 0 else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # Calculate timing errors
    if matched_count > 0:
        errors = [abs(error) for _, _, error in matched_pairs]
        mean_absolute_error = statistics.mean(errors)
        std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
        max_absolute_error = max(errors)
        min_error = min(errors)
    else:
        mean_absolute_error = 0.0
        std_error = 0.0
        max_absolute_error = 0.0
        min_error = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_count": matched_count,
        "total_test_set": total_test_set,
        "total_reference": total_reference,
        "unmatched_test_set_count": len(unmatched_test_set),
        "unmatched_reference_count": len(unmatched_reference),
        "mean_absolute_error": mean_absolute_error,
        "std_error": std_error,
        "max_absolute_error": max_absolute_error,
        "min_error": min_error,
        "tolerance": tolerance
    }


def print_metrics_summary(metrics: dict, label: str, reference_label: str = "ground truth"):
    """Print a summary of metrics to console."""
    print(f"\n{label} Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Matched: {metrics['matched_count']} / {metrics['total_test_set']} test set, "
          f"{metrics['matched_count']} / {metrics['total_reference']} {reference_label}")
    print(f"  Mean Absolute Error: {metrics['mean_absolute_error']*1000:.2f} ms")
    print(f"  Std Error: {metrics['std_error']*1000:.2f} ms")
    print(f"  Max Absolute Error: {metrics['max_absolute_error']*1000:.2f} ms")
    print(f"  Min Error: {metrics['min_error']*1000:.2f} ms")


def build_ordered_pairings(
    matched_pairs: List[Tuple[float, float, float]],
    unmatched_hits: List[float],
    unmatched_mqtt: List[float],
    hits_alignment_offset: float,
    mqtt_alignment_offset: float
) -> List[Tuple[Optional[float], Optional[float], Optional[float], float, float]]:
    """
    Build ordered pairings list with found and unfound pairs.
    
    Args:
        matched_pairs: List of (hits_time, mqtt_time, error) tuples
        unmatched_hits: List of unmatched hits_predictions timestamps
        unmatched_mqtt: List of unmatched MQTT commands timestamps
        hits_alignment_offset: Alignment offset for hits_predictions
        mqtt_alignment_offset: Alignment offset for MQTT commands
        
    Returns:
        List of (hits_time, mqtt_time, error, hits_aligned_time, mqtt_aligned_time) tuples, sorted by hits_aligned_time
        - For matched pairs: both times present, error calculated, both aligned times
        - For unmatched hits: hits_time present, mqtt_time=None, error=None, hits_aligned_time, mqtt_aligned_time=None
        - For unmatched mqtt: hits_time=None, mqtt_time present, error=None, hits_aligned_time=None, mqtt_aligned_time
    """
    pairings = []
    
    # Add matched pairs
    for hits_time, mqtt_time, error in matched_pairs:
        # Calculate aligned times for both
        aligned_hits = hits_time + hits_alignment_offset
        aligned_mqtt = mqtt_time + mqtt_alignment_offset
        pairings.append((hits_time, mqtt_time, error, aligned_hits, aligned_mqtt))
    
    # Add unmatched hits (use "x" placeholder for mqtt)
    for hits_time in unmatched_hits:
        aligned_hits = hits_time + hits_alignment_offset
        pairings.append((hits_time, None, None, aligned_hits, None))
    
    # Add unmatched mqtt (use "x" placeholder for hits)
    for mqtt_time in unmatched_mqtt:
        aligned_mqtt = mqtt_time + mqtt_alignment_offset
        pairings.append((None, mqtt_time, None, None, aligned_mqtt))
    
    # Sort by hits_aligned_time (or mqtt_aligned_time if hits is None)
    def sort_key(x):
        hits_aligned, mqtt_aligned = x[3], x[4]
        if hits_aligned is not None:
            return hits_aligned
        elif mqtt_aligned is not None:
            return mqtt_aligned
        else:
            return float('inf')
    
    pairings.sort(key=sort_key)
    
    return pairings


def compare_test_set_to_ground_truth(
    test_set_times: List[float],
    ground_truth_times: List[float],
    tolerance: float,
    alignment_offset: Optional[float] = None,
    test_set_name: str = "test set"
) -> Tuple[dict, Optional[float]]:
    """
    Compare a test set to ground truth and return results dictionary.
    
    Args:
        test_set_times: Test set timestamps
        ground_truth_times: Ground truth timestamps
        tolerance: Maximum distance for a match (in seconds)
        alignment_offset: Optional pre-calculated alignment offset
        test_set_name: Name of the test set for error messages
        
    Returns:
        Tuple of (results_dict, alignment_offset)
        results_dict: Dictionary with comparison results or error message
        alignment_offset: The offset used for alignment
    """
    if len(test_set_times) == 0 or len(ground_truth_times) == 0:
        return {"error": "Empty timings - cannot compare"}, None
    
    matched_pairs, unmatched_test_set, unmatched_gt, offset = compare_timestamps(
        test_set_times, ground_truth_times, tolerance, alignment_offset
    )
    
    metrics = compute_metrics(matched_pairs, unmatched_test_set, unmatched_gt, tolerance)
    
    return {
        "alignment_offset": offset,
        "metrics": metrics,
        "matched_pairs": [
            {"test_set": ts, "ground_truth": gt, "error": e}
            for ts, gt, e in matched_pairs
        ],
        "unmatched_test_set": unmatched_test_set,
        "unmatched_ground_truth": unmatched_gt
    }, offset


def compare_two_test_sets(
    test_set1_times: List[float],
    test_set2_times: List[float],
    tolerance: float,
    test_set1_alignment_offset: Optional[float],
    test_set2_alignment_offset: Optional[float],
    test_set1_name: str = "test set 1",
    test_set2_name: str = "test set 2"
) -> Tuple[dict, Optional[List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]]]:
    """
    Compare two test sets directly and return results dictionary with ordered pairings.
    
    Args:
        test_set1_times: First test set timestamps
        test_set2_times: Second test set timestamps
        tolerance: Maximum distance for a match (in seconds)
        test_set1_alignment_offset: Optional alignment offset for test_set1
        test_set2_alignment_offset: Optional alignment offset for test_set2
        test_set1_name: Name of first test set
        test_set2_name: Name of second test set
        
    Returns:
        Tuple of (results_dict, ordered_pairings)
        results_dict: Dictionary with comparison results or error message
        ordered_pairings: List of ordered pairings for CSV export, or None if error
    """
    if len(test_set1_times) == 0 or len(test_set2_times) == 0:
        return {"error": "Empty timings - cannot compare"}, None
    
    matched_pairs, unmatched_test_set1, unmatched_test_set2, offset = compare_test_sets_direct(
        test_set1_times, test_set2_times, tolerance,
        test_set1_alignment_offset, test_set2_alignment_offset
    )
    
    metrics = compute_metrics(matched_pairs, unmatched_test_set2, unmatched_test_set1, tolerance)
    
    # Build ordered pairings for CSV export
    ordered_pairings = None
    if test_set1_alignment_offset is not None and test_set2_alignment_offset is not None:
        ordered_pairings = build_ordered_pairings(
            matched_pairs, unmatched_test_set1, unmatched_test_set2,
            test_set1_alignment_offset, test_set2_alignment_offset
        )
    elif test_set1_alignment_offset is not None:
        # Use test_set1 offset for both if only one is available
        ordered_pairings = build_ordered_pairings(
            matched_pairs, unmatched_test_set1, unmatched_test_set2,
            test_set1_alignment_offset, test_set1_alignment_offset
        )
    elif test_set2_alignment_offset is not None:
        # Use test_set2 offset for both if only one is available
        ordered_pairings = build_ordered_pairings(
            matched_pairs, unmatched_test_set1, unmatched_test_set2,
            test_set2_alignment_offset, test_set2_alignment_offset
        )
    else:
        # Fallback: use calculated offset
        ordered_pairings = build_ordered_pairings(
            matched_pairs, unmatched_test_set1, unmatched_test_set2,
            0.0, offset
        )
    
    return {
        "alignment_offset": offset,
        "metrics": metrics,
        "matched_pairs": [
            {"test_set1": t1, "test_set2": t2, "error": e}
            for t1, t2, e in matched_pairs
        ],
        "unmatched_test_set1": unmatched_test_set1,
        "unmatched_test_set2": unmatched_test_set2
    }, ordered_pairings


def export_aligned_data_to_csv(
    csv_path: str,
    ground_truth: List[float],
    hits_timings: Optional[List[float]],
    mqtt_timings: Optional[List[float]],
    hits_alignment_offset: Optional[float],
    mqtt_alignment_offset: Optional[float],
    ordered_pairings: Optional[List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]] = None
):
    """
    Export aligned data sets to CSV file with optional ordered pairings section.
    
    Args:
        csv_path: Path to output CSV file
        ground_truth: Ground truth timestamps (sorted, no shift)
        hits_timings: Hits predictions timestamps (original, will be aligned)
        mqtt_timings: MQTT commands timestamps (original, will be aligned)
        hits_alignment_offset: Alignment offset for hits_predictions
        mqtt_alignment_offset: Alignment offset for MQTT commands
        ordered_pairings: Optional ordered pairings list from hits_predictions vs mqtt_commands comparison
    """
    # Align the timestamps
    aligned_hits = None
    aligned_mqtt = None
    
    if hits_timings is not None and hits_alignment_offset is not None:
        aligned_hits = apply_alignment_offset(hits_timings, hits_alignment_offset)
    
    if mqtt_timings is not None and mqtt_alignment_offset is not None:
        aligned_mqtt = apply_alignment_offset(mqtt_timings, mqtt_alignment_offset)
    
    # Create sorted lists for each source
    sorted_ground_truth = sorted(ground_truth) if ground_truth else []
    sorted_aligned_hits = sorted(aligned_hits) if aligned_hits else []
    sorted_aligned_mqtt = sorted(aligned_mqtt) if aligned_mqtt else []
    
    # Find maximum length to determine number of rows
    max_length = max(
        len(sorted_ground_truth),
        len(sorted_aligned_hits),
        len(sorted_aligned_mqtt)
    )
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Section 1: Aligned Data Sets
        writer.writerow(['ground_truth', 'hits_predictions_aligned', 'mqtt_commands_aligned'])
        
        # Write data rows - each row shows timestamps from all sources at similar positions
        for i in range(max_length):
            row = []
            
            # Ground truth
            if i < len(sorted_ground_truth):
                row.append(f"{sorted_ground_truth[i]:.6f}")
            else:
                row.append("")
            
            # Hits predictions (aligned)
            if i < len(sorted_aligned_hits):
                row.append(f"{sorted_aligned_hits[i]:.6f}")
            else:
                row.append("")
            
            # MQTT commands (aligned)
            if i < len(sorted_aligned_mqtt):
                row.append(f"{sorted_aligned_mqtt[i]:.6f}")
            else:
                row.append("")
            
            writer.writerow(row)
        
        # Section 2: Ordered Pairings (if provided)
        if ordered_pairings is not None:
            # Empty separator row
            writer.writerow([])
            
            # Header for ordered pairings
            writer.writerow(['hits_predictions', 'mqtt_commands', 'error', 'hits_predictions_aligned', 'mqtt_commands_aligned'])
            
            # Write ordered pairings
            for hits_time, mqtt_time, error, hits_aligned, mqtt_aligned in ordered_pairings:
                row = []
                
                # Hits predictions (original)
                if hits_time is not None:
                    row.append(f"{hits_time:.6f}")
                else:
                    row.append("x")
                
                # MQTT commands (original)
                if mqtt_time is not None:
                    row.append(f"{mqtt_time:.6f}")
                else:
                    row.append("x")
                
                # Error
                if error is not None:
                    row.append(f"{error:.6f}")
                else:
                    row.append("x")
                
                # Hits predictions aligned
                if hits_aligned is not None:
                    row.append(f"{hits_aligned:.6f}")
                else:
                    row.append("x")
                
                # MQTT commands aligned
                if mqtt_aligned is not None:
                    row.append(f"{mqtt_aligned:.6f}")
                else:
                    row.append("x")
                
                writer.writerow(row)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate kick timing accuracy from logs against ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--hits-log",
        help="Path to hits_predictions log file"
    )
    parser.add_argument(
        "--mqtt-log",
        help="Path to MQTT commands log file"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Tolerance for matching (in seconds, default: 0.1)"
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON report file (optional)"
    )
    parser.add_argument(
        "--csv-output",
        help="Path to output CSV file with aligned data sets (optional)"
    )
    
    args = parser.parse_args()
    
    # At least one log file must be provided
    if not args.hits_log and not args.mqtt_log:
        print("Error: At least one of --hits-log or --mqtt-log must be provided", file=sys.stderr)
        sys.exit(1)
    
    # Load ground truth
    print("Loading ground truth...")
    ground_truth = extract_ground_truth_timings(args.ground_truth)
    print(f"  Loaded {len(ground_truth)} ground truth kick timings")
    
    results = {}
    shared_alignment_offset = None
    hits_alignment_offset = None
    mqtt_alignment_offset = None
    hits_timings = None
    mqtt_timings = None
    
    # Process hits_predictions log
    if args.hits_log:
        print(f"\nProcessing hits_predictions log: {args.hits_log}")
        hits_timings = extract_hits_predictions_timings(args.hits_log)
        print(f"  Loaded {len(hits_timings)} kick hits")
        
        result, offset = compare_test_set_to_ground_truth(
                hits_timings, ground_truth, args.tolerance
            )
        results["hits_predictions_vs_ground_truth"] = result
        
        if "error" not in result:
            shared_alignment_offset = offset
            hits_alignment_offset = offset
            print_metrics_summary(result["metrics"], "Hits Predictions vs Ground Truth")
        else:
            print(f"  {result['error']}", file=sys.stderr)
    
    # Process MQTT commands log
    if args.mqtt_log:
        print(f"\nProcessing MQTT commands log: {args.mqtt_log}")
        mqtt_timings = extract_mqtt_commands_timings(args.mqtt_log)
        print(f"  Loaded {len(mqtt_timings)} kick predictions")
        
        if shared_alignment_offset is None:
                print("  Warning: No alignment offset available. MQTT log requires --hits-log for alignment.", file=sys.stderr)
                print("  Calculating independent alignment offset for MQTT...", file=sys.stderr)
            else:
            print(f"  Using shared alignment offset: {shared_alignment_offset:.6f} seconds")
        
        result, offset = compare_test_set_to_ground_truth(
            mqtt_timings, ground_truth, args.tolerance, shared_alignment_offset
        )
        results["mqtt_commands_vs_ground_truth"] = result
        
        if "error" not in result:
            mqtt_alignment_offset = offset if shared_alignment_offset is None else shared_alignment_offset
            print_metrics_summary(result["metrics"], "MQTT Commands vs Ground Truth")
        else:
            print(f"  {result['error']}", file=sys.stderr)
    
    # Compare MQTT commands to hits_predictions directly
    ordered_pairings = None
    if args.mqtt_log and args.hits_log and hits_timings is not None and mqtt_timings is not None:
        print(f"\nComparing MQTT Commands to Hits Predictions (direct comparison)")
        
        # Use shared alignment offset for both if available (aligns both to ground truth)
        # This matches original behavior: use hits_alignment_offset for both sources
        test_set1_offset = hits_alignment_offset
        test_set2_offset = hits_alignment_offset  # Use same offset for both (aligns both to ground truth)
        
        result, ordered_pairings = compare_two_test_sets(
            hits_timings, mqtt_timings, args.tolerance,
            test_set1_offset, test_set2_offset,
            "hits_predictions", "mqtt_commands"
        )
        
        # Rename keys to match expected output format
        if "error" not in result:
            results["mqtt_commands_vs_hits_predictions"] = {
                "alignment_offset": result["alignment_offset"],
                "metrics": result["metrics"],
                "matched_pairs": [
                    {"hits_predictions": pair["test_set1"], "mqtt_commands": pair["test_set2"], "error": pair["error"]}
                    for pair in result["matched_pairs"]
                ],
                "unmatched_hits_predictions": result["unmatched_test_set1"],
                "unmatched_mqtt_commands": result["unmatched_test_set2"]
            }
            print_metrics_summary(result["metrics"], "MQTT Commands vs Hits Predictions", reference_label="hits predictions")
        else:
            results["mqtt_commands_vs_hits_predictions"] = result
            print(f"  {result['error']}", file=sys.stderr)
    
    # Write output JSON if requested
    if args.output:
        print(f"\nWriting report to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print("  Report written successfully")
    
    # Write CSV with aligned data sets if requested
    if args.csv_output:
        print(f"\nWriting aligned data sets to CSV: {args.csv_output}")
        try:
            export_aligned_data_to_csv(
                args.csv_output,
                ground_truth,
                hits_timings,
                mqtt_timings,
                hits_alignment_offset,
                mqtt_alignment_offset,
                ordered_pairings
            )
            print("  CSV written successfully")
        except Exception as e:
            print(f"  Error writing CSV: {e}", file=sys.stderr)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

