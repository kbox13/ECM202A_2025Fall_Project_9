#!/usr/bin/env python3
"""
Simple Hits vs MQTT Comparison Tool

Directly compares kick timings from hits_predictions logs against MQTT commands logs.
No alignment needed - timestamps are already in the same reference frame.
"""

import json
import argparse
import statistics
import sys
import csv
from typing import List, Tuple, Optional


def extract_hits_timings(log_path: str) -> List[float]:
    """
    Extract kick hit timings from hits_predictions log.
    
    Args:
        log_path: Path to hits_predictions log file
        
    Returns:
        Sorted list of audio_time values for kick hits
    """
    timings = []
    
    try:
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


def extract_mqtt_timings(log_path: str) -> List[float]:
    """
    Extract kick timings from MQTT commands log.
    
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


def compare_timestamps(
    hits_times: List[float],
    mqtt_times: List[float],
    tolerance: float
) -> Tuple[List[Tuple[float, float, float]], List[float], List[float]]:
    """
    Compare hits timestamps to MQTT timestamps (no alignment needed).
    
    Args:
        hits_times: List of hits timestamps
        mqtt_times: List of MQTT timestamps
        tolerance: Maximum distance for a match (in seconds)
        
    Returns:
        Tuple of (matched_pairs, unmatched_hits, unmatched_mqtt)
        matched_pairs: List of (hits_time, mqtt_time, error) tuples
        unmatched_hits: List of hits timestamps that didn't match
        unmatched_mqtt: List of MQTT timestamps that didn't match
    """
    if len(hits_times) == 0 or len(mqtt_times) == 0:
        return [], hits_times.copy(), mqtt_times.copy()
    
    # Generate all candidate pairs with distances
    candidate_pairs = []
    for hits_idx, hits_time in enumerate(hits_times):
        for mqtt_idx, mqtt_time in enumerate(mqtt_times):
            distance = abs(hits_time - mqtt_time)
            if distance <= tolerance:
                candidate_pairs.append((distance, hits_idx, mqtt_idx, hits_time, mqtt_time))
    
    # Sort by distance (closest first) for globally optimal matching
    candidate_pairs.sort(key=lambda x: x[0])
    
    # Match greedily from closest to farthest
    matched_hits_indices = set()
    matched_mqtt_indices = set()
    matched_pairs = []
    
    for distance, hits_idx, mqtt_idx, hits_time, mqtt_time in candidate_pairs:
        # Only match if both are still available
        if hits_idx not in matched_hits_indices and mqtt_idx not in matched_mqtt_indices:
            error = hits_time - mqtt_time
            matched_pairs.append((hits_time, mqtt_time, error))
            matched_hits_indices.add(hits_idx)
            matched_mqtt_indices.add(mqtt_idx)
    
    # Find unmatched timestamps
    unmatched_hits = [hits_times[i] for i in range(len(hits_times)) 
                     if i not in matched_hits_indices]
    unmatched_mqtt = [mqtt_times[i] for i in range(len(mqtt_times))
                     if i not in matched_mqtt_indices]
    
    return matched_pairs, unmatched_hits, unmatched_mqtt


def compare_with_ground_truth(
    test_times: List[float],
    aligned_ground_truth_times: List[float],
    original_ground_truth_times: List[float],
    tolerance: float
) -> Tuple[List[Tuple[float, float, float]], List[float], List[float]]:
    """
    Compare test timestamps to ground truth timestamps using globally optimal matching.
    
    Args:
        test_times: List of test timestamps (hits or MQTT) - original timestamps
        aligned_ground_truth_times: List of pre-aligned ground truth timestamps (for matching)
        original_ground_truth_times: List of original (unaligned) ground truth timestamps (for return values)
        tolerance: Maximum distance for a match (in seconds)
        
    Returns:
        Tuple of (matched_pairs, unmatched_test, unmatched_ground_truth)
        matched_pairs: List of (test_time, gt_time_original, error) tuples where error = test_time - aligned_gt_time
        unmatched_test: List of test timestamps that didn't match (original timestamps)
        unmatched_ground_truth: List of ground truth timestamps that didn't match (original, unaligned timestamps)
    """
    if len(test_times) == 0 or len(aligned_ground_truth_times) == 0:
        return [], test_times.copy(), original_ground_truth_times.copy()
    
    # Generate all candidate pairs with distances (using aligned GT for matching)
    candidate_pairs = []
    for test_idx, test_time in enumerate(test_times):
        for gt_idx, aligned_gt_time in enumerate(aligned_ground_truth_times):
            distance = abs(test_time - aligned_gt_time)
            if distance <= tolerance:
                gt_time_original = original_ground_truth_times[gt_idx]
                candidate_pairs.append((distance, test_idx, gt_idx, test_time, gt_time_original, aligned_gt_time))
    
    # Sort by distance (closest first) for globally optimal matching
    candidate_pairs.sort(key=lambda x: x[0])
    
    # Match greedily from closest to farthest
    matched_test_indices = set()
    matched_gt_indices = set()
    matched_pairs = []
    
    for distance, test_idx, gt_idx, test_time, gt_time_original, aligned_gt_time in candidate_pairs:
        # Only match if both are still available
        if test_idx not in matched_test_indices and gt_idx not in matched_gt_indices:
            error = test_time - aligned_gt_time  # Error using aligned GT time
            matched_pairs.append((test_time, gt_time_original, error))  # Return original GT time
            matched_test_indices.add(test_idx)
            matched_gt_indices.add(gt_idx)
    
    # Find unmatched timestamps
    unmatched_test = [test_times[i] for i in range(len(test_times)) 
                     if i not in matched_test_indices]
    unmatched_ground_truth = [original_ground_truth_times[i] for i in range(len(original_ground_truth_times))
                             if i not in matched_gt_indices]
    
    return matched_pairs, unmatched_test, unmatched_ground_truth


def compute_metrics(
    matched_pairs: List[Tuple[float, float, float]],
    unmatched_series_1: List[float],
    unmatched_series_2: List[float],
    tolerance: float
) -> dict:
    """
    Compute accuracy metrics from comparison results.
    
    Args:
        matched_pairs: List of (hits_time, mqtt_time, error) tuples
        unmatched_hits: List of unmatched hits timestamps
        unmatched_mqtt: List of unmatched MQTT timestamps
        tolerance: Tolerance used for matching
        
    Returns:
        Dictionary of metrics
    """
    total_series_1 = len(matched_pairs) + len(unmatched_series_1)
    total_series_2 = len(matched_pairs) + len(unmatched_series_2)
    matched_count = len(matched_pairs)
    
    # Calculate precision, recall, F1
    match_1_prcnt = matched_count / total_series_1 if total_series_1 > 0 else 0.0
    match_2_prcnt = matched_count / total_series_2 if total_series_2 > 0 else 0.0
    
    if match_1_prcnt + match_2_prcnt > 0:
        f1_score = 2 * (match_1_prcnt * match_2_prcnt) / (match_1_prcnt + match_2_prcnt)
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
        "match_1_prcnt": match_1_prcnt,
        "match_2_prcnt": match_2_prcnt,
        "f1_score": f1_score,
        "matched_count": matched_count,
        "total_series_1": total_series_1,
        "total_series_2": total_series_2,
        "unmatched_series_1_count": len(unmatched_series_1),
        "unmatched_series_2_count": len(unmatched_series_2),
        "mean_absolute_error": mean_absolute_error,
        "std_error": std_error,
        "max_absolute_error": max_absolute_error,
        "min_error": min_error,
        "tolerance": tolerance
    }


def print_metrics_summary(metrics: dict):
    """Print a summary of metrics to console."""
    print(f"\nComparison Metrics:")
    print(f"  Match 1 Precision: {metrics['match_1_prcnt']:.4f}")
    print(f"  Match 2 Precision: {metrics['match_2_prcnt']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Matched: {metrics['matched_count']} / {metrics['total_series_1']} series_1, "
          f"{metrics['matched_count']} / {metrics['total_series_2']} series_2")
    print(f"  Mean Absolute Error: {metrics['mean_absolute_error']*1000:.2f} ms")
    print(f"  Std Error: {metrics['std_error']*1000:.2f} ms")
    print(f"  Max Absolute Error: {metrics['max_absolute_error']*1000:.2f} ms")
    print(f"  Min Error: {metrics['min_error']*1000:.2f} ms")


def build_ordered_pairings(
    hits_mqtt_matched: List[Tuple[float, float, float]],
    hits_mqtt_unmatched_hits: List[float],
    hits_mqtt_unmatched_mqtt: List[float],
    hits_gt_matched: Optional[List[Tuple[float, float, float]]] = None,
    hits_gt_unmatched_hits: Optional[List[float]] = None,
    hits_gt_unmatched_gt: Optional[List[float]] = None,
    mqtt_gt_matched: Optional[List[Tuple[float, float, float]]] = None,
    mqtt_gt_unmatched_mqtt: Optional[List[float]] = None,
    mqtt_gt_unmatched_gt: Optional[List[float]] = None,
    alignment_offset: Optional[float] = None
) -> List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]:
    """
    Build ordered pairings list with hits, MQTT, and optionally ground truth, sorted chronologically.
    
    Args:
        hits_mqtt_matched: List of (hits_time, mqtt_time, hits_mqtt_error) tuples
        hits_mqtt_unmatched_hits: List of unmatched hits timestamps
        hits_mqtt_unmatched_mqtt: List of unmatched MQTT timestamps
        hits_gt_matched: Optional list of (hits_time, gt_time_original, hits_error) tuples
        hits_gt_unmatched_hits: Optional list of unmatched hits timestamps from GT comparison
        hits_gt_unmatched_gt: Optional list of unmatched GT timestamps from hits comparison
        mqtt_gt_matched: Optional list of (mqtt_time, gt_time_original, mqtt_error) tuples
        mqtt_gt_unmatched_mqtt: Optional list of unmatched MQTT timestamps from GT comparison
        mqtt_gt_unmatched_gt: Optional list of unmatched GT timestamps from MQTT comparison
        alignment_offset: Optional alignment offset for calculating errors with original GT times
        
    Returns:
        List of (hits_time, mqtt_time, ground_truth_time, hits_error, mqtt_error, hits_mqtt_error) tuples
        - All timestamps are original (not aligned)
        - Errors are calculated: hits_error = hits_time - (gt_time + offset), mqtt_error = mqtt_time - (gt_time + offset)
        - Values are None when not applicable
    """
    # If no ground truth provided, use simple version
    if hits_gt_matched is None:
        pairings = []
        # Add matched pairs
        for hits_time, mqtt_time, error in hits_mqtt_matched:
            pairings.append((hits_time, mqtt_time, None, None, None, error))
        # Add unmatched hits
        for hits_time in hits_mqtt_unmatched_hits:
            pairings.append((hits_time, None, None, None, None, None))
        # Add unmatched mqtt
        for mqtt_time in hits_mqtt_unmatched_mqtt:
            pairings.append((None, mqtt_time, None, None, None, None))
        
        # Sort by hits_time (or mqtt_time if hits is None)
        def sort_key(x):
            hits_time, mqtt_time = x[0], x[1]
            if hits_time is not None:
                return hits_time
            elif mqtt_time is not None:
                return mqtt_time
            else:
                return float('inf')
        
        pairings.sort(key=sort_key)
        return pairings
    
    # Ground truth version: create lookup dictionaries
    hits_to_mqtt = {}
    hits_to_gt = {}
    mqtt_to_gt = {}
    
    # Populate hits to MQTT mapping
    for hits_time, mqtt_time, hits_mqtt_error in hits_mqtt_matched:
        hits_to_mqtt[hits_time] = (mqtt_time, hits_mqtt_error)
    
    # Populate hits to GT mapping
    for hits_time, gt_time_original, hits_error in hits_gt_matched:
        hits_to_gt[hits_time] = (gt_time_original, hits_error)
    
    # Populate MQTT to GT mapping
    for mqtt_time, gt_time_original, mqtt_error in mqtt_gt_matched:
        mqtt_to_gt[mqtt_time] = (gt_time_original, mqtt_error)
    
    # Collect all unique timestamps
    all_timestamps = set()
    
    # From hits_mqtt comparison
    for hits_time, mqtt_time, _ in hits_mqtt_matched:
        all_timestamps.add(('hits', hits_time))
        all_timestamps.add(('mqtt', mqtt_time))
    for hits_time in hits_mqtt_unmatched_hits:
        all_timestamps.add(('hits', hits_time))
    for mqtt_time in hits_mqtt_unmatched_mqtt:
        all_timestamps.add(('mqtt', mqtt_time))
    
    # From hits_gt comparison
    for hits_time, gt_time, _ in hits_gt_matched:
        all_timestamps.add(('hits', hits_time))
        all_timestamps.add(('gt', gt_time))
    for hits_time in hits_gt_unmatched_hits or []:
        all_timestamps.add(('hits', hits_time))
    for gt_time in hits_gt_unmatched_gt or []:
        all_timestamps.add(('gt', gt_time))
    
    # From mqtt_gt comparison
    for mqtt_time, gt_time, _ in mqtt_gt_matched:
        all_timestamps.add(('mqtt', mqtt_time))
        all_timestamps.add(('gt', gt_time))
    for mqtt_time in mqtt_gt_unmatched_mqtt or []:
        all_timestamps.add(('mqtt', mqtt_time))
    for gt_time in mqtt_gt_unmatched_gt or []:
        all_timestamps.add(('gt', gt_time))
    
    # Build rows for each unique timestamp
    rows = {}
    
    for source, timestamp in all_timestamps:
        if source == 'hits':
            hits_time = timestamp
            mqtt_time = None
            gt_time = None
            hits_error = None
            mqtt_error = None
            hits_mqtt_error = None
            
            # Look up MQTT match
            if hits_time in hits_to_mqtt:
                mqtt_time, hits_mqtt_error = hits_to_mqtt[hits_time]
            
            # Look up GT match
            if hits_time in hits_to_gt:
                gt_time, hits_error = hits_to_gt[hits_time]
            
            # If we have both MQTT and GT matches, also check if MQTT has GT match
            if mqtt_time is not None and mqtt_time in mqtt_to_gt:
                gt_time_from_mqtt, mqtt_error = mqtt_to_gt[mqtt_time]
                # Use GT from MQTT if we don't have one from hits, or if they're the same
                if gt_time is None:
                    gt_time = gt_time_from_mqtt
                elif abs(gt_time - gt_time_from_mqtt) < 0.001:  # Same GT time
                    mqtt_error = mqtt_to_gt[mqtt_time][1]
            
            # Calculate mqtt_error if we have GT but not from mqtt_to_gt
            if gt_time is not None and mqtt_error is None and mqtt_time is not None and alignment_offset is not None:
                mqtt_error = mqtt_time - (gt_time + alignment_offset)
            
            key = ('hits', hits_time)
            if key not in rows:
                rows[key] = (hits_time, mqtt_time, gt_time, hits_error, mqtt_error, hits_mqtt_error)
            else:
                # Merge with existing row
                existing = rows[key]
                rows[key] = (
                    hits_time,
                    mqtt_time if mqtt_time is not None else existing[1],
                    gt_time if gt_time is not None else existing[2],
                    hits_error if hits_error is not None else existing[3],
                    mqtt_error if mqtt_error is not None else existing[4],
                    hits_mqtt_error if hits_mqtt_error is not None else existing[5]
                )
        
        elif source == 'mqtt':
            mqtt_time = timestamp
            hits_time = None
            gt_time = None
            hits_error = None
            mqtt_error = None
            hits_mqtt_error = None
            
            # Look up GT match
            if mqtt_time in mqtt_to_gt:
                gt_time, mqtt_error = mqtt_to_gt[mqtt_time]
            
            # Look up hits match (reverse lookup)
            for h_time, (m_time, h_m_error) in hits_to_mqtt.items():
                if abs(m_time - mqtt_time) < 0.001:
                    hits_time = h_time
                    hits_mqtt_error = h_m_error
                    break
            
            # If we have hits match, check for GT match from hits
            if hits_time is not None and hits_time in hits_to_gt:
                gt_time_from_hits, hits_error = hits_to_gt[hits_time]
                if gt_time is None:
                    gt_time = gt_time_from_hits
                elif abs(gt_time - gt_time_from_hits) < 0.001:
                    hits_error = hits_to_gt[hits_time][1]
            
            # Calculate hits_error if we have GT but not from hits_to_gt
            if gt_time is not None and hits_error is None and hits_time is not None and alignment_offset is not None:
                hits_error = hits_time - (gt_time + alignment_offset)
            
            key = ('mqtt', mqtt_time)
            if key not in rows:
                rows[key] = (hits_time, mqtt_time, gt_time, hits_error, mqtt_error, hits_mqtt_error)
            else:
                # Merge with existing row
                existing = rows[key]
                rows[key] = (
                    hits_time if hits_time is not None else existing[0],
                    mqtt_time,
                    gt_time if gt_time is not None else existing[2],
                    hits_error if hits_error is not None else existing[3],
                    mqtt_error if mqtt_error is not None else existing[4],
                    hits_mqtt_error if hits_mqtt_error is not None else existing[5]
                )
        
        elif source == 'gt':
            gt_time = timestamp
            hits_time = None
            mqtt_time = None
            hits_error = None
            mqtt_error = None
            hits_mqtt_error = None
            
            # Look up hits match (reverse lookup)
            for h_time, (gt_time_orig, h_error) in hits_to_gt.items():
                if abs(gt_time_orig - gt_time) < 0.001:
                    hits_time = h_time
                    hits_error = h_error
                    break
            
            # Look up MQTT match (reverse lookup)
            for m_time, (gt_time_orig, m_error) in mqtt_to_gt.items():
                if abs(gt_time_orig - gt_time) < 0.001:
                    mqtt_time = m_time
                    mqtt_error = m_error
                    break
            
            # If we have both hits and MQTT, check for hits_mqtt match
            if hits_time is not None and mqtt_time is not None:
                if hits_time in hits_to_mqtt:
                    m_time, h_m_error = hits_to_mqtt[hits_time]
                    if abs(m_time - mqtt_time) < 0.001:
                        hits_mqtt_error = h_m_error
            
            # Calculate hits_mqtt_error if we have both but not matched
            if hits_time is not None and mqtt_time is not None and hits_mqtt_error is None:
                hits_mqtt_error = hits_time - mqtt_time
            
            key = ('gt', gt_time)
            if key not in rows:
                rows[key] = (hits_time, mqtt_time, gt_time, hits_error, mqtt_error, hits_mqtt_error)
            else:
                # Merge with existing row
                existing = rows[key]
                rows[key] = (
                    hits_time if hits_time is not None else existing[0],
                    mqtt_time if mqtt_time is not None else existing[1],
                    gt_time,
                    hits_error if hits_error is not None else existing[3],
                    mqtt_error if mqtt_error is not None else existing[4],
                    hits_mqtt_error if hits_mqtt_error is not None else existing[5]
                )
    
    # Convert to list and sort chronologically
    pairings = list(rows.values())
    
    def sort_key(x):
        hits_time, mqtt_time, gt_time = x[0], x[1], x[2]
        if hits_time is not None:
            return hits_time
        elif mqtt_time is not None:
            return mqtt_time
        elif gt_time is not None:
            return gt_time
        else:
            return float('inf')
    
    pairings.sort(key=sort_key)
    
    # Deduplicate based on (hits_time, mqtt_time, gt_time) combination
    seen = set()
    deduplicated = []
    for pairing in pairings:
        hits_time, mqtt_time, gt_time = pairing[0], pairing[1], pairing[2]
        # Create a key for deduplication (handle None values)
        key = (
            hits_time if hits_time is not None else None,
            mqtt_time if mqtt_time is not None else None,
            gt_time if gt_time is not None else None
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(pairing)
    
    return deduplicated


def export_to_csv(
    hits_times: List[float],
    mqtt_times: List[float],
    matched_pairs: List[Tuple[float, float, float]],
    unmatched_hits: List[float],
    unmatched_mqtt: List[float],
    output_path: str,
    hits_gt_matched: Optional[List[Tuple[float, float, float]]] = None,
    hits_gt_unmatched_hits: Optional[List[float]] = None,
    hits_gt_unmatched_gt: Optional[List[float]] = None,
    mqtt_gt_matched: Optional[List[Tuple[float, float, float]]] = None,
    mqtt_gt_unmatched_mqtt: Optional[List[float]] = None,
    mqtt_gt_unmatched_gt: Optional[List[float]] = None,
    alignment_offset: Optional[float] = None
):
    """
    Export comparison results to CSV file with ordered pairings.
    
    Args:
        hits_times: All hits timestamps (sorted)
        mqtt_times: All MQTT timestamps (sorted)
        matched_pairs: List of (hits_time, mqtt_time, error) tuples
        unmatched_hits: List of unmatched hits timestamps
        unmatched_mqtt: List of unmatched MQTT timestamps
        output_path: Path to output CSV file
        hits_gt_matched: Optional list of (hits_time, gt_time_original, hits_error) tuples
        hits_gt_unmatched_hits: Optional list of unmatched hits from GT comparison
        hits_gt_unmatched_gt: Optional list of unmatched GT from hits comparison
        mqtt_gt_matched: Optional list of (mqtt_time, gt_time_original, mqtt_error) tuples
        mqtt_gt_unmatched_mqtt: Optional list of unmatched MQTT from GT comparison
        mqtt_gt_unmatched_gt: Optional list of unmatched GT from MQTT comparison
        alignment_offset: Optional alignment offset for error calculations
    """
    # Build ordered pairings (chronologically sorted)
    ordered_pairings = build_ordered_pairings(
        matched_pairs, unmatched_hits, unmatched_mqtt,
        hits_gt_matched, hits_gt_unmatched_hits, hits_gt_unmatched_gt,
        mqtt_gt_matched, mqtt_gt_unmatched_mqtt, mqtt_gt_unmatched_gt,
        alignment_offset
    )
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header - different based on whether ground truth is provided
        if hits_gt_matched is not None:
            writer.writerow(['ground_truth_time', 'hits_time', 'mqtt_time', 'hits_error', 'mqtt_error', 'hits_mqtt_error'])
        else:
            writer.writerow(['hits_time', 'mqtt_time', 'error'])
        
        # Write ordered pairings
        for pairing in ordered_pairings:
            row = []
            
            if hits_gt_matched is not None:
                # Ground truth version: 6 columns
                # Use aligned timestamps (in ground truth frame)
                hits_time_orig, mqtt_time_orig, gt_time, hits_error, mqtt_error, hits_mqtt_error = pairing
                
                # Ground truth time (first column, already in GT frame, no adjustment needed)
                if gt_time is not None:
                    row.append(f"{gt_time:.6f}")
                else:
                    row.append("")
                
                # Hits time (aligned to GT frame)
                if hits_time_orig is not None:
                    if alignment_offset is not None:
                        hits_time_aligned = hits_time_orig - alignment_offset
                    else:
                        hits_time_aligned = hits_time_orig  # Fallback if offset not provided
                    row.append(f"{hits_time_aligned:.6f}")
                else:
                    row.append("")
                
                # MQTT time (aligned to GT frame)
                if mqtt_time_orig is not None:
                    if alignment_offset is not None:
                        mqtt_time_aligned = mqtt_time_orig - alignment_offset
                    else:
                        mqtt_time_aligned = mqtt_time_orig  # Fallback if offset not provided
                    row.append(f"{mqtt_time_aligned:.6f}")
                else:
                    row.append("")
                
                # Hits error
                if hits_error is not None:
                    row.append(f"{hits_error:.6f}")
                else:
                    row.append("")
                
                # MQTT error
                if mqtt_error is not None:
                    row.append(f"{mqtt_error:.6f}")
                else:
                    row.append("")
                
                # Hits-MQTT error
                if hits_mqtt_error is not None:
                    row.append(f"{hits_mqtt_error:.6f}")
                else:
                    row.append("")
            else:
                # Simple version: 3 columns
                # build_ordered_pairings returns 6-tuples even without GT: (hits_time, mqtt_time, None, None, None, hits_mqtt_error)
                hits_time, mqtt_time, _, _, _, error = pairing
                
                # Hits time
                if hits_time is not None:
                    row.append(f"{hits_time:.6f}")
                else:
                    row.append("")
                
                # MQTT time
                if mqtt_time is not None:
                    row.append(f"{mqtt_time:.6f}")
                else:
                    row.append("")
                
                # Error (hits_mqtt_error)
                if error is not None:
                    row.append(f"{error:.6f}")
                else:
                    row.append("")
            
            writer.writerow(row)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Compare hits_predictions log against MQTT commands log"
    )
    parser.add_argument(
        "--hits-log",
        required=True,
        help="Path to hits_predictions log file"
    )
    parser.add_argument(
        "--mqtt-log",
        required=True,
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
        help="Path to output CSV file (optional)"
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    # Load timings
    print("Loading hits_predictions log...")
    hits_timings = extract_hits_timings(args.hits_log)
    print(f"  Loaded {len(hits_timings)} kick hits")
    
    print("\nLoading MQTT commands log...")
    mqtt_timings = extract_mqtt_timings(args.mqtt_log)
    print(f"  Loaded {len(mqtt_timings)} kick predictions")
    
    # Compare hits vs MQTT
    print(f"\nComparing hits vs MQTT (tolerance: {args.tolerance}s)...")
    matched_pairs, unmatched_hits, unmatched_mqtt = compare_timestamps(
        hits_timings, mqtt_timings, args.tolerance
    )
    
    # Compute metrics for hits vs MQTT
    metrics = compute_metrics(matched_pairs, unmatched_hits, unmatched_mqtt, args.tolerance)
    print_metrics_summary(metrics)
    
    # Prepare results dictionary
    results = {
        "hits_vs_mqtt": {
            "metrics": metrics
        }
    }
    
    # Ground truth comparisons if provided
    alignment_offset = None
    hits_gt_matched = None
    hits_gt_unmatched_hits = None
    hits_gt_unmatched_gt = None
    mqtt_gt_matched = None
    mqtt_gt_unmatched_mqtt = None
    mqtt_gt_unmatched_gt = None
    
    if args.ground_truth:
        print("\nLoading ground truth...")
        ground_truth_timings = extract_ground_truth_timings(args.ground_truth)
        print(f"  Loaded {len(ground_truth_timings)} ground truth kick timings")
        
        if len(hits_timings) > 0 and len(ground_truth_timings) > 0:
            # Calculate alignment offset: align GT to hits by assuming first GT time equals first hits time
            alignment_offset = calculate_alignment_offset(ground_truth_timings, hits_timings)
            aligned_ground_truth = apply_alignment_offset(ground_truth_timings, alignment_offset)
            print(f"  Alignment offset: {alignment_offset:.6f} seconds")
            
            # Compare hits vs ground truth
            print(f"\nComparing hits vs ground truth (tolerance: {args.tolerance}s)...")
            hits_gt_matched, hits_gt_unmatched_hits, hits_gt_unmatched_gt = compare_with_ground_truth(
                hits_timings, aligned_ground_truth, ground_truth_timings, args.tolerance
            )
            
            hits_gt_metrics = compute_metrics(hits_gt_matched, hits_gt_unmatched_hits, hits_gt_unmatched_gt, args.tolerance)
            print_metrics_summary(hits_gt_metrics)
            results["hits_vs_ground_truth"] = {
                "alignment_offset": alignment_offset,
                "metrics": hits_gt_metrics
            }
            
            # Compare MQTT vs ground truth (using same alignment offset)
            print(f"\nComparing MQTT vs ground truth (tolerance: {args.tolerance}s)...")
            mqtt_gt_matched, mqtt_gt_unmatched_mqtt, mqtt_gt_unmatched_gt = compare_with_ground_truth(
                mqtt_timings, aligned_ground_truth, ground_truth_timings, args.tolerance
            )
            
            mqtt_gt_metrics = compute_metrics(mqtt_gt_matched, mqtt_gt_unmatched_mqtt, mqtt_gt_unmatched_gt, args.tolerance)
            print_metrics_summary(mqtt_gt_metrics)
            results["mqtt_vs_ground_truth"] = {
                "alignment_offset": alignment_offset,
                "metrics": mqtt_gt_metrics
            }
        else:
            print("  Warning: Cannot compare - empty timings", file=sys.stderr)
    
    # Write JSON output if requested
    if args.output:
        print(f"\nWriting report to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print("  Report written successfully")
    
    # Write CSV output if requested
    if args.csv_output:
        print(f"\nWriting CSV to: {args.csv_output}")
        export_to_csv(
            hits_timings, mqtt_timings, matched_pairs,
            unmatched_hits, unmatched_mqtt, args.csv_output,
            hits_gt_matched, hits_gt_unmatched_hits, hits_gt_unmatched_gt,
            mqtt_gt_matched, mqtt_gt_unmatched_mqtt, mqtt_gt_unmatched_gt,
            alignment_offset
        )
        print("  CSV written successfully")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

