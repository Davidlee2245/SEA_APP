#!/usr/bin/env python3
"""
Diagnostic script to check colocalization calculation.
Tests actual distances between detections to see why colocalization is 0.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path

def diagnose_colocalization(sample_name: str = "1"):
    """Diagnose colocalization issues for a sample."""
    
    csv_path = Path(f"data/output/{sample_name}/{sample_name}_detections.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return
    
    print(f"📊 Analyzing colocalization for sample: {sample_name}")
    print("=" * 60)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get unique channels
    channels = df['channel'].unique()
    print(f"\n🔬 Channels found: {len(channels)}")
    for ch in channels:
        count = len(df[df['channel'] == ch])
        print(f"   {ch}: {count} detections")
    
    # Analyze each channel pair
    print(f"\n📏 Distance Analysis:")
    print("-" * 60)
    
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            ch1, ch2 = channels[i], channels[j]
            
            ch1_data = df[df['channel'] == ch1]
            ch2_data = df[df['channel'] == ch2]
            
            if len(ch1_data) == 0 or len(ch2_data) == 0:
                continue
            
            # Extract coordinates (x, y)
            coords_ch1 = ch1_data[['x_coord', 'y_coord']].values
            coords_ch2 = ch2_data[['x_coord', 'y_coord']].values
            
            # Calculate all pairwise distances
            distances = cdist(coords_ch1, coords_ch2)
            
            # Find minimum distance for each ch1 detection
            min_distances = distances.min(axis=1)
            
            # Statistics
            min_dist = min_distances.min()
            max_dist = min_distances.max()
            mean_dist = min_distances.mean()
            median_dist = np.median(min_distances)
            
            # Count within different thresholds
            threshold_5 = (min_distances <= 5.0).sum()
            threshold_10 = (min_distances <= 10.0).sum()
            threshold_15 = (min_distances <= 15.0).sum()
            threshold_20 = (min_distances <= 20.0).sum()
            
            print(f"\n{ch1} ↔ {ch2}:")
            print(f"   Detections: {len(ch1_data)} vs {len(ch2_data)}")
            print(f"   Distance stats:")
            print(f"      Min:    {min_dist:.2f} pixels")
            print(f"      Max:    {max_dist:.2f} pixels")
            print(f"      Mean:   {mean_dist:.2f} pixels")
            print(f"      Median: {median_dist:.2f} pixels")
            print(f"   Within threshold:")
            print(f"      ≤ 5px:  {threshold_5} detections ({threshold_5/len(ch1_data)*100:.1f}%)")
            print(f"      ≤ 10px: {threshold_10} detections ({threshold_10/len(ch1_data)*100:.1f}%)")
            print(f"      ≤ 15px: {threshold_15} detections ({threshold_15/len(ch1_data)*100:.1f}%)")
            print(f"      ≤ 20px: {threshold_20} detections ({threshold_20/len(ch1_data)*100:.1f}%)")
            
            # Show closest pairs
            closest_idx = min_distances.argmin()
            closest_ch1 = ch1_data.iloc[closest_idx]
            closest_ch2_idx = distances[closest_idx].argmin()
            closest_ch2 = ch2_data.iloc[closest_ch2_idx]
            
            print(f"   Closest pair:")
            print(f"      Ch1: ID={closest_ch1['object_id']}, ({closest_ch1['x_coord']:.1f}, {closest_ch1['y_coord']:.1f})")
            print(f"      Ch2: ID={closest_ch2['object_id']}, ({closest_ch2['x_coord']:.1f}, {closest_ch2['y_coord']:.1f})")
            print(f"      Distance: {min_dist:.2f} pixels")
    
    # Check CSV colocalization column
    print(f"\n📋 CSV Colocalization Status:")
    print("-" * 60)
    total = len(df)
    coloc_true = (df['colocalized'] == True).sum()
    coloc_false = (df['colocalized'] == False).sum()
    
    print(f"   Total detections: {total}")
    print(f"   Colocalized=True:  {coloc_true} ({coloc_true/total*100:.1f}%)")
    print(f"   Colocalized=False: {coloc_false} ({coloc_false/total*100:.1f}%)")
    
    if coloc_true == 0:
        print(f"\n   ⚠️  WARNING: All detections marked as False!")
        print(f"   This suggests the CSV marking code is broken.")
    
    # Summary
    print(f"\n🎯 Summary:")
    print("-" * 60)
    if mean_dist > 20:
        print(f"   ⚠️  Average distance is HIGH ({mean_dist:.1f} pixels)")
        print(f"   → Channels are likely MISALIGNED")
        print(f"   → Need to fix alignment first!")
    elif mean_dist > 10:
        print(f"   ⚠️  Average distance is MODERATE ({mean_dist:.1f} pixels)")
        print(f"   → Threshold of 10px might be too strict")
        print(f"   → Try increasing to 15-20 pixels")
    else:
        print(f"   ✅ Average distance is LOW ({mean_dist:.1f} pixels)")
        print(f"   → Alignment looks good")
        print(f"   → Issue might be in calculation algorithm")
    
    if coloc_true == 0 and threshold_10 > 0:
        print(f"   ⚠️  CSV marking is BROKEN!")
        print(f"   → {threshold_10} pairs within 10px, but CSV shows 0")
        print(f"   → Need to fix CSV marking code!")

if __name__ == "__main__":
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "1"
    diagnose_colocalization(sample)


