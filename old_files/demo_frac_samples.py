#!/usr/bin/env python3
"""
Quick demo of the --frac-samples parameter
Shows how random sampling works with different fractions
"""

import argparse
import random

def simulate_sampling(total_size, frac_samples, seed=42):
    """Simulate what the environment does with frac-samples"""
    random.seed(seed)
    
    # Split into train/test (mimics environment behavior)
    train_set = list(range(0, 800))
    test_set = list(range(800, 1020))
    
    print(f"\nOriginal Dataset:")
    print(f"  Train set: {len(train_set)} queries (0-799)")
    print(f"  Test set: {len(test_set)} queries (800-1019)")
    
    # Sample training set
    if frac_samples < 1.0 and frac_samples > 0.0:
        sample_size = int(len(train_set) * frac_samples)
        sample_size = max(1, sample_size)
        sampled_train = random.sample(train_set, sample_size)
        
        print(f"\nAfter Sampling (frac_samples={frac_samples}):")
        print(f"  Train set: {len(sampled_train)} queries ({frac_samples*100:.1f}%)")
        print(f"  Test set: {len(test_set)} queries (unchanged)")
        print(f"  Speedup estimate: ~{100 - frac_samples*100:.0f}% faster")
        
        # Show first few sampled indices
        print(f"\n  First 10 sampled training indices: {sorted(sampled_train)[:10]}")
    else:
        print(f"\nNo sampling (frac_samples={frac_samples}, using all data)")

def main():
    parser = argparse.ArgumentParser(description='Demo of frac-samples parameter')
    parser.add_argument('--frac-samples', type=float, default=1.0,
                        help='fraction to sample (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Sampling Demo")
    print("="*60)
    
    simulate_sampling(1020, args.frac_samples, args.seed)
    
    print("\n" + "="*60)
    print("Try different values:")
    print("  python demo_frac_samples.py --frac-samples 0.1")
    print("  python demo_frac_samples.py --frac-samples 0.25")
    print("  python demo_frac_samples.py --frac-samples 0.5")
    print("  python demo_frac_samples.py --frac-samples 1.0")
    print("="*60)

if __name__ == "__main__":
    main()
