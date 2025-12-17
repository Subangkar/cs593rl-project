#!/usr/bin/env python3
"""Test the logistic transformation function"""

import math

def logistic_transform_reward(x, k=10.0):
    """Transform similarity score using logistic function centered at 0.6
    
    Args:
        x: Raw similarity score from judge (0.0 to 1.0)
        k: Steepness parameter for logistic function (default: 10.0)
        
    Returns:
        Transformed reward score using f(x) = (S(x) - S(0)) / (S(1) - S(0))
        where S(x) = 1 / (1 + e^(-k(x - 0.6)))
    """
    def logistic(val):
        # S(x) = 1 / (1 + e^(-k(x - 0.6)))
        return 1.0 / (1.0 + math.exp(-k * (val - 0.6)))
    
    # Calculate S(x), S(0), and S(1)
    s_x = logistic(x)
    s_0 = logistic(0.0)
    s_1 = logistic(1.0)
    
    # Apply formula: f(x) = (S(x) - S(0)) / (S(1) - S(0))
    reward = (s_x - s_0) / (s_1 - s_0)
    
    # Ensure reward is in [0, 1] range
    return max(0.0, min(1.0, reward))

def test_transformation():
    """Test the logistic transformation with various inputs"""
    print("Testing Logistic Transformation:")
    print("=" * 50)
    print("Raw Score | Transformed Score | Interpretation")
    print("-" * 50)
    
    test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    
    for x in test_values:
        transformed = logistic_transform_reward(x)
        if x < 0.5:
            interpretation = "Low reward"
        elif x < 0.6:
            interpretation = "Medium-low"  
        elif x < 0.7:
            interpretation = "Medium-high"
        else:
            interpretation = "High reward"
            
        print(f"  {x:5.2f}   |     {transformed:6.3f}     | {interpretation}")
    
    print("\nKey Properties:")
    print(f"- At x=0.6 (threshold): f(0.6) = {logistic_transform_reward(0.6):.3f}")
    print(f"- Steep gradient around 0.6 due to logistic centering")
    print(f"- Values below 0.6 get lower rewards, above 0.6 get higher rewards")

if __name__ == "__main__":
    test_transformation()