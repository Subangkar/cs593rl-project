#!/usr/bin/env python3
"""Test the judge response parsing logic directly"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_client import OllamaClient

def test_parsing():
    client = OllamaClient()
    
    # Test cases for different response formats
    test_cases = [
        # JSON format (should work)
        ('{"score": 0.2, "explanation": "Responses are quite different"}', "JSON format"),
        
        # Natural language (R1 model style)
        ("Alright, I'm trying to figure out how similar these two responses are. The similarity score is 0.3 because they differ significantly.", "Natural language with score"),
        
        # Natural language variation 
        ("Looking at these responses, I would rate it 0.8 since they're quite similar in approach.", "Rate it pattern"),
        
        # Natural language with different pattern
        ("I give this a score of 0.1 because the responses are very different.", "Give score pattern"),
        
        # Just a number
        ("The responses are different. 0.4", "Number only"),
    ]
    
    print("Testing judge response parsing...")
    print("=" * 60)
    
    for response_text, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {response_text}")
        try:
            score, explanation = client._parse_judge_response(response_text)
            print(f"✅ SUCCESS: Score={score}, Explanation='{explanation}'")
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Parsing test complete!")

if __name__ == "__main__":
    test_parsing()