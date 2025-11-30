#!/usr/bin/env python3
"""
Quick test to see if Ollama LLaVA can read the current image format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_gen import text_to_image, text_step_by_step
import ollama
import io

def test_ollama_image_reading():
    """Test if Ollama can read our generated images"""
    print("🔍 Testing Ollama LLaVA image reading...")
    
    # Create a simple test image
    test_instruction = "How to bake cookies"
    formatted_text = text_step_by_step(test_instruction, wrap=True)
    
    print(f"Generated text: {repr(formatted_text)}")
    
    try:
        # Generate image using current method
        img = text_to_image(formatted_text)
        img.save("ollama_test.png")
        print(f"✓ Image created: {img.size}, mode: {img.mode}")
        
        # Convert to bytes for Ollama
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        print(f"✓ Image converted to bytes: {len(img_bytes)} bytes")
        
        # Test with Ollama LLaVA
        print("📤 Sending to Ollama LLaVA...")
        response = ollama.chat(
            model='llava:7b',
            messages=[{
                'role': 'user',
                'content': 'Please read this image carefully and tell me exactly what text you see. List all the text content.',
                'images': [img_bytes]
            }],
            options={
                'temperature': 0.1,
                'num_predict': 200
            }
        )
        
        result = response['message']['content']
        print(f"📥 LLaVA Response:")
        print(f"   {result}")
        
        # Analyze response quality
        print(f"\n📊 Analysis:")
        print(f"   Response length: {len(result)} characters")
        
        # Check if it recognizes key elements
        key_elements = ['1.', '2.', '3.', 'cookies', 'bake', 'How to']
        found_elements = [elem for elem in key_elements if elem.lower() in result.lower()]
        print(f"   Found key elements: {found_elements}")
        print(f"   Recognition score: {len(found_elements)}/{len(key_elements)}")
        
        if len(found_elements) < 2:
            print("⚠️  LOW RECOGNITION - LLaVA is not reading the image text properly")
        else:
            print("✅ GOOD RECOGNITION - LLaVA can read the image")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_image_reading()