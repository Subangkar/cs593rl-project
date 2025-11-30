#!/usr/bin/env python3
"""
Test different image formats and settings to see what works best with Ollama's LLaVA
"""

import ollama
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from prompt_gen import text_to_image, text_step_by_step

def test_current_image_generation():
    """Test the current image generation approach"""
    print("=== Testing Current Image Generation ===")
    
    # Generate image using current method
    test_instruction = "How to make a cake"
    wrapped_instruction = text_step_by_step(test_instruction, wrap=True)
    print(f"Wrapped instruction: {repr(wrapped_instruction)}")
    
    try:
        img = text_to_image(wrapped_instruction)
        img.save("test_current.png")
        print(f"✓ Image created successfully: {img.size}, mode: {img.mode}")
        return img
    except Exception as e:
        print(f"✗ Error creating image: {e}")
        return None

def test_image_with_llava(img, description="current method"):
    """Test an image with LLaVA"""
    if img is None:
        return None
        
    try:
        # Convert to bytes (current method)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        response = ollama.chat(
            model='llava:latest',
            messages=[{
                'role': 'user',
                'content': 'What text do you see in this image? Please read all the text carefully and tell me exactly what is written.',
                'images': [img_bytes]
            }],
            options={
                'temperature': 0.1,
                'num_predict': 300
            }
        )
        
        print(f"LLaVA response for {description}:")
        print(f"  {response['message']['content'][:200]}...")
        return response['message']['content']
        
    except Exception as e:
        print(f"✗ Error testing with LLaVA: {e}")
        return None

def test_different_formats(base_img):
    """Test different image formats and settings"""
    print("\n=== Testing Different Image Formats ===")
    
    formats_to_test = [
        # (format, quality, description)
        ('PNG', None, 'PNG default'),
        ('JPEG', 95, 'JPEG high quality'),
        ('JPEG', 85, 'JPEG medium quality'),
    ]
    
    for fmt, quality, desc in formats_to_test:
        try:
            img_bytes = io.BytesIO()
            if quality:
                base_img.save(img_bytes, format=fmt, quality=quality)
            else:
                base_img.save(img_bytes, format=fmt)
            img_bytes = img_bytes.getvalue()
            
            print(f"\nTesting {desc} (size: {len(img_bytes)} bytes):")
            test_image_with_llava_bytes(img_bytes, desc)
            
        except Exception as e:
            print(f"✗ Error with {desc}: {e}")

def test_image_with_llava_bytes(img_bytes, description):
    """Test image bytes directly with LLaVA"""
    try:
        response = ollama.chat(
            model='llava:latest',
            messages=[{
                'role': 'user',
                'content': 'What do you see in this image? Please describe the text content.',
                'images': [img_bytes]
            }],
            options={'temperature': 0.1, 'num_predict': 200}
        )
        
        content = response['message']['content']
        print(f"  Response length: {len(content)} chars")
        print(f"  First 100 chars: {content[:100]}")
        
        # Check if response seems relevant
        relevant_keywords = ['1.', '2.', '3.', 'cake', 'list', 'numbered']
        relevance_score = sum(1 for keyword in relevant_keywords if keyword.lower() in content.lower())
        print(f"  Relevance score: {relevance_score}/{len(relevant_keywords)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

def test_base64_encoding(base_img):
    """Test base64 encoding approach"""
    print("\n=== Testing Base64 Encoding ===")
    
    try:
        img_bytes = io.BytesIO()
        base_img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Convert to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        print(f"Base64 length: {len(img_b64)} chars")
        # Note: Ollama typically expects raw bytes, not base64, but let's see
        
    except Exception as e:
        print(f"✗ Error with base64: {e}")

if __name__ == "__main__":
    print("🔍 Diagnosing Image Issues with Ollama LLaVA\n")
    
    # Test 1: Current approach
    img = test_current_image_generation()
    
    if img:
        # Test 2: Current image with LLaVA
        test_image_with_llava(img, "current image")
        
        # Test 3: Different formats
        test_different_formats(img)
        
        # Test 4: Base64 (informational)
        test_base64_encoding(img)
    
    print("\n🏁 Diagnosis complete!")