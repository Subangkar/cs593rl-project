"""
Hybrid approach: Use transformers for VLM (full precision), Ollama for text-only models
"""

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import ollama
from PIL import Image
import io

class HybridVLMModel:
    """Use transformers for VLM (full precision) and Ollama for text-only"""
    
    def __init__(self, vlm_model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        print(f"Loading full-precision VLM: {vlm_model_name}")
        self.processor = LlavaNextProcessor.from_pretrained(vlm_model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto"
        )
        
    def generate(self, prompt, image=None, max_new_tokens=512, temperature=0.1, top_p=0.9):
        if image is None:
            raise ValueError("This model requires an image")
            
        # Prepare inputs
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if prompt in response:
            response = response.split(prompt)[-1].strip()
            
        return response

class TextLLMModel:
    """Keep using Ollama for text-only tasks (paraphrasing)"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                'temperature': temperature,
                'top_p': top_p,
                'num_predict': max_new_tokens,
            }
        )
        return response['response'].strip()

# Usage example:
if __name__ == "__main__":
    # Use full-precision transformers VLM
    vlm = HybridVLMModel("llava-hf/llava-v1.6-mistral-7b-hf")
    
    # Use Ollama for text-only (fast)
    text_llm = TextLLMModel("mistral:7b-instruct")
    
    print("Hybrid setup complete: Full-precision VLM + Ollama text model")