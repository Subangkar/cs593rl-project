import torch
from PIL import Image
from transformers import (
    AutoConfig, AutoProcessor, AutoModelForVision2Seq,
)
from transformers import LlavaForConditionalGeneration  # for LLaVA specifically

from prompt_gen import QueryType, gen_query


class LLMModel:
    def __init__(self, MODEL_ID):
        self.model_name = MODEL_ID

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[info] loading {MODEL_ID} on {device} ({dtype})")
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        # Pick correct model class:
        if getattr(config, "model_type", "") in {"llava"}:
            model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            # Good for IDEFICS2 and many other open-source VLMs
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

    def generate(self, prompt):
        # Simulate LLM response generation
        return f"Generated response for: {prompt}"



class MaliciousQueryExecutor:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model

    def paraphrase_prompt(self, instruction):
        prompt = f"Paraphrase the following instruction while retaining its meaning:\n\n{instruction}"
        response = self.llm_model.generate(prompt)
        return response

    def generate_malicious_output(self, query_type, question, instruction):
        prompt, image = gen_query(query_type, question, instruction)
        response = self.llm_model.generate(prompt)
        return response


class BenignQueryExecutor:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def generate_benign_output(self, question):
        response = self.llm_model.generate(question)
        return response