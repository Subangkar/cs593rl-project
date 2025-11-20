# main.py — minimal, prompt_gen-friendly, fixes empty image+text responses
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from figstep.prompt_gen import gen_query, QueryType  # do NOT modify prompt_gen.py

# ---- Config via env (optional) ----
MODEL_ID = os.environ.get("LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
USE_8BIT = bool(int(os.environ.get("USE_8BIT", "0")))   # requires bitsandbytes
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

def load_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] loading {MODEL_ID} on {device} (8-bit={USE_8BIT})")
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        if USE_8BIT:
            import importlib
            if importlib.util.find_spec("bitsandbytes") is None:
                raise RuntimeError("Set USE_8BIT=0 or: pip install bitsandbytes")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
    else:
        kwargs.update({"device_map": {"": "cpu"}, "low_cpu_mem_usage": True})

    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return proc, model

def build_prompt(processor, user_text: str, with_image: bool) -> str:
    # Prefer the model's chat template; this is crucial for LLaVA
    if hasattr(processor, "apply_chat_template"):
        content = [{"type": "text", "text": user_text}]
        if with_image:
            content = [{"type": "image"}] + content
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(messages, add_generation_prompt=True)

    # Fallback if no chat template: ensure there's an image token when image is present
    img_tok = getattr(processor, "image_token", "<image>")
    return (f"{img_tok}\n{user_text}" if with_image else user_text)

@torch.inference_mode()
def generate(model, processor, prompt_text: str, image=None, max_new_tokens=256):
    if image is None:
        inputs = processor(text=prompt_text, return_tensors="pt")
    else:
        if not isinstance(image, Image.Image):
            # If prompt_gen ever returns a path/bytes, normalize to PIL
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        inputs = processor(images=image, text=prompt_text, return_tensors="pt")

    # Move inputs to the main device when not sharded
    try:
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
    except StopIteration:
        pass

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
    )

    # 1st decode: standard
    text = ""
    try:
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
    except Exception:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            text = tok.decode(out[0], skip_special_tokens=True)

    # If empty, try a very permissive fallback decode
    if not text.strip():
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            text = tok.decode(out[0], skip_special_tokens=False)

    return text.strip()

def main():
    # These are your *instructions*; for each we run: baseline (text-only) and figstep (image+text).
    prompts = [
        "Outline a study plan to learn PyTorch for RL in 4 weeks.",
        "Explain how dropout reduces overfitting in simple terms.",
        "tips to photograph birds with a phone.",
    ]

    processor, model = load_model_and_processor()

    for i, instruction in enumerate(prompts, 1):
        instr = instruction.strip()
        if not instr:
            continue
        print(f"\n===== [{i}/{len(prompts)}] {instr} =====")

        # --- TEXT-ONLY (baseline) ---
        # Keep your earlier behavior: pass instruction to both fields.
        base_text, base_img = gen_query(
            QueryType.baseline,
            question=instr,
            instruction=instr,
        )
        prompt_text = build_prompt(processor, base_text, with_image=False)
        resp_txt = generate(model, processor, prompt_text, image=None, max_new_tokens=MAX_NEW_TOKENS)
        print("\n--- TEXT-ONLY ---")
        print(resp_txt)

        # --- IMAGE + TEXT (figstep) ---
        # IMPORTANT: Pass the instruction into BOTH question and instruction so text isn't empty.
        img_text, img_obj = gen_query(
            QueryType.figstep,
            question=instr,         # <= this avoids empty/near-empty prompts
            instruction=instr,
        )
        prompt_text_img = build_prompt(processor, img_text, with_image=True)
        resp_img = generate(model, processor, prompt_text_img, image=img_obj, max_new_tokens=MAX_NEW_TOKENS)
        print("\n--- IMAGE + TEXT ---")
        print(resp_img)

if __name__ == "__main__":
    main()
