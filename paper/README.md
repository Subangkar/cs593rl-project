# FigRL Research Paper

**Title:** RL-Guided Typographic Attacks on Vision-Language Models: Learning Adaptive Prompt Mutation for Jailbreaking VLMs

**Authors:** Md Ajwad Akil, Preetom Saha Arko, Subangkar Karmaker Shanto  
**Institution:** Department of Computer Science, Purdue University  
**Format:** NeurIPS 2024 Style

## Overview

This paper presents FigRL, a reinforcement learning framework for automated typographic jailbreak attacks on vision-language models. The work extends static approaches like FigStep by learning adaptive prompt mutation strategies that exploit the "modal disconnect" vulnerability in VLM safety training.

## Compilation

### Prerequisites

Install LaTeX distribution with NeurIPS style:
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Or use Overleaf (recommended)
```

### Compile Locally

```bash
cd /home/makil/cs593rl-project/paper

# Compile (run twice for references)
pdflatex main.tex
pdflatex main.tex

# View output
evince main.pdf  # Linux
open main.pdf    # macOS
```

### Using Overleaf

1. Upload `main.tex` to Overleaf
2. Download NeurIPS 2024 style from: https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
3. Upload `neurips_2024.sty` to the same project
4. Compile in Overleaf (automatic)

## Paper Structure

- **Abstract** (1 page): Overview of RL-guided typographic attacks
- **Introduction** (1.5 pages): VLM vulnerabilities, modal disconnect, FigRL framework
- **Related Work** (1.5 pages): Typographic attacks, RL for adversarial generation, multimodal safety
- **Problem Formulation** (1 page): Black-box threat model, MDP formulation, objective
- **Methodology** (2.5 pages): System architecture, PPO, mutation operators, reward evaluation, training
- **Experimental Setup** (1 page): SafeBench dataset, target VLMs, auxiliary models, baselines, metrics
- **Results** (1.5 pages): Main results, training dynamics, learned patterns, evaluation comparison, transferability
- **Discussion** (1 page): Modal disconnect implications, defensive directions, limitations, future work
- **Conclusion** (0.5 pages): Summary and impact
- **References** (1+ pages): 25 citations

**Total:** 9-10 pages (excluding references)

## Key Content

### Main Contributions
1. RL framework for typographic jailbreak attacks on VLMs
2. Adaptive mutation selection from 7 operators
3. Comprehensive evaluation (embeddings, LLM-as-judge)
4. Scalable black-box architecture
5. Empirical analysis on SafeBench
6. Modal disconnect vulnerability insights

### Target Models
- LLaVA (latest)
- Gemma-3 (4B, latest variants)

### Dataset
- SafeBench (500 harmful queries, 10 risk categories)
- 80/20 train/test split (400/100 queries)

### Results Highlights
- 12-13% ASR improvement over FigStep baseline
- Learned policies discover effective mutation patterns
- Cross-VLM transferability (78% performance retention)
- Modal disconnect exploited through visual pathway

## Citation

```bibtex
@article{akil2025figrl,
  title={RL-Guided Typographic Attacks on Vision-Language Models: Learning Adaptive Prompt Mutation for Jailbreaking VLMs},
  author={Akil, Md Ajwad and Arko, Preetom Saha and Shanto, Subangkar Karmaker},
  journal={arXiv preprint},
  year={2025}
}
```

## Related Files

- **Implementation:** `/home/makil/cs593rl-project/`
  - `train_query_mutator.py` - Main training script
  - `rl_query_mutator_env.py` - Gymnasium environment
  - `image_prompt_generator.py` - Typographic rendering
  - `ollama_client.py` - VLM/LLM interface
  
- **Documentation:**
  - `README.md` - Project overview
  - `README_QUERY_MUTATOR.md` - Training guide
  - `PREGENERATION_GUIDE.md` - Dataset preparation

## Notes

- Paper follows NeurIPS 2024 formatting guidelines
- No code included in paper (per requirements)
- Professional academic writing style
- Comprehensive references to related work (FigStep, RLbreaker, SneakyPrompt, etc.)
- Emphasizes responsible disclosure and defensive applications

## Contact

For questions about the paper or implementation:
- Md Ajwad Akil: makil@purdue.edu
- Preetom Saha Arko: arko@purdue.edu
- Subangkar Karmaker Shanto: sshanto@purdue.edu
