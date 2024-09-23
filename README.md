# CLIP-txt2img-diffusers-scripts
## Example scripts for using my [fine-tuned CLIP models](https://huggingface.co/zer0int) with HuggingFace Transformers / Diffusors Pipeline. 🤗

### ➡️ `CLIP-L_and_LongCLIP-TE-Flux.py`

- Use OpenAI/CLIP-L vs. My CLIP-L and [beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP) vs. My Long-CLIP
- ⚠️ EXAMPLE SCRIPT: 'How to use CLIP' with Flux.1. Not optimized.
- ⚠️ Insert your own optimal Flux Pipeline / config for best performance!

- ✨ Pro TIP 👀 (especially [short] CLIP-L)! Hide the text-for-in-image from CLIP. If prompt `'some text here'`:
- Long `my prompt yadda [...] long yadda` | Truncate due to >77 tokens | token 78++ is: `holding a sign that says 'some lengthy text'`
- Or `my normal prompt` | PAD, PAD, PAD, PAD | token 78++ is: `holding a sign that says 'some lengthy text'`
- Result: Better image due to better CLIP-L guidance. Better TEXT in image due to hiding that part of the prompt from CLIP!

![hide-text-from-CLIP](https://github.com/user-attachments/assets/ce2a1654-59dc-46e0-8ece-3e3e0dd323f2)
