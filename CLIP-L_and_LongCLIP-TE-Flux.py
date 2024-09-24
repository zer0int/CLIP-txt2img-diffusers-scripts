import os
import torch
from diffusers import FluxPipeline
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
import warnings
warnings.simplefilter("ignore") # Stop spam of future warnings I'm seeing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model (my finetune CLIP 77 tokens, my Long finetune (248), original OpenAI (77 tokens)

clipmodel = 'long' # 'norm', 'long' (my fine-tunes) - 'oai', 'orgL' (OpenAI / BeichenZhang original)
selectedprompt = 'long' # 'tiny' (51 tokens), 'short' (75), 'med' (116), 'long' (203)

# CLIP's effective token attention is much smaller than max tokens.
# Close-to-max tokens prompt = bad. Why? More info here: https://arxiv.org/abs/2403.15378v1

# Also, remember Flux has 2 Text Encoders -> T5 takeover when prompt >> max tokens!


if clipmodel == "long":
    model_id = "zer0int/LongCLIP-GmP-ViT-L-14"
    config = CLIPConfig.from_pretrained(model_id)
    maxtokens = 248

if clipmodel == "orgL":
    model_id = "zer0int/LongCLIP-L-Diffusers"
    config = CLIPConfig.from_pretrained(model_id)
    maxtokens = 248
    
if clipmodel == "norm":
    model_id = "zer0int/CLIP-GmP-ViT-L-14"
    config = CLIPConfig.from_pretrained(model_id)
    maxtokens = 77
    
if clipmodel == "oai":
    model_id = "openai/clip-vit-large-patch14"
    config = CLIPConfig.from_pretrained(model_id)
    maxtokens = 77

if selectedprompt == "long":
    prompt = "A photo of a fluffy Maine Coon cat, majestic in size, with long, luxurious fur of silver and charcoal tones. The cat has striking heterochromatic eyes, one a deep sapphire blue, the other a brilliant emerald green, giving it an aura of mystery and intrigue. The Maine Coon is wearing a tiny, playful hat tilted slightly to the side, a miniature top hat made of soft velvet, in a deep shade of royal purple, with a golden ribbon tied around the base. The cat is wearing a delicate silver chain necklace with a small pendant in the shape of a crescent moon. The cat is sitting up proudly, holding a large wooden sign between its front paws. The sign is made from old, weathered wood. Written on the sign in elegant, hand-painted script are the words 'Long CLIP is long, Long CAT is lovely!'. The cat is sitting on a lush green patch of grass, with small wildflowers blooming around it on a sunny day with some cumulus clouds in the sky."

if selectedprompt == "med":
    prompt = "A photo of a fluffy Maine Coon cat with long fur of silver and charcoal tones. The cat has heterochromatic eyes, one eye is a deep sapphire blue, the other eye is a brilliant emerald green. It is wearing a playful miniature top hat made of soft velvet in a deep shade of royal purple. The cat is holding a large wooden sign made from old, weathered wood between its front paws. Written on the sign in hand-painted script are the words 'Long CLIP is long, Long CAT is lovely!'. Background lush green patch of grass with small wildflowers."

if selectedprompt == "short":
    prompt = "A photo of a Maine Coon cat with heterochromatic eyes, one eye is sapphire blue, the other eye is emerald green. It is wearing a miniature top hat. the cat is holding a sign made from weathered wood. Written on the sign in hand-painted script are the words 'long CLIP is long and long CAT is lovely!'. Background grass with wildflowers."
 
if selectedprompt == "tiny":
    prompt = "A photo of a Maine Coon with heterochromatic eyes, one eye is sapphire blue, the other eye is emerald green. The cat is holding a wooden sign. The sign says 'long CLIP is long and long CAT is lovely!'. Background meadow."

    
# Adjust max_position_embeddings
config.text_config.max_position_embeddings = maxtokens

clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, config=config).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=maxtokens, return_tensors="pt", truncation=True)


# BELOW ARE EXAMPLE SETTINGS FOR MAXIMUM COMPATIBILITY (4 GB VRAM).
# CONSTRUCT YOUR OWN PIPELINE TAILORED TO **YOUR** SYSTEM!
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)


# Set the custom tokenizer and text encoder from your CLIP model
pipe.tokenizer = clip_processor.tokenizer  # Replace with the custom CLIP tokenizer
pipe.text_encoder = clip_model.text_model  # Replace with the custom CLIP text encoder
pipe.tokenizer_max_length = maxtokens           # Long-CLIP token max
pipe.text_encoder.dtype = torch.bfloat16  # Ensure the text encoder uses bfloat16


# Enable low VRAM mode. Choose either (!) one of the two following:
#pipe.enable_model_cpu_offload() # tends to be just over 24 GB unless no GUI (monitor) used. :(
pipe.enable_sequential_cpu_offload() # Super slow, but <4 GB VRAM & 30 GB RAM instead

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Just to look at the tokens / confirm settings:
tokens = clip_processor(
    [prompt], padding="max_length", max_length=maxtokens, return_tensors="pt", truncation=True
)

print(f"\n------\nNumber of tokens: {tokens['input_ids'].shape[1]}")
token_ids = tokens['input_ids'][0]
print(f"Token IDs: {token_ids}")
non_padding_count = torch.sum(token_ids != clip_processor.tokenizer.pad_token_id).item()
print(f"Number of non-padding (actual prompt content) tokens: {non_padding_count}\n------\n")


seed = 425533035839096
generator = torch.manual_seed(seed)
# Pipeline run:
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=20,
    generator=generator
).images[0]

out.save(f"image_{clipmodel}_prompt-{selectedprompt}.png")
