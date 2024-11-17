import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Define the custom cache directory on your SSD
custom_cache_dir = "/home/panthera/NVME4TB/LLM_MODELS_BACKUPS/unsloth/Llama-3.2-11B-Vision-Instruct"

model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"

# Load the model with the specified cache directory
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=custom_cache_dir,  # Use the custom cache directory
    force_download=False,
    load_in_8bit=False,
)
processor = AutoProcessor.from_pretrained(model_id, cache_dir=custom_cache_dir)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image = Image.open('book.jpg')
image = Image.open('restaurant.jpg')
# image = Image.open('workplan.png')


messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe in great detail this image."}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=3000)
print(processor.decode(output[0]))