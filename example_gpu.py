from modeling_memoryllm import MemoryLLM
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StreamStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_token = tokenizer.eos_token_id
        self.generated_text = ""

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the most recently generated token
        token_id = input_ids[0, -1].item()
        token = self.tokenizer.decode([token_id], skip_special_tokens=True)
        self.generated_text += token
        print(token, end="", flush=True)  # Stream the token to console

        # Stop when we hit the stop token
        return token_id == self.stop_token

# Load model and tokenizer
model = MemoryLLM.from_pretrained("YuWangX/memoryllm-8b-chat")
# Check for multiple GPUs and set up distributed training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    torch.cuda.empty_cache()  # Clear any leftover memory before setting up multi-GPU
    #model = torch.nn.DataParallel(model)  # Wrap the model to distribute it across GPUs
    torch.set_autocast_enabled(True)
    model.to(device)

tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b-chat")

# Self-Update with the new context
ctx = (
    "Last week, John had a wonderful picnic with David. During their conversation, "
    "David mentioned multiple times that he likes eating apples. Though he didn't mention "
    "any other fruits, John says he can infer that David also likes bananas."
)
context_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids

model.inject_memory(context_ids, update_memory=True)

# Generation setup
messages = [{'role': 'user', "content": "What fruits does David like and how do you know that?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

# Define input_ids and attention_mask
input_ids = inputs[:, 1:]  # Remove BOS token
attention_mask = torch.ones_like(input_ids)  # Assuming no padding for simplicity

# Move tensors to the same device as the model
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Define the stopping criteria for streaming
streaming_criteria = StreamStoppingCriteria(tokenizer)

# Generate outputs with streaming
print("Response: ", end="", flush=True)  # Prepare to stream output

# Make sure to move the model, input_ids, and attention_mask to the correct device before generating
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    stopping_criteria=StoppingCriteriaList([streaming_criteria]),
)

print()  # Add a newline after the response
