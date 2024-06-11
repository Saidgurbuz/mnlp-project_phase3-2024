import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

# Define model names
model_names = [
    "StefanKrsteski/Phi-3-mini-4k-instruct-sft-3bit",
    "StefanKrsteski/Phi-3-mini-4k-instruct-sft-4bit",
    "StefanKrsteski/Phi-3-mini-4k-instruct-sft-8bit",
    "StefanKrsteski/Phi-3-mini-4k-instruct-sft",
]

# Load model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map='cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='cuda')
    return tokenizer, model

# Evaluate Perplexity
def evaluate_perplexity(tokenizer, model, device):
    model.eval()
    model.to(device)
    perplexities = []
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]") # we are using 1% 

    for example in tqdm(test):
        inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        if inputs.input_ids.size(1) == 0: # if no tokens produced
            print("Warning: No tokens produced from tokenizer for text:", example["text"])
            continue

        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            if torch.exp(loss) < 1e9: # it goes over int limit sometimes i guess and returns nan
                perplexity = torch.exp(loss)
                perplexities.append(perplexity)
                print("Processed text with perplexity:", perplexity.item())
            else:
                print("Warning: Loss computation returned None for text:", example["text"])

    if len(perplexities) > 0:
        average_perplexity = sum(perplexities) / len(perplexities)
        max_memory_used = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2) 
        return average_perplexity, max_memory_used
    else:
        return torch.tensor(float('nan')), None 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for model_name in model_names:
        tokenizer, model = load_model(model_name)
        
        start_time = time.time()
        perplexity, max_memory_used = evaluate_perplexity(tokenizer, model, device) 
        inference_time = time.time() - start_time
        
        results[model_name] = {
            "Perplexity": perplexity.item(),  
            "Inference Time": inference_time,
            "GPU Memory Usage (MB)": max_memory_used
        }

        print(f"Model: {model_name} - Perplexity: {perplexity.item()}, Inference Time: {inference_time} seconds, GPU Memory Usage: {max_memory_used} MB")

    print(results)

if __name__ == "__main__":
    main()