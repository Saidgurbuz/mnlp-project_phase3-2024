""" Quantize using PTQ - GPTQ """

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from peft import PeftModel
import json
import torch 
import shutil

def read_jsonl_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                full_text = entry["question"] + entry["answer"]
                # if data has 5k entries break
                if len(data) == 5000:
                    break
                data.append(full_text)
    return data

# load dataset
data_paths = ["../datasets/mcqa_mmlu_train.jsonl", "../datasets/mcqa_ai2arc_train.jsonl"]
dataset = read_jsonl_files(data_paths)

# set paths
model_id = "StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL"
base_model = "microsoft/Phi-3-mini-4k-instruct"

# load peft model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code = True, torch_dtype=torch.float16)
model_peft = PeftModel.from_pretrained(model, model_id, trust_remote_code = True, torch_dtype=torch.float16)
model_peft = model_peft.merge_and_unload()
# create temporary save
model_peft.save_pretrained("../checkpoints/temp")
tokenizer.save_pretrained("../checkpoints/temp")

# quantize model
quantize_config = GPTQConfig(
    bits=8,
    group_size=128,
    desc_act=False,
    dataset=dataset,
)

model_gptq = AutoModelForCausalLM.from_pretrained("../checkpoints/temp", quantization_config = quantize_config, trust_remote_code=True, torch_dtype=torch.float16)

# delete the temporary save
shutil.rmtree("../checkpoints/temp")

print(f"Model size in GB: {sum(p.numel() for p in model_gptq.parameters())*4/1024**3}")

# # Save model and tokenizer
# model_gptq.save_pretrained("../checkpoints/quantized_gptq", use_safetensors=True)
# tokenizer.save_pretrained("../checkpoints/quantized_gptq")

# push to hub
model_gptq.push_to_hub(
 repo_id="StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL-GPTQ-8bit",
)

tokenizer.push_to_hub("StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL-GPTQ-8bit")
