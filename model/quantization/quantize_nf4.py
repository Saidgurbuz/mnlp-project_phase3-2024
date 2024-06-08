""" Quantize using PTQ - NF4 """

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL"
base_model = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=nf4_config, trust_remote_code = True)
model_nf4 = PeftModel.from_pretrained(model, model_id, quantization_config=nf4_config, trust_remote_code = True)
model_nf4 = model_nf4.merge_and_unload()

# print model size in GB, original is 14GB
print(f"Model size in GB: {sum(p.numel() for p in model_nf4.parameters())*4/1024**3}")

# save model 
model_nf4.save_pretrained("checkpoints/quantization")
# save tokenizer
tokenizer.save_pretrained("checkpoints/quantization")