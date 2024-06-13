
from dataclasses import dataclass, field
from typing import Optional, List
import torch.nn.functional as F
from transformers import set_seed, TrainingArguments
import torch
from utils_sft import create_and_prepare_model, create_datasets
from trl import SFTTrainer, SFTConfig

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    gradient_checkpointing: Optional[bool] = field(default=True)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        # The list below is for the model in the SFT example
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enablesc loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=True,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
@dataclass
class DataArguments:
    train_dataset_names: List[str] = field(
        default_factory=lambda: ["../..data/mcqa_ai2arc_train.jsonl","../..data/mcqa_mmlu_train.jsonl"],#"data/dpo/ultrafeedback/train_ultrafeedback.jsonl"], # NOTE: Add datasets here!
    )

    eval_dataset_names: List[str] = field(
        default_factory=lambda: ["../datasets/mcqa_example.jsonl"],
    )

    max_seq_length: Optional[int] = field(default=1024)
    
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )

    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )

    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def main(model_args, data_args):

    # Set seed for reproducibility
    set_seed(233)

    # Load data
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args)

    # Gradient ckpt. Gonna leave this for now however i dont think we need it DPOConfig supports checkpointing
    model.config.use_cache = not model_args.gradient_checkpointing
    model_args.gradient_checkpointing = model_args.gradient_checkpointing and not model_args.use_unsloth
    if model_args.gradient_checkpointing:
        model_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    train_dataset, validation_dataset = create_datasets(data_args.train_dataset_names, data_args.eval_dataset_names)

    args = SFTConfig(
        output_dir="../checkpoints/sft", # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        logging_steps=50,                       
        do_eval = True,
        save_steps = 500,
        load_best_model_at_end = True,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        evaluation_strategy = "steps",
        eval_steps = 500,
        fp16=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset = validation_dataset,
        peft_config=peft_config,
        max_seq_length=256,
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        }
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    # Define args 
    model_args = ModelArguments()
    data_args = DataArguments()
    main(model_args, data_args)
