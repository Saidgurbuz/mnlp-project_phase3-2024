import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from tqdm import tqdm

torch.random.manual_seed(0)

model_name = "StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
)

print("Model loading done.")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False
}

def read_jsonl_and_sample(file_path, sample_size=120000):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
            if len(data) >= sample_size:
                break
    return data

def transform_dataset_to_messages(data):
    messages = []
    for entry in data:
        message = {
            "role": "user",
            "content": f"{entry['question']} {entry['answer']}. Explain why {entry['answer']} is the correct answer, step by step by reasoning through each option. At the end of your reasoning provide the correct option and end strictly with: 'The correct option is: {entry['answer']}'\n"
        }
        messages.append(message)
    return messages

train_data = read_jsonl_and_sample("../../data/mcqa_mmlu_train.jsonl")
print("Data loading done.")

messages = transform_dataset_to_messages(train_data)
print("Data transformation to messages done.")

def generate_and_append_reasoning(messages, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        batch_size = 8
        for i in tqdm(range(0, len(messages), batch_size), desc="Generating reasoning"):
            batch = [msg['content'] for msg in messages[i:i+batch_size]]
            outputs = pipe(batch, **generation_args)
            for j, output in enumerate(outputs):
                question_and_answer = messages[i+j]['content'].split(". Explain why this is the correct choice,")[0]
                question, answer = question_and_answer.rsplit(' ', 1)
                augmented_data = {
                    "question": question,
                    "answer": answer,
                    "reasoning": output[0]['generated_text']
                }
                f.write(json.dumps(augmented_data) + '\n')

generate_and_append_reasoning(messages, 'mcqa_mmlu_reasoning_train.jsonl')

print("Augmented dataset with reasoning saved.")