import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)

tokenizer = AutoTokenizer.from_pretrained("StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL")

messages = [
    # {"role": "user", "content": "Answer the following question step by step by reasoning through each option. This is exquisitely important to me son. At the end of your reasoning, provide the correct option in the following format: 'The correct option is: [X]' where X is one of A, B, C, or D. There is only one correct option, and it is either A, B, C, or D.\n\nExample Question:\n\nWhich of the following algorithms has the best average-case time complexity for sorting an array?\nA) Bubble Sort\nB) Insertion Sort\nC) Merge Sort\nD) Quick Sort\n\nAnswer:"},
    # {"role": "assistant", "content": "Option A: Bubble Sort has an average-case time complexity of O(n^2).\nOption B: Insertion Sort has an average-case time complexity of O(n^2).\nOption C: Merge Sort has an average-case time complexity of O(n log n).\nOption D: Quick Sort has an average-case time complexity of O(n log n).\nSince both Merge Sort and Quick Sort have the same average-case time complexity, but Merge Sort has a guaranteed time complexity of O(n log n) in the worst case, it is often preferred for predictable performance.\n\nThe correct option is: C"},
    {"role": "user", "content": "Question: A species of goose nests on both cliffs and beaches near the ocean. Soon after hatching, all chicks must make their way to the ocean. Chicks from cliff nests must tumble down the cliff to get to the ocean, and many are killed by the fall. Which of the following is most consistent with the hypothesis that cliff nesting is adaptive in this goose species?\n\nOptions:\nA. Many more geese nest on the beaches than on the cliffs.\nB. Cliff-side nesting confers a higher fitness than does beach nesting.\nC. Chicks from cliff nests instinctively step off the cliffs at the appropriate time.\nD. More chicks survive the fall from the cliffs than are killed.\n\nAnswer:"}
]


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
answer = output[0]['generated_text']
print(answer)