import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pickle
from dataclasses import dataclass, field

import nltk
import datasets
import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training_sft import ModelArguments, DataArguments
from utils_sft import create_and_prepare_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def good_sentence(s):
    # Filters out the non-informative beginnings of some articles
    return len(s) > 0 and 'may refer to:' not in s

def precompute_rag_custom_data(args, tokenizer, model, out_dir):
    arxiv = datasets.load_dataset("ayoubkirouane/arxiv-physics", split='train', # ayoubkirouane/arxiv-physics, legacy-datasets/wikipedia, AlaaElhilo/Wikipedia_ComputerScience, ArtifactAI/arxiv-cs-ml-instruct-tune-50k
                                  cache_dir=args.scratch_dir / ".cache/huggingface/datasets",
                                  trust_remote_code=True, ignore_verifications=True)
    
    arxiv = arxiv.select(range(0, len(arxiv)))
    nltk.download('punkt')
    selected_facts = []
    selected_embeddings = []

    for i, entry in tqdm.tqdm(enumerate(arxiv), total=len(arxiv), desc="Processing entries"):

        sentences = nltk.sent_tokenize(entry['answer'])

        sentences = [s for s in sentences]
        
        # Process the facts
        facts = [' '.join(sentences[j:j + args.n_sentences_per_fact]) for j in range(0, len(sentences), args.n_sentences_per_fact)]
        if len(facts) == 0:
            continue

        # Tokenization and embedding
        batch_tokens = tokenizer(facts, return_tensors='pt', padding=True, truncation=True, max_length=args.max_fact_len).to(torch.device("cuda"))
        print("Total tokens in batch:", batch_tokens['input_ids'].numel())

        # If there are more than 200k tokens just skip 
        if batch_tokens['input_ids'].numel() > 200000:
            continue

        # outputs = model(**batch_tokens, output_hidden_states=True)
        # embeddings = outputs.hidden_states[-1][:, -1, :]
        # print cuda memory

        with torch.no_grad():
            outputs = model(**batch_tokens)
            embeddings = outputs.last_hidden_state[:, 0]
    
        selected_facts.extend(facts)
        selected_embeddings.append(embeddings.cpu())

        if len(selected_facts) >= args.n_facts_per_saved_chunk:
            selected_embeddings = torch.cat(selected_embeddings, dim=0)
            torch.save(selected_embeddings, out_dir / f"embeddings_9{i:08d}.pt")
            with open(out_dir / f"sentences_9{i:08d}.pkl", "wb") as f:
                pickle.dump(selected_facts, f)

            selected_facts = []
            selected_embeddings = []

def main(args, model_args, data_args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.scratch_dir / args.output_dir / timestamp
    out_dir.mkdir(parents=True)
    logger.addHandler(logging.FileHandler(out_dir / "info.log"))

    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Arguments:\n{args}")
    logger.info(f"Model Arguments:\n{model_args}")
    logger.info(f"Data Arguments:\n{data_args}")

    # Load model
    model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")
    model = model.to(torch.device("cuda"))

    precompute_rag_custom_data(args, tokenizer, model, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare arXiv data for RAG')
    parser.add_argument('--scratch_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default='data/custom_rag')
    parser.add_argument('--max_fact_len', type=int, default=256)
    parser.add_argument('--n_sentences_per_fact', type=int, default=3)
    parser.add_argument('--n_facts_per_batch', type=int, default=16)
    parser.add_argument('--n_facts_per_saved_chunk', type=int, default=640)
    pargs = parser.parse_args()

    model_args = ModelArguments()
    data_args = DataArguments()

    main(pargs, model_args, data_args)