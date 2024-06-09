import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pickle

import nltk
import datasets
import tqdm
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training_sft import ModelArguments, DataArguments
from utils_sft import create_and_prepare_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def precompute_rag_wikipedia(args, tokenizer, model):
    # Load the Wikipedia dataset
    wiki = datasets.load_dataset("wikipedia", args.wiki_dump,
                                 cache_dir=args.scratch_dir / ".cache/huggingface/datasets",
                                 trust_remote_code=True)
    wiki = wiki['train']

    # Disable when debugging is done
    # wiki = wiki.select(range(10))

    nltk.download('punkt')

    selected_sentences = []
    selected_embeddings = []

    last_emb = None
    sum_cossim_same = 0
    sum_cossim_other = 0

    for i, wiki_entry in tqdm.tqdm(enumerate(wiki), total=len(wiki), desc="Processing Wikipedia entries"):
        sentences = nltk.sent_tokenize(wiki_entry['text'])
        sentences = [s for s in sentences if len(s) <= args.max_sentence_len]
        sentences = sentences[:args.first_n_sentences]
        selected_sentences.extend(sentences)
        # print(sentences[0])
        tokens = [tokenizer(s, return_tensors='pt').to(torch.device("cuda")) for s in sentences]
        # print(tokens[0])
        # compute embeddings from a generative model
        outputs = [model(**t, output_hidden_states=True) for t in tokens]

        def get_emb(out):
            if args.agg_strategy == 'last':
                return out.hidden_states[-1][0, -1, :]
            elif args.agg_strategy == 'avg':
                return out.hidden_states[0, ...].mean(dim=0)

        embeddings = [get_emb(out) for out in outputs]
        embeddings = torch.stack(embeddings, dim=0)

        if last_emb is not None:
            cossim_same = torch.nn.functional.cosine_similarity(embeddings[0, :], embeddings[1, :], dim=-1).mean()
            cossim_other = torch.nn.functional.cosine_similarity(embeddings[0, :], last_emb, dim=-1).mean()
            sum_cossim_same += cossim_same
            sum_cossim_other += cossim_other
        last_emb = embeddings[0, :]

        # Probably cannot fit all embeddings on a GPU
        selected_embeddings.append(embeddings.cpu())

    selected_embeddings = torch.cat(selected_embeddings, dim=0)
    logger.info(f"Selected {selected_embeddings.size(0)} sentences")
    logger.info(f"Average cosine similarity between two sentences from the same article:"
                f" {sum_cossim_same / (len(wiki) - 1)}")
    logger.info(f"Average cosine similarity between two sentences from different articles:"
                f" {sum_cossim_other / (len(wiki) - 1)}")

    return selected_sentences, selected_embeddings


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
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args)
    model = model.to(torch.device("cuda"))
    # tokenizer = tokenizer.to(torch.device("cuda"))

    sentences, embeddings = precompute_rag_wikipedia(args, tokenizer, model)

    torch.save(embeddings, out_dir / "embeddings.pt")
    with open(out_dir / "sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binge Wikipedia into a source for RAG')
    parser.add_argument('--scratch_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default='data/rag_wikipedia')
    parser.add_argument('--wiki_dump', type=str, default="20220301.en")
    parser.add_argument('--max_sentence_len', type=int, default=400)
    parser.add_argument('--agg_strategy', type=str, choices=['last', 'avg'], default='last')
    parser.add_argument('--first_n_sentences', type=int, default=3)
    pargs = parser.parse_args()

    model_args = ModelArguments()
    data_args = DataArguments()

    main(pargs, model_args, data_args)
