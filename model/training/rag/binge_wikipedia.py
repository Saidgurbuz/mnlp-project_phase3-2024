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


def good_sentence(s):
    # Filters out the non-informative beginnings of some articles
    return len(s) > 0 and 'may refer to:' not in s


def precompute_rag_wikipedia(args, tokenizer, model, out_dir):
    assert args.n_facts_per_saved_chunk % args.n_facts_per_batch == 0, \
        f"Got {args.n_facts_per_batch=} and {args.n_facts_per_saved_chunk=}"

    # Load the Wikipedia dataset
    wiki = datasets.load_dataset("wikipedia", args.wiki_dump,
                                 cache_dir=args.scratch_dir / ".cache/huggingface/datasets",
                                 trust_remote_code=True)
    wiki = wiki['train']

    # Disable when debugging is done
    # wiki = wiki.select(range(10))

    nltk.download('punkt')

    selected_facts = []
    selected_embeddings = []

    cur_batch = []

    last_emb = None
    sum_cossim_same = 0
    sum_cossim_other = 0

    # Deterministically shuffle the dataset
    wiki = wiki.shuffle(seed=239)
    saved_art_ids = []
    chunk_art_ids = []

    # sample_q = "College students constantly hear the praises of education.We have all become used to believing that a college education is always a guarantee of an easier life.I was nine years old when my fourth-grade teacher presented me with a task, to write down all of the things I wanted in my life.I filled my paper with things like: own a big house and have servants; be rich and have a good job.The next day my teacher handed back my paper and in red ink she wrote: \" GO TO COLLEGE.\" For a long time, I was convinced that once I obtained an education, BAM! Life would be easier. However, education cannot promise all wishes, dreams, and desires.Society must reject the foolish idea that a college education's main purpose is to satisfy our desires and secure success.Like most challenging things, education is a gamble   in which results depend entirely on people's ability to look past their wants to see the realism and reason behind their wants. For instance, my first year of college, I took a sociology class.In class, we were taught that Third World countries were poor.We learned that our quality of life would be almost impossible for an average person in those countries.I began to examine my own desire to be rich.To always go after money felt selfish when knowing others had none at all.Learning about other society's financial situations forced me to look beyond what I wanted. Through the process of education, everything once desired is tested.Wanting something no longer is enough; it's more important to examine why we want it and whether we really want it.When my desire for money changed, everything changed.I stopped longing for money-driven careers and stopped valuing the people who had them.I began to examine the things I purchased and my reason for wanting them. Education is a tool to be used to develop and advance our desires, so we can discover the things that are truly significant in life.Education is a source to expand our society to see beyond the superficial   appeals and the \"quick fixes\" , leaving the belief of an effortless life behind in order to desire a meaningful one. What's the main idea of the passage?\n\nOptions:\nA. College education promises an effortless life.\nB. College education tests and guides our life desires.\nC. College education offers solutions to social problems.\nD. College education turns young people into gamblers"
    # tokens = tokenizer(sample_q, return_tensors='pt').to(torch.device("cuda"))
    # print(f"number of tokens: {len(tokens['input_ids'])}")

    for i, wiki_entry in tqdm.tqdm(enumerate(wiki), total=len(wiki), desc="Processing Wikipedia entries"):
        sentences = nltk.sent_tokenize(wiki_entry['text'])
        sentences = [s for s in sentences if good_sentence(s)]
        if len(sentences) == 0:
            continue
        sentences = sentences[:args.n_sentences_per_fact * args.n_facts_per_article]
        # Each fact is a concatenation of n_sentences_per_fact consecutive sentences from the article
        facts = [' '.join(sentences[i:i + args.n_sentences_per_fact]) \
                 for i in range(0, len(sentences), args.n_sentences_per_fact)]
        cur_batch.extend(facts)
        chunk_art_ids.append(wiki_entry['id'])

        if len(cur_batch) >= args.n_facts_per_batch:
            batch_tokens = tokenizer(cur_batch, return_tensors='pt', padding=True, truncation=True,
                                     max_length=args.max_fact_len).to(torch.device("cuda"))
            # get lengths of each fact
            # fact_lens = batch_tokens['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1)

            # get the embeddings of the last token of each fact
            outputs = model(**batch_tokens, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1][range(len(cur_batch)), -1, :]

            selected_facts.extend(cur_batch)
            selected_embeddings.append(embeddings.cpu())

            del outputs
            del embeddings
            del batch_tokens

            cur_batch = []

        if len(selected_facts) >= args.n_facts_per_saved_chunk:
            selected_embeddings = torch.cat(selected_embeddings, dim=0)
            # logger.info(f"Selected {selected_embeddings.size(0)} sentences")
            # logger.info(f"Average cosine similarity between two sentences from the same article:"
            #             f" {sum_cossim_same / (len(wiki) - 1)}")
            # logger.info(f"Average cosine similarity between two sentences from different articles:"
            #             f" {sum_cossim_other / (len(wiki) - 1)}")

            torch.save(selected_embeddings, out_dir / f"embeddings_{i:08d}.pt")
            with open(out_dir / f"sentences_{i:08d}.pkl", "wb") as f:
                pickle.dump(selected_facts, f)

            saved_art_ids.extend(chunk_art_ids)
            with open(out_dir / f"saved_article_ids.pkl", "wb") as f:
                pickle.dump(saved_art_ids, f)

            selected_facts = []
            selected_embeddings = []
            chunk_art_ids = []

    # return selected_facts, selected_embeddings


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
    logger.info(model)
    # tokenizer = tokenizer.to(torch.device("cuda"))

    precompute_rag_wikipedia(args, tokenizer, model, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binge Wikipedia into a source for RAG')
    parser.add_argument('--scratch_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default='data/rag_wikipedia')
    parser.add_argument('--wiki_dump', type=str, default="20220301.en")
    parser.add_argument('--max_fact_len', type=int, default=256)
    # parser.add_argument('--agg_strategy', type=str, choices=['last', 'avg'], default='last')
    parser.add_argument('--n_sentences_per_fact', type=int, default=3)
    parser.add_argument('--n_facts_per_article', type=int, default=1)
    parser.add_argument('--n_facts_per_batch', type=int, default=16)
    parser.add_argument('--n_facts_per_saved_chunk', type=int, default=64 * 10)
    pargs = parser.parse_args()

    model_args = ModelArguments()
    data_args = DataArguments()

    main(pargs, model_args, data_args)
