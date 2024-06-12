import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import tqdm
from models.model_dpo import AutoDPOModelForCausalLM
import faiss
import numpy as np
from abc import ABC, abstractmethod
import random
from pathlib import Path
import pickle
from transformers import AutoModel, AutoTokenizer


class ANNSearch(ABC):
    @abstractmethod
    def get_top_k(self, batch_query, k=1):
        pass

    @staticmethod
    def create(ann_type, **kwargs):
        if ann_type == "faiss":
            return FaissANNSearch(**kwargs)
        elif ann_type == "naive":
            return NaiveANNSearch(**kwargs)
        else:
            raise ValueError(f"Unknown ANNSearch type: {ann_type}")



class FaissANNSearch(ANNSearch):
    def __init__(self, document_dir):
        self.index = None
        self.sentences = []
        self.embeddings = []

        sentence_files = sorted(list(Path(document_dir).glob("sentences*.pkl")))
        data_files = sorted(list(Path(document_dir).glob("*.pt")))
        assert len(sentence_files) == len(data_files), "Mismatch between sentence and data files"

        for sentence_file, data_file in tqdm.tqdm(zip(sentence_files, data_files), total=len(sentence_files)):
            with open(sentence_file, "rb") as f:
                self.sentences.extend(pickle.load(f))
            loaded_embeddings = torch.load(data_file).numpy()
            if not loaded_embeddings.flags['C_CONTIGUOUS']:
                loaded_embeddings = np.ascontiguousarray(loaded_embeddings)
            self.embeddings.append(loaded_embeddings)

        self.embeddings = np.vstack(self.embeddings).astype(np.float32)
        if not self.embeddings.flags['C_CONTIGUOUS']:
            self.embeddings = np.ascontiguousarray(self.embeddings)

        # Initialize the FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def get_top_k(self, batch_query, k=1):
        batch_query_np = batch_query.cpu().numpy()
        if not batch_query_np.flags['C_CONTIGUOUS']:
            batch_query_np = np.ascontiguousarray(batch_query_np)
            
        distances, indices = self.index.search(batch_query_np, k)
        top_sentences = [[self.sentences[idx] for idx in indices[i]] for i in range(len(batch_query))]
        return distances, top_sentences


class NaiveANNSearch(ANNSearch):
    def __init__(self, document_dir):
        self.sentences = []
        self.data = []

        # has extension .pkl or .pt
        sentence_files = sorted(list(Path(document_dir).glob("sentences*.pkl")))
        data_files = sorted(list(Path(document_dir).glob("*.pt")))
        assert len(sentence_files) == len(data_files), \
            f"Number of sentence files and data files do not match, {len(sentence_files)} != {len(data_files)}"

        self.chunk_indices = []
        for i, (sentence_file, data_file) in tqdm.tqdm(enumerate(zip(sentence_files, data_files)),
                                                       total=len(sentence_files), desc="Loading RAG documents"):
            with open(sentence_file, "rb") as f:
                self.sentences.append(pickle.load(f))
            self.data.append(F.normalize(torch.load(data_file).to(torch.float32), dim=1))

        for i, data in enumerate(self.data):
            self.chunk_indices.extend([(i, j) for j in range(data.shape[0])])

    def get_top_k(self, batch_query, k=1):
        cosine_sims = []
        batch_query = F.normalize(batch_query, dim=1)  # Normalize batch_query
        if torch.cuda.is_available():
            batch_query_cuda = batch_query.cuda()
        else:
            batch_query_cuda = batch_query

        for sentences, data in tqdm.tqdm(zip(self.sentences, self.data), total=len(self.sentences),
                                        desc="Computing cosine similarities"):
            if torch.cuda.is_available():
                data_cuda = data.cuda()
            else:
                data_cuda = data
            cosine_sims.append(batch_query_cuda @ data_cuda.T)
            if torch.cuda.is_available():
                del data_cuda

        cosine_sims = torch.cat(cosine_sims, dim=1)
        topk_vals, topk_inds = torch.topk(cosine_sims, k=k, dim=1)

        top_sentences = []
        for i in range(topk_inds.shape[0]):
            top_row = topk_inds[i]
            top_sentences.append([self.sentences[self.chunk_indices[j][0]][self.chunk_indices[j][1]] for j in top_row])

        return topk_vals, top_sentences


class AutoRAGModelForCausalLM(AutoDPOModelForCausalLM):
    """
    An autoregressive model doing RAG to answer multiple choice questions.
    """

    transformers_parent_class = AutoDPOModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

    @staticmethod
    def apply_facts_template(facts): #LlamaIndex prompt
        return "We have provided context information below." + "\n---------------------\n".join(facts) + "\n---------------------\n" + "Given this information, please answer the questions."  

    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()

    ####################################################################################

    def __init__(self, pretrained_model, ann_args, topk, device=torch.device("cuda"), **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to any `CustomModule` class.
        """
        super().__init__(pretrained_model, **kwargs)

        self.ann = ANNSearch.create(**ann_args)
        self.topk = topk

        self.embedding_model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.embedding_model.eval()
        self.embeddings_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (dict of list):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (PreTrainedTokenizerBase): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (dict): A dictionary containing the model predictions given input questions.
        """
        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================

        output_dict = {"preds": []}

        # TODO (Optional): pre-compute the nearest neighbors for the example questions too, so that the
        # few shots are closer in the format to the final question

        example_questions = [
            {
                'user': 'A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations:\nADD Rs1, Rs2, Rd (Add Rs1 to Rs2 and put the sum in Rd)\nMUL Rs1, Rs2, Rd (Multiply Rs1 by Rs2 and put the product in Rd)\nAn operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence.\nConsider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?\n\nOptions:\nA. 5\nB. 6\nC. 7\nD. 8\n\nAnswer:',
                'assistant': 'B'},
            {
                'user': 'Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p''(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\n\nOptions:\nA. ST = 0\nB. ST = T\nC. ST = TS\nD. ST - TS is the identity map of V onto itself.\n\nAnswer:',
                'assistant': "D"},
            {
                'user': 'Which of the following represents an accurate statement concerning arthropods?\n\nOptions:\nA. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.\n\nAnswer:',
                'assistant': "B"}
        ]

        # Omit the `Question:` part from the batch, omit first 9 characters from each question 
        # find the embeddings
        with torch.no_grad():
            enc_tokens = self.embeddings_tokenizer([q[9:] for q in batch["question"]], return_tensors='pt', padding=True, truncation=True)
            outputs = self.embedding_model(**enc_tokens)
            embeddings = outputs.last_hidden_state[:, 0]

        topk_vals, top_sentences = self.ann.get_top_k(embeddings, k=self.topk)        
        # delete outputs and embeddings to free up mem 
        del outputs
        del embeddings
        torch.cuda.empty_cache()

        # tokenize the questions
        for qi, question in enumerate(batch["question"]):
            messages = []
            for ex_question in example_questions:
                messages.append({'role': 'user', 'content': ex_question['user']})
                messages.append({'role': 'assistant', 'content': ex_question['assistant']})

            # append documents only if cosine similarity is above the threshold
            effective_docs = [s for i, s in enumerate(top_sentences[qi]) if topk_vals[qi][i] >= 0.69]
            if effective_docs:
                extended_question = self.apply_facts_template(effective_docs) + "\n\n" + question
            else:
                extended_question = question

            messages.append({'role': 'user', 'content': extended_question})
            input_ids = tokenizer.apply_chat_template(messages, add_generation_promt=True, return_tensors="pt").to(self.device)

            flag = 0

            for _ in range(10):  # generate max 10 new tokens to try and find the answer
                outputs = self.pretrained_model(input_ids=input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                next_token = tokenizer.decode(next_token_id).strip()
                print(f"{next_token} , correct={batch['answer'][qi]}")
                if next_token in ['A', 'B', 'C', 'D']:
                    flag = 1
                    break

            if flag:
                output_dict["preds"].append(next_token)
            else:
                output_dict["preds"].append("C")  # Fallback if no answer is found

            flag = 0

        return output_dict
        # You need to return one letter prediction for each question.
        # ======================================================================
        ########################################################################
