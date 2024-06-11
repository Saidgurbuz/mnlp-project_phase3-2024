import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import tqdm
from models.model_dpo import AutoDPOModelForCausalLM

from abc import ABC, abstractmethod
import random
from pathlib import Path
import pickle


class ANNSearch(ABC):
    @abstractmethod
    def get_top_k(self, batch_query, k=1):
        pass

    @staticmethod
    def create(ann_type, **kwargs):
        if ann_type == "naive":
            return NaiveANNSearch(**kwargs)
        else:
            raise ValueError(f"Unknown ANNSearch type: {ann_type}")


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

        # TODO DELETE THIS
        # self.sentences = self.sentences[:2]
        # self.data = self.data[:2]

        for i, data in enumerate(self.data):
            self.chunk_indices.extend([(i, j) for j in range(data.shape[0])])

    def get_top_k(self, batch_query, k=1):
        cosine_sims = []
        batch_query = F.normalize(batch_query, dim=1)
        batch_query_cuda = batch_query.cuda()
        for sentences, data in tqdm.tqdm(zip(self.sentences, self.data), total=len(self.sentences),
                                         desc="Computing cosine similarities"):
            data_cuda = data.cuda()
            cosine_sims.append(batch_query_cuda @ data_cuda.T)
            del data_cuda

        cosine_sims = torch.cat(cosine_sims, dim=1)
        topk_vals, topk_inds = torch.topk(cosine_sims, k=k, dim=1)

        # print(f"Topk vals: {topk_vals}")

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
    def apply_facts_template(facts):
        return "Here is some information that may or may not be useful for the task:\n\n" + "\n\n".join(facts)

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

        # find the embeddings
        tokens = tokenizer(batch["question"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.pretrained_model(**tokens, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, -1, :]
        topk_vals, top_sentences = self.ann.get_top_k(embeddings, k=self.topk)

        # tokenize the questions
        for qi, question in enumerate(batch["question"]):
            for num_facts in reversed(range(1, self.topk + 1)):
                messages = []
                for ex_question in example_questions:
                    messages.append({'role': 'user', 'content': ex_question['user']})
                    messages.append({'role': 'assistant', 'content': ex_question['assistant']})

                extended_question = self.apply_facts_template(top_sentences[qi][:num_facts]) + "\n\n" + question
                messages.append({'role': 'user', 'content': extended_question})

                input_ids = tokenizer.apply_chat_template(messages, add_generation_promt=True, return_tensors="pt").to(
                    self.device)
                if input_ids.shape[1] < 4000:
                    break

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
