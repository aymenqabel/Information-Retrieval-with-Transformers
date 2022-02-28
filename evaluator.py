from sentence_transformers.evaluation import SentenceEvaluator
from utils import *
from tqdm import tqdm
import torch
from sentence_transformers import util
import random

class accuracyevaluator(SentenceEvaluator):
    def __init__(self, contexts, questions, qc_map):
        self.contexts = contexts
        self.questions = questions
        self.qc_map = qc_map
        self.accuracy = []
        self.mrrlist = []

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        predictions, test_labels = [], []
        corpus_embeddings = model.encode(self.contexts, convert_to_tensor=True, show_progress_bar=False)
        elements_to_test = random.sample(list(self.qc_map.keys()), 1000)
        for qid in elements_to_test:
            query = self.questions[qid]
            query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=10)[1].cpu().detach().numpy()

            predictions.append(top_results)
            test_labels.append(self.qc_map[qid])
        
            acc,  mrr = compute_errors(test_labels, predictions)
        self.accuracy.append(acc)
        self.mrrlist.append(mrr)
        print(acc, mrr)
        return 1-mrr