from sentence_transformers import InputExample
from datetime import datetime
from torch.utils.data import Dataset




class SQuAD(Dataset):
    """PyTorch Dataset class for EHRs"""

    def __init__(self, queries, corpus, qc_map):
        super(SQuAD, self).__init__()
        
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.qc_map = qc_map
        
        
    def __getitem__(self, item):
        query = self.queries_id[item]
        query_text = self.queries[query]
        context = self.corpus[self.qc_map[query]]
        
        return InputExample(texts = [query_text, context])
    
        
    def __len__(self):
        return len(self.queries)

    