import numpy as np
import json

from sklearn.metrics import precision_recall_curve


def read_data(file):
    '''
    Read the json data and returns the contexts and questions.
    Train set will be used to fine tune the information retrieval models.
    Dev set will be used to validate the model.
    '''
    train = json.load(open(file, 'rb'))
    contexts, questions, qc_map = QuestionContextExtractor(train['data'])
    return contexts, questions, qc_map
    
def QuestionContextExtractor(data):
    '''
    Loop through a dataset and extract questions and their corresponding contexts.
    
    Returns:
        Questions (dictionary) - a dictionary of questions with the index of the context that responds the question
        Contexts (dictionary) - The index of the context mapped to the content of the paragraph
    
    '''
    contexts = {}
    question = {}
    qc_map = {}
    pid = -1
    qid = -1
    
    for article in data:
        for i, paragraph in enumerate(article['paragraphs']):
            pid+=1
            contexts[pid] = paragraph['context']
            for questions in paragraph['qas']:
                try: 
                    qid +=1
                    _ = questions['answers'][0]['text']
                    question[qid] = questions['question']
                    qc_map[qid] = pid
                except:
                    continue
                    
    print(f'Data contains {total+1} question/answer pairs with a short answer.'+
          f'\nThere are {len(contexts)} unique article paragraphs.')
    return contexts, question, qc_map




# def compute_errors(labels, predictions):
#     accuracy = 
#     precision = 
#     recall = 
#     f1_score =
#     mrr = 
#     return accuracy, precision, recall, f1_score, mrr

