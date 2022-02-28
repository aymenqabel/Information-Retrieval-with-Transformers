import numpy as np
import json
import urllib.request  
import os
import os.path
from os import path

def download_data():
    '''
    Download the data needed for training and testing
    '''
    train = os.path.join("data/squad/", 'train-v1.1.json')
    test = os.path.join("data/squad/", 'dev-v1.1.json')
    if not(path.exists(train)):
        print("Downloading Data")
        url_train = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'
        urllib.request.urlretrieve(url_train, train)
    if not(path.exists(test)):
        url_test = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
        urllib.request.urlretrieve(url_test, test)
        
def read_data(file):
    '''
    Read the json data and returns the contexts and questions.
    Train set will be used to fine tune the information retrieval models.
    Dev set will be used to validate the model.
    '''
    train = json.load(open(file, 'rb'))
    contexts, questions, qc_map = question_context_extractor(train['data'])
    return contexts, questions, qc_map
    
def question_context_extractor(data):
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
                    
    print(f'Data contains {qid+1} question/answer pairs with a short answer.'+
          f'\nThere are {len(contexts)} unique article paragraphs.')
    return contexts, question, qc_map




def compute_errors(labels, predictions):
    '''
    Compute the accuracy and the MRR of the model in the test set
    Returns:
        Accuracy - Accuracy of the model
        MRR - List of MRR@k for different values of k
    '''
    mrr = np.zeros_like(labels, dtype=np.float64)
<<<<<<< HEAD
    prediction = np.ones((len(labels), len(predictions[0])))
=======
    prediction = np.zeros((len(labels), len(predictions[0])))
>>>>>>> 6f10c5fe2ca8fe681d7390bc44a58d8e180f431e
    for i in range(len(predictions)):
      for j in range(len(predictions[0])):
          if predictions[i][j] == labels[i]:
              mrr[i] = 1/(1+j)
              break 
          else:
            prediction[i][j] = 0

    list_accuracies = np.mean(prediction, axis= 0)
    accuracy = list_accuracies[0]
    mrr_score = np.mean(mrr)
    return accuracy, mrr_score, list_accuracies


def save_json(path_result, name, x):
    """
    Saves x into path_result with the given name
    """
    with open(os.path.join(path_result, f'{name}.json'), 'w') as f:
        json.dump(x, f, indent=4)
