import re
from typing import Any, Literal
import time
from sklearn.metrics import accuracy_score, f1_score
import random


emotions = ['angry', 'happy', 'sad', 'others']
emotion2word = {
    # angry
    0: [
        'angry', 'poisonous', 'damn', 'hate', 'infuriate', 'irate', 'kill', 'mad', 'loathe', 'irate', 'pissed', 'rage', 'stupid', 'suck', 'terrible', 'ugly', 'useless'
        ],
        
    # happy
    1: [
        'happy', 'amazing', 'awesome', 'blissful', 'celebrated', 'cheerful', 'delighted', 'ecstatic', 'elated', 'enchanting', 'enjoy', 'excelent', 'exciting', 'fantastic', 'fun', 'glad', 'gleeful', 'good', 'great', 'joy', 'joyful', 'lovely', 'maginficent', 'nice', 'pleasant', 'pleased', 'superb', 'wonderful'
        ],

    # sad
    2: [
        'sad', 'alone', 'cry', 'depressed', 'despair', 'dismay', 'down', 'grief', 'heartbroken', 'lonely', 'lost', 'miserable', 'nothing', 'pain', 'why', 'tears', 'unhappy'
        ]
}

word2emotion = {word: emotion for emotion in emotion2word for word in emotion2word[emotion]}
negation_tokens = set(['no', 'not', "ain't", "don't", 'none', "isn't", "aren't", 'non', 'neither', 'nor'])

def get_emotion(text: str) -> Literal['angry', 'happy', 'sad', 'others']:
    # return random.choice(['angry', 'happy', 'sad', 'others']) even this has accuracy .25 and f1 .33 lol
    # return 'others' has accuracy .85 and weighted f1 .78 lol
    text = text.lower()
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
    res = [0, 0, 0, 1]
    for i, token in enumerate(tokens):
        if token in word2emotion: # if there was a negation token among the last two tokens
            if i >= 1 and tokens[i-1] in negation_tokens or \
                i >= 2 and tokens[i-2] in negation_tokens:
                continue
            res[word2emotion[token]] += 1

    return emotions[argmax(res)]
    # return 'angry'

def argmax(arr: list[Any]) -> int:
    max_val = arr[0]
    max_index = 0
    
    for i in range(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_index = i
            
    return max_index

test_data, test_labels = [], []
with open('data\\test.txt', encoding='utf-8') as test:
    for i, line in enumerate(test):
        test_data.append(' '.join(line.rstrip().split('\t')[1:-1]))
        test_labels.append(line.rstrip().split('\t')[-1])

start = time.time()
y_pred = [get_emotion(item) for item in test_data]

print('prediction time: {:.4f} seconds'.format(time.time() - start))

y_true = test_labels
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
print("Accuracy: {:.2f}".format(accuracy))
print("f1: {:.2f}".format(f1))