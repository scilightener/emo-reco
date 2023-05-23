from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import time

start = time.time()

train_data, train_labels = [], []
test_data, test_labels = [], []
dev_data, dev_labels = [], []

with open('data\\train.txt', encoding='utf-8') as train, open('data\\dev.txt', encoding='utf-8') as dev, open('data\\test.txt', encoding='utf-8') as test:
    train.readline()
    dev.readline()
    test.readline()
    for i, line in enumerate(train):
        train_data.append(' '.join(line.rstrip().split('\t')[1:-1]))
        train_labels.append(line.rstrip().split('\t')[-1])
    for i, line in enumerate(test):
        test_data.append(' '.join(line.rstrip().split('\t')[1:-1]))
        test_labels.append(line.rstrip().split('\t')[-1])
    for i, line in enumerate(dev):
        dev_data.append(' '.join(line.rstrip().split('\t')[1:-1]))
        dev_labels.append(line.rstrip().split('\t')[-1])

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_data)
x_test = vectorizer.transform(test_data)

clf = MultinomialNB()
y_train = train_labels
clf.fit(x_train, y_train)

print('training time: {:.2f} seconds'.format(time.time() - start))
start = time.time()

y_pred = clf.predict(x_test)

print('prediction time: {:.4f} seconds'.format(time.time() - start))

y_true = test_labels
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
print("Accuracy: {:.2f}".format(accuracy))
print("f1: {:.2f}".format(f1))