from math import sqrt, exp, pi
from collections import Counter
from random import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def my_bow_vectorizer(texts):
    vocab = Counter()
    for text in texts:
        vocab.update(text.split())
        
    vectors = []
    for text in texts:
        vector = [0] * len(vocab)
        for i, word in enumerate(vocab.keys()):
            vector[i] = text.split().count(word)
        vectors.append(vector)
    
    return vectors

class NaiveBayes:
    def __init__(self):
        self.summary = []

    def classSeparator(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector, classVal = dataset[i][:-1], dataset[i][-1]
            if classVal not in separated:
                separated[classVal] = []
            separated[classVal].append(vector)  # Append only the feature vector
        return separated


    def mean(self, numList):
        try:
            return sum(numList) / float(len(numList))
        except TypeError as e:
            print(f"Error in mean: {e}")
            #print(f"numList: {numList}")
            for i, item in enumerate(numList):
                print(f"Item {i} is of type {type(item)}")
            raise


    def stdev(self, numList):
        avg = self.mean(numList)
        variance = sum([(x - avg) ** 2 for x in numList]) / float(len(numList) - 1)
        return sqrt(variance)

    def summarize(self, dataset):
        try:
            summary = [(self.mean(attribute), self.stdev(attribute), len(attribute)) for attribute in zip(*dataset)]
        except TypeError as e:
            print(f"Error in summarize: {e}")
            #print(f"Dataset: {dataset}")
            raise
        return summary

    def classSummarize(self, dataset):
        try:
            separated = self.classSeparator(dataset)
            summaries = {}
            for classValue, instances in separated.items():
                summaries[classValue] = self.summarize(instances)
        except TypeError as e:
            print(f"Error in classSummarize: {e}")
            #print(f"Dataset: {dataset}")
            raise
        return summaries


class ImplementNB:
    def __init__(self, filename):
        self.filename = filename
        self.dataset = self.readFile()
        self.preprocess()
        self.nb = NaiveBayes()

    def readFile(self):
        data = []
        with open(self.filename, 'r') as f:
            for row in f:
                msg, label = row.strip().split('\t')  # Swapped label and msg
                data.append([msg, label])
        return data


    def preprocess(self):
        X = my_bow_vectorizer([msg for msg, _ in self.dataset])
        y = [0 if label == 'ham' else 1 for _, label in self.dataset]
        self.dataset = list(zip(X, y))

    def evaluate(self):
        numFolds = 5
        foldSize = len(self.dataset) // numFolds
        shuffle(self.dataset)
        folds = [self.dataset[i:i + foldSize] for i in range(0, len(self.dataset), foldSize)]

        for i in range(numFolds):
            testSet = folds[i]
            trainSet = [x for fold in folds[:i] + folds[i + 1:] for x in fold]
            print("Dataset before classSummarize:", trainSet)

            self.nb.classSummarize(trainSet)

            predictions = []
            for row in testSet:
                probabilities = self.nb.classProbability(self.nb.summary, row[:-1])
                bestLabel, bestProb = None, -1
                for classValue, probability in probabilities.items():
                    if bestLabel is None or probability > bestProb:
                        bestProb = probability
                        bestLabel = classValue
                predictions.append(bestLabel)

            actual = [label for _, label in testSet]

            accuracy = accuracy_score(actual, predictions)
            precision = precision_score(actual, predictions, average='weighted')
            recall = recall_score(actual, predictions, average='weighted')
            f1 = f1_score(actual, predictions, average='weighted')

            print(f"Fold {i + 1} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# Main code
filename = 'script.txt'  # Replace with the path to your file
classifier = ImplementNB(filename)
classifier.evaluate()
