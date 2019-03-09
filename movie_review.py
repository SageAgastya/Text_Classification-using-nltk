import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import random
import pickle

doc_store = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        doc_store.append((list(movie_reviews.words(fileid)), category))

# print(doc_store)

random.shuffle(doc_store)
all_words = []
for word in movie_reviews.words():
    all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(5))
word_features = list(all_words.keys())[:3000]
# print(word_features)

# print(movie_reviews.words('neg/cv000_29416.txt'))

def feature_miner(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print(feature_miner(movie_reviews.words("neg/cv000_29416.txt")))

featuresets = []
for (rev,category) in doc_store:
    featuresets.append([feature_miner(rev),category])


trainset = featuresets[:1900]
testset = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(trainset)  #not an sklearn classifier
classifier.show_most_informative_features(30)
accuracy = nltk.classify.accuracy(classifier,testset)
# print(accuracy*100)                               #not satisfactory as there is scope of better accuracy



#------------------------------------------------------------------------------------------------------------------

#training using sklearn:

multinofier = SklearnClassifier(MultinomialNB())   #less accuracy as we have to do binary classification only so we don't need softmax kind of things
multinofier.train(trainset)
# print(nltk.classify.accuracy(multinofier, testset)*100)

binafier = SklearnClassifier(BernoulliNB())    # this classifier best classifies the positive and negative sentiments
binafier.train(trainset)
print("accuracy for binafier:",nltk.classify.accuracy(binafier, trainset)*100)


savemodel = open("binafier.pickle", 'wb')  #saving the model
pickle.dump(classifier, savemodel)
savemodel.close()


"""
We tried for all types of naive bayes classifiers and found that bernoulli classifier is doing the best binary classification.
We have also classified the texts as positive and negative shadowing the impression of each word as positive and negative.
"""