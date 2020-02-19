import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import random
import pandas as pd
import string
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')
# nltk.download('wordnet')
class NaiveBayesModel:

	def __init__(self,path1,path2,cat1,cat2):
		self.data1=pd.read_csv(path1)
		self.data2=pd.read_csv(path2)
		self.cat1 = cat1
		self.cat2 = cat2
		self.text_fake=[]
		self.text_real = []
		self.all_words=[]
		self.dataset_document=[]
		self.classifier=None
		self.train_set=[]
		self.test_set=[]
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = PorterStemmer()

	# takes a string, removes the html components and returns a string
	def htmlCleanup(self,text):
		yummysoup=BeautifulSoup(text,'lxml')
		html_free_text =yummysoup.get_text()
		return html_free_text

	# takes a string, removes the punctuations and returns a string
	def removePunctuations(self,text):
		temp = ["-"]
		punc_free = "".join([c for c in text if c not in string.punctuation and c not in temp]) 
		return punc_free

	# takes a list, removes the stopwords and returns a list
	def stopwordsRemove(self,textList):
		words =[]
		for w in textList:
			if (w not in stopwords.words('english')):
				words.append(w.lower())
		# words = [w for w in textList if w not in stopwords.words('english')]
		# print(words)
		return words

	def lemmatizeWords(self,text):
		lemmatized_text = [self.lemmatizer.lemmatize(i) for i in text]
		return lemmatized_text

	def stemmWords(self,text):
		stem_text = [self.stemmer.stem(i) for i in text]
		return stem_text

	def preProcessing(self):

		# removing html content and punctuations
		for i in range(len(self.data1['title'])):
			self.data1['title'][i] = self.htmlCleanup(self.data1['title'][i])
			self.data1['title'][i] = self.removePunctuations(self.data1['title'][i])
		for i in range(len(self.data2['title'])):
			self.data2['title'][i] = self.htmlCleanup(self.data2['title'][i])
			self.data2['title'][i] = self.removePunctuations(self.data2['title'][i])

		# Splitting the text into the list of words for each row 
		self.data1["title"]= self.data1["title"].str.split(" ", n = -1, expand = False)
		self.data2["title"]= self.data2["title"].str.split(" ", n = -1, expand = False)

		# removing stopwords
		for i in range(len(self.data1['title'])):
			self.data1['title'][i] = self.stopwordsRemove(self.data1['title'][i])
			# print(self.data1['title'][i])
			self.data1['title'][i] = self.lemmatizeWords(self.data1['title'][i])
			# self.data1['title'][i] = self.stemmWords(self.data1['title'][i])		
			# print(self.data1['title'][i])
		for i in range(len(self.data2['title'])):
			self.data2['title'][i] = self.stopwordsRemove(self.data2['title'][i])
			# print(self.data2['title'][i])
			self.data2['title'][i] = self.lemmatizeWords(self.data2['title'][i])
			# self.data2['title'][i] = self.stemmWords(self.data2['title'][i])
			# print(self.data2['title'][i])

		self.text_fake = self.data1['title'].tolist()
		self.text_real = self.data2['title'].tolist()

		
		# making all the words to lowercase
		for line in self.text_fake:
			for i in range(len(line)):
				line[i] = line[i].lower()
		for line in self.text_real:
			for i in range(len(line)):
				line[i] = line[i].lower()


	def document_features(self,document,word_features):
		document_words = set(document)
		features ={}
		for word in word_features:
			features[word]  =(word in document_words)
		return features


	def formDataSet(self):
		
		for line in self.text_fake:
			self.dataset_document.append((line,self.cat1))
			for word in line:
				self.all_words.append(word)

		for line in self.text_real:
			self.dataset_document.append((line,self.cat2))
			for word in line:
				self.all_words.append(word)


		random.shuffle(self.dataset_document)

		all_words_with_frequency = nltk.FreqDist(w.lower() for w in self.all_words)
		# word_features = list(all_words_with_frequency)[:1000]
		word_features = list(all_words_with_frequency)
		
		featuresets = [(self.document_features(d,word_features),c) for(d,c) in self.dataset_document]
		

		# self.train_set = featuresets[:400]+featuresets[430:830]
		# self.test_set = featuresets[400:430]+featuresets[830:]
		# print(self.test_set)

		self.train_set, self.test_set = featuresets[:800],featuresets[800:]


	def trainModel(self):
		self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)

	def testModel(self):
		accuracy = nltk.classify.accuracy(self.classifier,self.test_set)
		print('Accuracy is '+ str(accuracy*100)+'%')
		print(self.classifier.show_most_informative_features(5))
		return accuracy
