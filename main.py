# Himanshu Singh
# 2017291

from ClassiferModel import NaiveBayesModel


path1 ="politifact_fake.csv"
path2 ="politifact_real.csv"
category1='fake'
category2='real'
iterations = 10
total_accuracy=0

print('Let\'s Begin with Sentiment Analysis')
print('(Note - 10 iterations are used to take an average of accuracy)')

for i in range(10):
	print('Round '+str(i))
	m = NaiveBayesModel(path1,path2,category1,category2)
	m.preProcessing()
	m.formDataSet()
	m.trainModel()
	total_accuracy+=m.testModel()

print('\n')
print('Average of accuracy for '+str(iterations)+' iterations is '+str((total_accuracy/iterations)*100)+' %')
