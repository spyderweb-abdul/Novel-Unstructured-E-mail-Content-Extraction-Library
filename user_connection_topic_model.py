import os
import sys
import pandas as pd
import time
import datetime
from datetime import timedelta
from pprint import pprint

import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers import LdaMallet
from gensim.models.ldamodel import LdaModel
# Plotting tools

import matplotlib.pyplot as plt

import mysql.connector
import spacy

from smart_open import open

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def main():

	start_time = time.monotonic()

	mydb = mysql.connector.connect(host = "localhost", user = "root", password = "", database = "enron_project_db")
	mycursor = mydb.cursor()

	print('Building Corpus...')

	sql_select_email_ids = mycursor.execute(" SELECT DISTINCT (connection_id) FROM communication_bow_table")
	sql_result = mycursor.fetchall()

	#length of documents, D
	N_body = len(sql_result)


	#for each email ids, we want to extract the terms, absolute/relative term frequency. We have them stored in a table prior
	conn_id_list = []
	word_bucket = []
	for conn_ids in sql_result:

		fetch_terms = mycursor.execute(" SELECT word FROM communication_bow_table WHERE connection_id = %s ", conn_ids)
		term_result = mycursor.fetchall()

		word_list = []
		for w in term_result:
			word = w[0]
			#print("file id: {}\t word: {} \t tf_idf: {}".format(file_id, word, tf_idf))
			word_list.append(word)

		word_bucket.append(word_list)

		conn_id_list.append(conn_ids[0])

	#print(word_bucket)
	#print(conn_id_list)

	#create a dictionary
	wordIDs = corpora.Dictionary(word_bucket)

	#print(wordIDs)
	#Dictionary(984 unique tokens: ['abroad', 'access', 'accessible', 'address', 'answer']...)

	#term document frequency
	corpus = [wordIDs.doc2bow(word) for word in word_bucket]
	#print(corpus)
	
	"""
	Now, build topic model
	alpha and eta hyperparameter default = 1.0/num_topics
	chunksize = number of document in each training
	update_every = determines how  often model param should be updated
	passes = total number of training passes
	"""

	mallet_path = "c:/mallet-2.0.8/bin/mallet"
	#To install and configure mallet, visit: https://programminghistorian.org/en/lessons/topic-modeling-and-mallet
	
	print('\nPretraining and Optimal Coherence Value computation...')

	#finding optimal number of K
	def compute_coherence_value(dictionary, corpus, texts, limit, start, step, mallet_path):

		coherence_values = []
		model_list = []
		for num_topics in range(start, limit, step):
			model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
			#model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
			model_list.append(model)

			coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
			coherence_values.append(coherencemodel.get_coherence())

		return model_list, coherence_values
	
	limit=50; start=2; step=8;

	model_list, coherence_values = compute_coherence_value(wordIDs, corpus, word_bucket, limit, start, step, mallet_path)
	#model_list, coherence_values = compute_coherence_value(wordIDs_tfidf, corpus_tfidf, word_bucket_tfidf, limit, start, step, mallet_path)

	#show graph

	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel("Num of Topics")
	plt.ylabel("Coherence Score")
	plt.legend(("Coherence Values"), loc='best')
	#plt.show()	
	plt.savefig(r'../enron analysis/statistics/topic model stats/chart_plot_comm_graph_'+str(limit)+'.png')

	#x = (2, 8, 14, 20, 26, 32, 38)
	#coherence_values = [0.3281766898633087, 0.47847379519667743, 0.5208091162327398, 0.5736633177260627, 0.584419650624877, 0.5951845106795495, 0.5944736315101253]
	
	count = 0
	topic_value_dict = {}
	for m, cv in zip(x, coherence_values):
		topic_value_dict[count] = {'k': m, 'co_val': cv} 
		print("Num of Topics: ", m, " Corresponding coherence value: ", round(cv, 4))
		count += 1
	#print(topic_value_dict)

	compare = 0
	for key, value in topic_value_dict.items():
		if value['co_val'] > compare:
			compare = value['co_val']
			modelID = key
			no_of_k_topics = value['k']

	print('\nCount No: ', modelID, 'Topic Number: ', no_of_k_topics)

	optimal_model = model_list[modelID]
	model_topics = optimal_model.show_topics(num_topics=no_of_k_topics, formatted=False)
	#print(optimal_model.print_topics(num_words=10))
	#print('\n', model_topics)
	#print('\n', optimal_model[corpus])

	"""
	To find the dominant topic in each sentence, we find the topic number that has
	the highest percentage contribution in that document.
	corpus = [[(0, 0.020711500974658875), (1, 0.017625081221572456), (2, 0.010680636777128008), (3, 0.01839668615984406), (4, 0.024569525666016907),...]]
	"""
	#nmf_temp_file = datapath('nmf_model')
	optimal_model.save('lda_main.model')

	print('\nTraining NMF Model...')
	#optimal_model = LdaMallet.load('lda_main.model')
	#optimal_model.update(corpus)

	#for i, row in enumerate(optimal_model[corpus_tfidf]):
	for i, row in enumerate(optimal_model[corpus]):


		i = conn_id_list[i]

		row = sorted(row, key=lambda x: (x[1]), reverse=True)
		#print('\n', row)

		#get the dominant topic, percentage contribution and keywords for each document
		for topic_prob in row:
			topics = topic_prob[0]
			probs = topic_prob[1]
			insert_topic_prop = mycursor.execute(" INSERT INTO comm_graph_topic_prob_table VALUES (%s, %s, %s) ", (i, topics, round(probs, 4)))
			mydb.commit()

		for j, (topic_num, topic_prob) in enumerate(row):

			word_prob = optimal_model.show_topic(topic_num, topn=15)
			#print('\n', topic_num)
			#print(word_perc)
			check_topic_existence = mycursor.execute("SELECT topic_number FROM comm_graph_topic_term_prob_table WHERE topic_number = '%s' " % topic_num)
			f = mycursor.fetchall()
			if not f:
				for terms in word_prob:
					word = terms[0]
					prob = terms[1]
					insert_into_table = mycursor.execute("INSERT INTO comm_graph_topic_term_prob_table VALUES (%s, %s, %s)", (str(topic_num), str(word), str(round(prob, 4))))
					mydb.commit()

			if j == 0:  #=> dominant topic

				#sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
				insert_into_table = mycursor.execute("INSERT INTO comm_graph_topic_model_table VALUES (%s, %s, %s)", (i, str(topic_num), str(round(topic_prob, 4))))
				mydb.commit()					

			else:
				break
	

	"""
	compute Model Perplexity and Coherence Score
	#These two helps to provide a convinient measure to judge 
	#how good the topic model is.
	The lower perplexity the better
	"""
	#lda_model = LdaModel(corpus=corpus, id2word=wordIDs, num_topics=no_of_k_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
	#lda_model = LdaModel(corpus=corpus_tfidf, id2word=wordIDs_tfidf, num_topics=no_of_k_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
	#print(lda_model.print_topics())

	#print('\nPerplexity Measure: ', lda_model.log_perplexity(corpus))
	#print('\nPerplexity Measure: ', lda_model.log_perplexity(corpus_tfidf))

	#To show the topics in the most optimized occurence value:
	
	
	#visualize the topics
	import pyLDAvis
	import pyLDAvis.gensim
	#pyLDAvis.enable_notebook()

	mallet_2_lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model)

	
	visualisation = pyLDAvis.gensim.prepare(mallet_2_lda_model, corpus, wordIDs)
	#visualisation = pyLDAvis.gensim.prepare(mallet_2_lda_model, corpus_tfidf, wordIDs_tfidf)
	pyLDAvis.save_html(visualisation, '../enron analysis/statistics/topic model stats/pyldavis_comm_graph_'+str(limit)+'.html')


	
	print('All done!')
	end_time = time.monotonic()
	print('Total Execution Time: {}'.format(timedelta(seconds=end_time - start_time)))

if __name__ == "__main__":
	main()