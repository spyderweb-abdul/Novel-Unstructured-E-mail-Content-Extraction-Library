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
from gensim.test.utils import common_texts
from gensim.models.nmf import Nmf
from gensim.test.utils import datapath
from smart_open import open
import math

import matplotlib.pyplot as plt
import numpy as np

import mysql.connector
import pickle
import random


mydb = mysql.connector.connect(host = "localhost", user = "root", password = "", database = "enron_project_db")
mycursor = mydb.cursor()

def main():

	def nmf_latent_topics(sender, recipient):

		#initialize document words contatiner
		word_bucket = []

		#get file_ids of communications between pair within a predefine time period 
		get_file_ids = mycursor.execute("SELECT file_id, receipt_category FROM graph_structure_alt_table WHERE sender = %s \
										 AND recipient = %s ", (sender, recipient))
		file_ids = mycursor.fetchall()

		#if the query returns not empty
		if file_ids:

			#count the number of emails exchanged
			file_id_len = len(file_ids)
			#compute a certain percentage
			email_samples = int((75/100)*file_id_len)
			#sample a certain percentage of the emails between pairs
			sample_files = random.sample(file_ids, k=email_samples)

			#for each of the files:
			for f in sample_files:

				file_id, receipt_category = f[0], f[1]

				#initialize counts for mail categories for each email
				to_count = 0
				cc_count = 0
				bcc_count = 0

				if receipt_category == 'TO':
					to_count += 1
				elif receipt_category == 'CC':
					cc_count += 1
				else:
					bcc_count += 1

				#for each email communication pair, we want to extract words in the email from our processed, stored data.
				fetch_terms = mycursor.execute("SELECT word FROM communication_bow_table WHERE file_id = %s ", (file_id,))
				term_result = mycursor.fetchall()

				#initialize word list
				word_list = []
				for w in term_result:
					word = w[0]
					
					#push words into word list
					word_list.append(word)
				
				#push word list into documents container
				word_bucket.append(word_list)
			
			#create a dictionary
			wordIDs = corpora.Dictionary(word_bucket)

			#term document frequency
			corpus = [wordIDs.doc2bow(word) for word in word_bucket]
			#print(corpus)

			#if the corpus is not empty
			if corpus:

				#load the trained nmf model and pass the new (unseen) document to it
				optimal_model = Nmf.load('nmf_main.model')
				
				#update the trained model
				optimal_model.update(corpus)

				for i, row in enumerate(optimal_model[corpus]):

					#we can sort to get topic prob. in decreasing order
					#row = sorted(row, key=lambda x: (x[1]), reverse=True)

					#initialize topic distribution list
					prob_list = []

					r"""We will create an arbitrary list of zeros depending on the 
					number of topics the nmf model returns. Usually, it should be 10 topics,
					but most times, it returns less than 10. So, we need to fill the topics
					that are not represented with zeros. 
					"""

					arbitrary_list = [(0, float(0))] * (10 - len(row))

					for i in row:

						t_ = i[0]  				#topic no.
						p_ = i[1]  				#prob. distributions

						#refill the arbitrary list with (topic, (topic, prob.distr.)
						arbitrary_list.insert(t_, (t_, p_))
					
					#unpack the arbirary list				
					for topic_prob in arbitrary_list:
						topics = topic_prob[0]
						probs = topic_prob[1]

						#append prob. distributions to the prob_list
						prob_list.append(round(probs, 4))

						
					r"""try to fill prob array with zeros in case the size is not up to
					the number of specified number of topics, especially in nmf case
					"""
					#def padarray(arr, size):
						#diff = size - len(arr)
						#return np.pad(arr, pad_width=(0, diff), mode='constant')		

					#prob_arr = padarray(prob_list, 10)
					#prob_arr = list(prob_arr)


					def normalize0_1(x):
						#res: https://stats.stackexchange.com/questions/380276/how-to-normalize-data-between-0-and-1
						#normalize some communication values to range btw 0 and 1
						norm_val = round(1/(1 + math.exp(-x)), 4)
						return norm_val

					#normalize between 0 and 1 all the values gotten from counts
					if to_count != 0:
						to_freq = float(to_count)
					else:
						to_freq = 0.0
					if cc_count != 0:
						cc_freq = float(cc_count)
					else:
						cc_freq = 0.0
					if bcc_count != 0:
						bcc_freq = float(bcc_count)
					else:
						bcc_freq = 0.0

					norm_freq = float(file_id_len)
					
					#append all the values to the distribution array
					prob_list.append(norm_freq)
					prob_list.append(to_freq)
					prob_list.append(cc_freq)
					prob_list.append(bcc_freq)

			#if corpus is empty:
			else:
				prob_list = None
				
			return prob_list

	with open('../../VGRNN/data/enron_new_data/graph_user_pair.pickle', 'rb') as handle:
		user_list = pickle.load(handle)

	user_list = user_list[7:34]

	users_sample_list = []
	for i in user_list:
		
		#get the length of each users list
		get_len = len(i)
		#compute a certain percentage of the numbers of values in the list
		sample_percentage = int((80/100)*get_len)

		#sample a certain percentage of the user list
		random_samples = random.sample(i, k=sample_percentage)

		#remove the samples from the original user list
		#for s in samples_to_remove:
			#remaining_samples = i.remove(s)

		#populate the initialised list		
		users_sample_list.append(random_samples)


	random_graph_list = []
	#from the sampled list
	for sampled in users_sample_list:		#[(1,0), (10,2), (45,34)]
		#unpack each tuples
		edge_attr_list = []
		for j in sampled:
			s, r = j[0], j[1]

			node_1 = mycursor.execute('SELECT account FROM account_id_alt_table WHERE node_id = "%s" ' % (s))
			sender = mycursor.fetchone()

			node_2 = mycursor.execute('SELECT account FROM account_id_alt_table WHERE node_id = "%s" ' % (r))
			recipient = mycursor.fetchone()

			edge_attr = nmf_latent_topics(sender[0], recipient[0])
			if edge_attr is not None:
				edge_attr_list.append((s, r, edge_attr))

		print('\n', edge_attr_list)
		random_graph_list.append(edge_attr_list)


	with open('../../VGRNN/data/enron_data_random/random_graph_list_80_user_75_email.pickle', 'wb') as f:
		pickle.dump(random_graph_list, f)


if __name__ == "__main__":
	main()
