import mysql.connector
import math
import networkx as nx
import matplotlib.pyplot as plt

import time
import datetime
from datetime import timedelta
from pprint import pprint
import string
import scipy as sp
import numpy as np
import pickle
import pandas as pd

mydb = mysql.connector.connect(host = "localhost", user = "root", password = "", database = "enron_project_db_test")
mycursor = mydb.cursor()

def main():

	G = nx.Graph()

	graph_list = []
	get_senders = mycursor.execute(" SELECT connection_id, sender, recipient, frequency FROM communication_frequency_table")
	senders_result = mycursor.fetchall()

	print('Processing...')

	for s in senders_result:
		connection_id, sender, recipient, frequency = s[0], s[1], s[2], s[3]

		get_sender_id = mycursor.execute(' SELECT node_id FROM account_id_table WHERE account = %s ', (sender,))
		sender_id = mycursor.fetchone()

		get_recipient_id = mycursor.execute(' SELECT node_id FROM account_id_table WHERE account = %s ', (recipient,))
		recipient_id = mycursor.fetchone()

		#extract topic probability distribution based on communication between pairs
		#get_topic_prob = mycursor.execute(" SELECT prob_topic_number, prob FROM comm_graph_topic_prob_table WHERE connection_id = %s" % (connection_id))
		get_topic_prob = mycursor.execute(" SELECT prob_topic_number, prob FROM comm_graph_nmf_topic_prob_table WHERE connection_id = %s" % (connection_id))

		topic_prob_result = mycursor.fetchall()


		prob_list = []
		#prob_dict = {}
		for tp in topic_prob_result:
			topic_no, prob = tp[0], float(tp[1])

			prob_list.append(prob)
			#prob_dict.update({str(topic_no) : prob})

		
		#try to fill prob array with zeros in case the size is not up to
		#the number of specified number of topics, especially in nmf case
		def padarray(arr, size):
			diff = size - len(arr)
			return np.pad(arr, pad_width=(0, diff), mode='constant')
		

		prob_arr = padarray(prob_list, 10)
		prob_arr = list(prob_arr)

		norm_freq = round(1/(1 + math.exp(-frequency)), 4)

		prob_arr.append(norm_freq)

		print(prob_arr)		

		graph_list.append((sender, recipient, prob_arr)) #-for probabilistic distribution
		#graph_list.append((sender_id[0], recipient_id[0], prob_arr)) #-for probabilistic distribution
		#graph_list.append((sender_id[0], recipient_id[0], frequency))  #- for communication frequency as weight


	#print(len(graph_list))
	
	#graph_lists = [('dbartley@o2wireless.com', 'jeff.king@enron.com', [0.1462, 0.1462, 0.2008, 0.2144, 0.1462, 0.1462]),
	 #('dbartley@o2wireless.com', 'e-mail.allison@enron.com', [0.1462, 0.1462, 0.2105, 0.2047, 0.1462, 0.1462])]
	

	#with open('graph_list_test.data', 'wb') as f:
		#pickle.dump(graph_list, f)



	'''	
	
	with open('graph_list_test.data', 'rb') as f:
		graph_list = pickle.load(f)

	#print(graph_list)
	
	G.add_weighted_edges_from(graph_list)
	#print(G.edges(data=True))


	#print(G['dbartley@o2wireless.com']['jeff.king@enron.com']['weight'])
	#A = nx.to_pandas_adjacency(G, dtype=object)
	#print(A)

	#list(G.adjacency_iter())


	#plt.figure(figsize = (9, 9))
	#pos = nx.random_layout(G)
	#nx.draw_networkx(G, arrows=True, with_labels=False)
	#plt.show()

	#nx.write_gexf(G, 'graphviz_main.gexf')


	#print('Total No. of Nodes: ', int(G.number_of_nodes()))
	#print('Total No. of Edges: ', int(G.number_of_edges()))
	#print('Graph Density: ', nx.density(G))
	print('Graph Info: ', nx.info(G, n=None))
	#print('\nList of all_nodes: ', list(G.nodes()))
	#print('\nList of all edges: ', list(G.edges(data=True)))
	#print('\nDegree of all Nodes: ', dict(G.degree()))
	#print('Total No. of self-loops: ', int(G.number_of_selfloops()))
	#print('\nList of all nodes with self-loops: ', list(G.nodes_with_selfloops()))
	#print('List of all nodes we can go to in a single step from node "mary.cook@enron.com": ', list(G.neighbors('mary.cook@enron.com')))
	
	'''
	
if __name__ == "__main__":
	main()

