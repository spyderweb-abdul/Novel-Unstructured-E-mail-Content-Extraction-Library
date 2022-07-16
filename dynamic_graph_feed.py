import mysql.connector
import math
import networkx as nx


import time
import datetime
from datetime import timedelta

import numpy as np
from numpy import asarray
from numpy import savez, save, savez_compressed, load
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
#import scipy as sp

import scipy.spatial as sp
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import pickle
import pandas as pd

import dg_process_functions as dg
import project_utils as utils
from sparse import as_coo, stack

mydb = mysql.connector.connect(host = "localhost", user = "root", password = "", database = "enron_project_db")
mycursor = mydb.cursor()

def main():

	G = nx.Graph()
	
	start_time = time.monotonic()

	print('Processing...')


	#extract distinct time period from the database
	#from_graph = mycursor.execute("SELECT DISTINCT(msg_time) FROM graph_structure_table WHERE sender LIKE %s AND recipient LIKE %s ORDER BY msg_time ASC" , ('%'+'@enron.com'+'%', '%'+'@enron.com'+'%'))
	from_graph = mycursor.execute("SELECT DISTINCT(msg_time) FROM graph_structure_alt_table ORDER BY msg_time ASC")
	graph_result = mycursor.fetchall()

	#print(graph_result)

	#convert time array to a numpy array
	time_array = np.array(graph_result, dtype=int)
	#print(time_array.shape) #(132301, 1)

	#get the first and last  element of the array. The beginning and end of the time period
	start_date = datetime.datetime.fromtimestamp(time_array[0])
	end_date = datetime.datetime.fromtimestamp(time_array[-1])

	#slice time periods into weeks, in this case 1 week
	period_series = pd.date_range(start=start_date, end=end_date, freq='M') #for weekly communications

	#print(period_series.shape)  #(1175,)

	count = 0

	adj_attribute_list, adj_time_list, node_attribute_list = [], [], []
	adj_dense_list = []

	all_graph_list = []

	#for i in range(int(10)):  #range is 271, we do not want to consider the 271st array
	for i in range(int(period_series.shape[0] - 1)):  #range is 271, we do not want to consider the 271st array
		#we pick only 1 month range
		x = period_series[i :  i + 2] #result format: DatetimeIndex(['1979-12-31 16:00:00', '1980-01-31 16:00:00'], dtype='datetime64[ns]', freq='M')
		
	
		init_period = x[0].timestamp() #convert to timestamp
		last_period = x[1].timestamp() #convert to timestamp

		
		#initialize empty list
		graph_time_list = []

		#extract records of communications between the time periods cond=sidering only enron email account holders
		#extract_comm = mycursor.execute("SELECT DISTINCT sender, recipient FROM graph_structure_table WHERE msg_time BETWEEN %s AND %s \
										 #AND sender LIKE %s AND recipient LIKE %s ", (int(init_period), int(last_period), '%'+'@enron.com'+'%', '%'+'@enron.com'+'%'))
		extract_comm = mycursor.execute("SELECT DISTINCT sender, recipient FROM graph_structure_alt_table WHERE msg_time BETWEEN %s AND %s ", \
										 (int(init_period), int(last_period)))

		time_period_result = mycursor.fetchall()

		#we will process only months in which there were communications
		if time_period_result:

			print('\nTimestamp: {} - {}'.format(datetime.datetime.fromtimestamp(init_period), datetime.datetime.fromtimestamp(last_period)))
			
			#for each time step, do
			for t in time_period_result:

				sender, recipient = t[0], t[1]

				if sender and recipient:

					#get the latent topic distribution of each of the  emails
					prob_list = dg.nmf_latent_topics(sender, recipient, init_period, last_period)

					#get the id representation of the sender and the receiver
					sender_id, recipient_id = dg.get_node_ids(sender, recipient)
					
					graph_time_list.append((sender_id, recipient_id, prob_list))	#for to`pic distributions, comm. freq, as egde attr.
					#graph_list.append((sender_id, recipient_id, 1))	#for normal adjacency dense weighted list
		
		if graph_time_list:

			print(graph_time_list)
			all_graph_list.append(graph_time_list)
			
			#adj_coo_dense_matrix, adj_sparse_matrix, adj_dense_matrix, node_attr_matrix = utils.get_adj_matrix(graph_list)

			#adj_attribute_list.append(adj_coo_dense_matrix)
			#adj_time_list.append(adj_sparse_matrix)
			#adj_dense_list.append(adj_dense_matrix)
			#node_attribute_list.append(node_attr_matrix)


			count += 1

	print(count)

	with open('../../VGRNN/data/enron_new_data/graph_list2.pickle', 'wb') as f:
		pickle.dump(all_graph_list, f)
	
	'''
	#3D - NXNXN adjacency matrix list with edge attributes as channels
	with open('../../VGRNN/data/enron_new_data/enron_edge_attribute_matrix.pickle', 'wb') as f:
		pickle.dump(adj_attribute_list, f)
	
	#csr_sparse matrix of the different snapshots adj. matrix
	with open('../../VGRNN/data/enron_new_data/enron_adj_sparse_matrix.pickle', 'wb') as f:
		pickle.dump(adj_time_list, f)
	
	#main dense adjacencny matrix at different snapshots
	with open('../../VGRNN/data/enron_new_data/enron_adj_dense_matrix.pickle', 'wb') as f:
		pickle.dump(adj_dense_list, f)
	
	#node attribute matrix
	with open('../../VGRNN/data/enron_data/enron_node_attribute_matrix.pickle', 'wb') as f:
		pickle.dump(node_attribute_list, f)	

	'''
	print('All done!')
	end_time = time.monotonic()
	print('Total Execution Time: {}'.format(timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
	main()
