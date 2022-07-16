import os
import pandas as pd
import json
import sys
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
import email.utils
import time
import datetime
from datetime import timedelta
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import email
from email.parser import Parser
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from nltk.tag import pos_tag
from nltk.util import ngrams
import mysql.connector
import math
import networkx as nx
import matplotlib.pyplot as plt

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def main():


	start_time = time.monotonic()

	mydb = mysql.connector.connect(host = "localhost", user = "root", password = "", database = "enron_project_db_v2")

	mycursor = mydb.cursor()

	#print(os.getcwd())
	stemmer, word_lemma = PorterStemmer(), WordNetLemmatizer()

	dir_root = "../enron test/"
	#dir_root = "C:/python37/research_project/enron test/"
	#dir_root = "C:/python37/research_project/Enron Main/"


	stop_word_file = "../enron analysis/stop_words_0.txt"

	#create a function to clean data especially as observed in enron sent email contents
	def clean_data(data):

		if data:
			data = data.replace('\n', '')
			data = data.replace('\t', '')
			data = data.replace('"', '')
			data = data.replace('\\','')
			#data = data.replace(',', '')       #especially for commas seen in recipient names
			data = data.replace('(','')
			data = data.replace(')','')
			return data 

	#function to extract all emails appearing alongside recipients' name in enron sent email content
	def re_email(data):

		common_exp = ['<?\S+@\S+>?', '@\s\w+', '</?\S+>']  #list the observed patterns
		for exp in common_exp:
			reg = re.findall(exp, data)
			for regs in reg:
				data = data.replace(regs, '')      #replace the extracted patterns

		return data

	#function to process the body text
	def  body_preprocess(data):

		#data = data.replace('\t', '')
		data = data.replace('*', '')
		data = data.replace('_','')

		#match characters in tags
		#match any unicode digit or any character which is not a unicode word character.
		#match and delete any url string - ((?:https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s\`\!\(\)\[\]\{\}\;\:\'\"\.\,\<\>\?\«\»\“\”\‘\’]))
		#match email address - a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
		#CSS Codes - (([a-zA-Z]+\-[a-zA-Z]+)|([a-zA-Z]+))\:[/\s/\S](\#|[/\s/\S])[a-zA-Z0-9.]+;
		#match and delete anything after ----Original message and some other observed text
		# (which signifies that the message is replied to or forwarded) 
		#Text after that could be duplicates of existing messages
		sub_exp = [ '-*Original[/\s/\S]*', 'Sincerely[/\s/\S]*', 'Sincerely,[/\s/\S]*', 'Regards,[/\s/\S]*', 'Cordially,[/\s/\S]*', 'Best Regards,[/\s/\S]*', 'Thanks,[/\s/\S]*', 'Kind regard,[/\s/\S]*', \
		'This e-mail is the property of Enron[/\s/\S]*', '(/\d|/\W)+', '<!--?.*-->', '<\s*[^>]*>[/\s/\S]*', '-*Returned[/\s/\S]*', '-*Forwarded[/\s/\S]*', \
		'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '((?:https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,12}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s\`\!\(\)\[\]\{\}\;\:\'\"\.\,\<\>\?\«\»\“\”\‘\’]))', \
		'(([a-zA-Z]+\-[a-zA-Z]+)|([a-zA-Z]+))\:[/\s/\S](\#|.)[a-zA-Z0-9.]+\;', 'Start Date\:[/\s/\S]*', 'LOG MESSAGES\:[/\s/\S]*', 'Delivery Date\:[/\s/\S]*', 'LOG MESSAGES\:[/\s/\S]*', \
		'PARSING FILE[/\s/\S]*', '-*Energy[/\s/\S]*', 'COB NS[/\s/\S]*', 'EOL D[/\s/\S]*', '[=]+[/\s/\S]*', '[-]+[/\s/\S]*' ]
		

		for exp in sub_exp:                
			reg = re.findall(exp, data)
			#datas = ''
			for regs in reg:
				data = data.replace(str(regs), '')

		return data

	#terms count
	def word_count(data):
		count = len(re.findall(r'\w+', data))
		return count

	#date format converter
	def date_converter(data):
		data = email.utils.parsedate(data)
		#(2009, 11, 5, 07, 08, 06, 0, 1, -1)
		data = time.mktime(data)
		#1258378322.0
		#data = datetime.datetime.fromtimestamp(data)
		#datetime.datetime(2009, 11, 16, 13, 32, 2)
		return data

	#list of stopwords 
	def stop_words(file_path):
		
		with open(file_path, 'r', encoding='utf-8') as f:
			stopwords = f.readlines()
			stop_set = set(m.strip() for m in stopwords)
			
			return frozenset(stop_set)

	#stem function
	def stemmed_data(filtered, stemmer):
		stemmed = []
		for item in filtered:
			stemmed.append(stemmer.stem(item))
		return stemmed
	#message id generator for each email
	def message_identifier(sender, date, receiver=[]):

		#empty list to populate receivers' array
		rec_list = []
		#split the list
		#receiver_list = receiver.split(',')
		for r in receiver:
			r.split(',')
			r = r.replace(' \n\t','') #replace observed withspaces
			r = r.replace(' ','')	  #replace observed space
			r = r.replace('<','')
			r = r.replace('>','') 
			rec_list.append(r)		  #append to new list

		#Another empty list to populate the message ids	
		msg_id_list = []
		for mail in rec_list:
			msg_id = str(sender)+':'+str(mail)+':'+str(date)  #format:abc@enron.com:2001-12-27 15:37:31
			msg_id_list.append(msg_id)						  #append the message id to list
		#msg_id = str(sender)+':'+str(date)  #format:abc@enron.com:123@enron.com:2001-12-27 15:37:31

		return msg_id_list


	#message parse function	
	def parse_process(file_):
		#with open(file_, 'r', encoding='utf-8') as f:
		with open(file_, 'r', encoding='latin-1') as f:
			text = f.read()
			mail_parser = Parser().parsestr(text)

			#extract the message id
			#message_id = check_pattern(mail_parser['Message-ID'])

			#extract the subject of the email
			subjects = mail_parser['Subject']
			#if subject is not empty
			if subjects:
				subjects = clean_data(subjects)

			#check if message is replied, forwarded or the main initiation by detecting the first 3 xters
			subject_char = subjects[:3]
			if subject_char == 'RE:':
				mail_cat = 'RE'
			elif subject_char == 'Re:':
				mail_cat = 'RE'
			elif subject_char == 're:':
				mail_cat = 'RE'
			elif subject_char == 'FW:':
				mail_cat = 'FWD'
			elif subject_char == 'Fw:':
				mail_cat = 'FWD'
			elif subject_char == 'Fwd:':
				mail_cat = 'FWD'
			elif subject_char == 'fw:':
				mail_cat = 'FWD'
			elif subject_char == 'fwd:':
				mail_cat = 'FWD'		
			else:
				mail_cat = 'INIT'

			#extract cc, bcc an to mails
			cc, bcc, to_mail = mail_parser['Cc'], mail_parser['Bcc'], mail_parser['To']

			to_list = []
			if to_mail:
				to_rec_list = to_mail.split(',')
				for r in to_rec_list:
					r = r.replace(' \n\t','') #replace observed withspaces
					r = r.replace(' ','')	  #replace observed space
					r = r.replace('<','')
					r = r.replace('>','') 
					to_list.append(r)		
						
			cc_list = []
			if cc:
				receiver_list = cc.split(',')
				for r in receiver_list:
					r = r.replace(' \n\t','') #replace observed withspaces
					r = r.replace(' ','')	  #replace observed space
					r = r.replace('<','')
					r = r.replace('>','') 
					cc_list.append(r)		
				
			bcc_list = []
			if bcc:				
				bcc_rec_list = bcc.split(',')
				for r in bcc_rec_list:
					r = r.replace(' \n\t','') #replace observed withspaces
					r = r.replace(' ','')	  #replace observed space
					r = r.replace('<','')
					r = r.replace('>','') 
					bcc_list.append(r)	

			#extract the sender's email
			from_mail = mail_parser['From']
			#Just to handle exceptions
			if from_mail:
				from_mail = from_mail
			else:
				from_mail  = None
			
			#Extract the name of the recipient
			to_name = clean_data(re_email(mail_parser['X-to']))
			#check if name is not empty
			if to_name:
				to_name = to_name
			#else, use the value in the X-cc
			else:
				to_name = clean_data(re_email(mail_parser['X-cc']))

			#extract the sender's name
			from_name = clean_data(re_email(mail_parser['X-From']))
			#extract the date and format
			date = date_converter(mail_parser['Date'])

			main_body = mail_parser.get_payload()

			if main_body:
				main_body = main_body
			else:
				main_body = None

			#docs = subjects + ' ' + main_body

			return subjects, cc_list, bcc_list, to_list, to_name, from_mail, from_name, date, main_body, mail_cat


	def lemma_data(words, lemma):
		lemmas = []

		pos_tags = pos_tag(words)	
		
		for item, tag in pos_tags:
			if tag == "NN" or tag == "NNS":
				yield lemma.lemmatize(item, pos='n')
			elif tag.startswith('VB'):
				yield lemma.lemmatize(item, pos='v')
			elif tag.startswith('JJ'):
				yield lemma.lemmatize(item, pos='a')
			elif tag.startswith('RB'):
				yield lemma.lemmatize(item, pos='r')
			else:
				yield item

			#lemmas.append(lemma.lemmatize(item))
		#return pos_tags



	def content_processor(data):
	    
	    #preprocess with set of predefined methods
		words = body_preprocess(data)

		words = word_tokenize(words)
	
		#convert all to lower case
		words = [w.lower() for w in words]

		#remove remaining strings that are not alphabetic
		words = [w for w in words if w.isalpha()]

		#remove punctuations
		table = str.maketrans('', '', string.punctuation)
		words = [w.translate(table) for w in words]

		#bigram = Phrases(words, min_count = 5, threshold = 100)
		#trigram = Phrases(bigram[words], threshold = 100)

		#bigram_mod = Phraser(bigram)
		#trigram_mod = Phraser(trigram)

		#words = bigram_mod[words]
		#words = [trigram_mod[bigram_mod[doc]] for doc in words]
		
		words = lemma_data(words, word_lemma)

		#filter out additional stop words 
		#words = stopwords.words('english')
		stop_word = stop_words(stop_word_file)
		words = [w for w in words if not w in stop_word]


		return words

	def tf_calculator(documents, subjects):
		#initialize unique word frequency count dict
		wf_counter_body = {}
		#initialize all word in the document count variable 
		word_count_body = 0
		#word_count_total = 0

		for d in documents:
			#tokenize the document
			token = word_tokenize(d)
			for w in token:
				if w in wf_counter_body:
					#increment word frequency counter if it exist
					wf_counter_body[w] = wf_counter_body[w] + 1
				else:
					#otherwise, start a new one
					wf_counter_body[w] = 1
				#increment word_count
				word_count_body += 1	
		#word_count_total += word_count_body

	 	#word frequency for subjects
		wf_counter_subj = {}
		word_count_subj = 0
		#subj_count_total = 0

		for s in subjects:
	 		sub_token = word_tokenize(s)
	 		for t in sub_token:
	 			if t in wf_counter_subj:
	 				wf_counter_subj[t] = wf_counter_subj[t] + 1
	 			else:
	 				wf_counter_subj[t] = 1
	 			word_count_subj += 1
		#subj_count_total += word_count_subj

		#initialize term frequency dict for body
		tf_body = {}
		#loop through wf_counter
		for k, v in wf_counter_body.items():
			#calculate tf by ratio of word freq(wf) by total word count
			tf_body[k] = round((v)/(word_count_body), 4)
		
		#initialize term frequency dict for subject
		tf_subj = {}
		#loop through wf_counter
		for k, v in wf_counter_subj.items():
			#calculate tf by ratio of word freq(wf) by total word count
			tf_subj[k] = round((v)/(word_count_subj), 4)
		
		return wf_counter_body, wf_counter_subj, tf_body, tf_subj


	op_counter = 1
	for dir_, subdir_, file_ in os.walk(dir_root):
		for filename in file_:
			file_path = os.path.join(dir_, filename)

			print(file_path)

			#We have to replace the forward slashes in the file_path, 
			#otherwise retrieving from database might be difficult
			def replace_link(file_path):
				file_path = file_path.replace('/', '//')
				file_path = file_path.replace('\\', '//')
				return file_path

			file_link = replace_link(file_path)					

			try:				

				subjects, cc, bcc, to_mail, to_name, from_mail, from_name, date, main_body, mail_cat = parse_process(file_path)

					#for each email, join the recipient list
				full_to_list = cc + bcc + to_mail
							
				processed_doc = content_processor(main_body)
				processed_subjects = content_processor(subjects)

				if processed_doc:
					print(processed_doc)
				else:
					print('\nMail Not Processed')
		
			#except Exception as e:	
			except:
				#print(e)
				print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
				#pass	

	print('All done!')
	end_time = time.monotonic()
	print('Total Execution Time: {}'.format(timedelta(seconds=end_time - start_time)))

if __name__ == "__main__":
	main()