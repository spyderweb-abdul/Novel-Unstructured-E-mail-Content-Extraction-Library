###This code will extract files from all enron's folders in the dataset and create new files based on most common to, from and subject lists
from collections import Counter
from email.parser import Parser
import os
import json
import re

rootdir = "C:/python37/research_project/enron test/"

interest_folder = ['sent_items']   				

#All users folder list from enron email dataset
users_folder_list = ['allen-p', 'arnold-j', 'arora-h', 'badeer-r', 'bailey-s', 'bass-e', 'baughman-d', 'beck-s', 'benson-r', 'blair-l', 'brawner-s', 'buy-r', 'campbell-l', 
'carson-m', 'cash-m', 'causholli-m', 'corman-s', 'crandell-s', 'cuilla-m', 'dasovich-j', 'davis-d', 'dean-c', 'delainey-d', 'derrick-j', 'dickson-s', 'donoho-l', 'donohoe-t', 
'dorland-c', 'ermis-f', 'farmer-d', 'fischer-m', 'forney-j', 'fossum-d', 'gang-l', 'gay-r', 'geaccone-t', 'germany-c', 'gilbertsmith-d', 'giron-d', 'griffith-j', 'grigsby-m', 
'guzman-m', 'haedicke-m', 'hain-m', 'harris-s', 'hayslett-r', 'heard-m', 'hendrickson-s', 'hernandez-j', 'hodge-j', 'holst-k', 'horton-s', 'hyatt-k', 'hyvl-d', 'jones-t', 
'kaminski-v', 'kean-s', 'keavey-p', 'keiser-k', 'king-j', 'kitchen-l', 'kuykendall-t', 'lavorato-j', 'lay-k', 'lenhart-m', 'lewis-a', 'linder-e', 'lokay-m', 'lokey-t', 'love-p', 
'lucci-p', 'maggi-m', 'mann-k', 'martin-t', 'may-I', 'mccarty-d', 'mcconnell-m', 'mckay-b', 'mckay-j', 'mclaughlin-e', 'merriss-s', 'meyers-a', 'mims-thurston-p', 'motley-m', 'neal-s', 
'nemec-g', 'panus-s', 'parks-j', 'pereira-s', 'perlingiere-d', 'phanis-s', 'pimenov-v', 'platter-p', 'presto-k', 'quenet-j', 'quigley-d', 'rapp-b', 'reitmeyer-j', 'richey-c', 'ring-a', 
'ring-r', 'rodrique-r', 'rogers-b', 'ruscitti-k', 'sager-e', 'saibi-e', 'salisbury-h', 'sanchez-m', 'scholtes-d', 'schoolcraft-d', 'schwieger-j', 'scott-s', 'semperger-c', 'shackleton-s', 
'shankman-j', 'shapiro-r', 'shively-h', 'skilling-j', 'slinger-r', 'smith-m', 'snaders-r', 'solberg-g', 'south-s', 'staab-t', 'stclair-c', 'steffes-j', 'stepenovitch-j', 'stokley-c', 
'storey-g', 'sturm-f', 'swerzbin-m', 'symes-k', 'taylor-m', 'tholt-j', 'thomas-p', 'townsend-j', 'tycholiz-b', 'ward-k', 'watson-k', 'weldon-c', 'whalley-g', 'whalley-l', 'white-s', 
'whitt-m', 'william-w3', 'williams-j', 'wolfe-j', 'ybarbo-p', 'zipper-a', 'zufferli-j']

#create a function to clean data especially as observed in enron sent email contents
def clean_data(data):

	if data:
		data = data.replace('\n', '')
		data = data.replace('\t', '')
		data = data.replace('"', '')
		data = data.replace('\\','')
		data = data.replace(',', '')       #especially for commas seen in recipient names
		data = data.replace(' (PST)', '')  #seen in time format
		data = data.replace(' (PDT)', '')  #seen in time format
		data = data.replace(' -0800','')   #seen in time format
		data = data.replace(' -0700', '')  #seen in time format


		return data

#function to extract all emails appearing alongside recipients' name in enron sent email content
def re_email(data):

	common_exp = ['<?\S+@\S+>?', '@\s\w+', '</?\S+>']  #list the observed patterns
	for exp in common_exp:
		reg = re.findall(exp, data)
		for regs in reg:
			data = data.replace(regs, '')      #replace the extracted patterns

	return data

#nitialize source list array
source_list = []

for directory, subdirectory, filenames in os.walk(rootdir, topdown=True):  #first, walk through the directory
		for folder in subdirectory:
			#this will get the base name of the folders. (e.g: allen-p/, allen-p/inbox, allen-p/sent_items,... )
			f_path = os.path.basename(os.path.join(directory, folder))  

			#this will compare if the base name has (/sent_items as basis), 
			#as well as check if the bases is something like ('/enron email/allen-p/').
			#particularly, we want only directory structure like: '../allen-p/sent_items'
			if f_path in interest_folder and f_path not in users_folder_list:
				x = os.path.join(directory)

				#append sent_items directory with the root folder. So that we have '../enron email/allen-p\sent_items'
				interest_src = x + '\sent_items'

				#push to source list array
				source_list.append(interest_src)



#print(to_email_list)

# for each root source of the interest folder as listed
for src_ in source_list:
	#create empty list to populate all to emai list
	to_email_list = []    
	#We list the files in source directory
	for filename in os.listdir(src_):
		src_file = os.path.join(src_, filename)

		#open and read each file
		with open(src_file, 'r') as f:        			    
			data = f.read()

			#callthe email parser library
			email = Parser().parsestr(data)					

			#sender's email
			from_email = email['from']
			#sender's name
			from_name = clean_data(re_email(email['X-From']))						
			#email_body = email.get_payload()

			#the Cc will be useful incase the to_email is empty as observed in some cases
			cc = email['Cc']
			#the recipient's email
			to_email = clean_data(email['to'])	
			#populate the recipients'email list as you traverse the directory	
			if to_email:
				to_email_list.append(to_email)
			else:
				to_email_list.append(cc)


 	#convert to a set, so we can have only unique values
	email_list_set = set(to_email_list)	

	# create an empty initial dict (we want the sender to appear 
	init_dict = {}

	# the initial dict only once in the loop of json'd file)															
	init_dict['from'] = [from_email, from_name]
	print(init_dict) #just to see when operation starts for each sender



	json_file = "enron analysis/sent/"+from_name+"-sent.json"

	isFIleExist = os.path.exists(json_file)

	if not isFIleExist:

		
		#open a file for append or create new 
		with open(json_file, "a") as f:
			#dump dict as json 		
			json.dump(init_dict, f, ensure_ascii=False, indent=4)		   		

		#for each email in recipients' email list - loop
		for to1 in email_list_set:

			#empty subject list
			unique_subject_list = []

			#another empty dict for recipient details and mail subject
			cont_dict = {}

			#for each file in the interest folder
			for filename in os.listdir(src_):
				src_file = os.path.join(src_, filename)

				#open and read each file
				with open(src_file, 'r') as f:        			    
					txt = f.read()

					mail_cont = Parser().parsestr(txt)

					subjects = mail_cont['Subject']
					to_email2 = mail_cont['to']

					#Making sure subject is not empty or having a 'None' as value
					if subjects:
						subjects = clean_data(mail_cont['Subject'])

					#Making sure the to_email is not empty. If it is, use the CC
					if to_email2:
						to_email2 = to_email2
					else:
						to_email2 = mail_cont['Cc']

					date = clean_data(mail_cont['Date'])
					to_name = clean_data(re_email(mail_cont['X-to']))

					#Making sure that to_name is not empty. If it is, use the X-Cc
					if to_name:
						to_name = to_name
					else:
						to_name = clean_data(re_email(mail_cont['X-cc']))

					#if the new recipient email = the recipient email coming from the upper loop
					if to_email2 == to1:

						subj_date = subjects+" ("+date+")"		

						#append all subjects with dates to that recipient
						unique_subject_list.append(subj_date)					

						#pass the value to the dict
						cont_dict['to'] = [to_email2, to_name]
						cont_dict['subject'] = [a for a in unique_subject_list]
						
		
						#('\n'.join('%s , %s' % (x, y) for x, y in zip(unique_subject_list, date_list)))

						#(a, b) for a in unique_subject_list for b in date_list

			#make sure the currently looped dict is not empty before appending			
			if cont_dict != {}:	
				#open the file and append the dics as json																
				with open("enron analysis/sent/"+from_name+"-sent.json", "a") as f:
					#dump dict as json			
					json.dump(cont_dict, f, ensure_ascii=False, indent=4)
