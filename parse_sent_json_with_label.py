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
			from_email = email['From']
			#sender's name
			from_name = clean_data(re_email(email['X-From']))						
			#email_body = email.get_payload()

			#the Cc will be useful incase the to_email is empty as observed in some cases
			cc = email['Cc']
			#the recipient's email
			to_email = clean_data(email['To'])	
			#populate the recipients'email list as you traverse the directory	
			if to_email:
				to_email_list.append(to_email)
			else:
				to_email_list.append(cc)


 	#convert to a set, so we can have only unique values
	email_list_set = set(to_email_list)	

	json_file = "enron analysis/sent_json_label/"+from_name+"-sent.json"

	#We want to split the names into comma separated entities. 
	#The idea is to extract and concatenate onlt the 1st xters in the names as labels
	from_splitter = from_name.split(' ')
	#print(from_splitter)

	concat_char = ''
	for ele_ in from_splitter:
		first_char_ = ele_[:1]
		concat_char += first_char_

		#print(concat_char)


	isFIleExist = os.path.exists(json_file)
	#check if file exist, so we would not have to process again
	if not isFIleExist:

		print(json_file)

		json_content = '{"from": {"email": ' + '"'+from_email+'", "name": "'+from_name+'", "label": "'+concat_char+'"}, "to": ['


		#open a file for append or create new 
		with open(json_file, "w") as f:
			f.write(json_content)	   		

		#for each email in recipients' email list - loop
		for to1 in email_list_set:

			#empty subject list
			unique_subject_list = []

			#for each file in the interest folder
			for filename in os.listdir(src_):
				src_file = os.path.join(src_, filename)

				#open and read each file
				with open(src_file, 'r') as f:        			    
					txt = f.read()

					mail_cont = Parser().parsestr(txt)

					subjects = mail_cont['Subject']
					to_email2 = mail_cont['To']

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

						#We check again if to_name is not 'None'
						if to_name:

							#We want to split the names into comma separated entities. 
							#The idea is to extract and concatenate onlt the 1st xters in the names as labels
							to_name_splitter = to_name.split(' ')

							#check for unwanted xters at the beginning of name str 
							extra_char = ["'", "-", "_", "\\", "/", "<", ">", "@"]
							concat_f_char = ''
							for elem in to_name_splitter:
								#first char in a str
								f_char = elem[:1]
								#second char in a str
								s_char = elem[1:2]
								if f_char != '':
									if f_char not in extra_char:
										concat_f_char += f_char
									else:
										concat_f_char += s_char
						#otherwise use the first 2 xters of to_email as label
						else:
							concat_f_char = to_email2[0:2]
						
						#append the other json objects 
						json_content2 = '{"email": "'+str(to_email2)+'", "name": "'+str(to_name)+'", "label": "'+str(concat_f_char)+'", "subject": [' + ', '.join([ '"'+ str(a) + '"' for a in unique_subject_list ]) +']}, '

			#make sure the currently looped dict is not empty before appending			
			if json_content2 != '':	
				#open the file and append the dics as json																
				with open("enron analysis/sent_json_label/"+from_name+"-sent.json", "a") as f:
					#dump dict as json			
					#json.dump(cont_dict, f, ensure_ascii=False, indent=4)
					f.write(json_content2)	   		
	
	#we want to remove the comma that separates the json object loop
	with open("enron analysis/sent_json_label/"+from_name+"-sent.json", "r+") as f:
		f.seek(0, os.SEEK_END)			#or f.seek(0, 2) to sek the end of file
		f.seek(f.tell() -2, 0)          #to seek second to the last character
		f.truncate()					#remove the , (which is the second to the last character in the file)
					
	#Then append the closing tag to the json file
	close_tag = ']}'
	with open("enron analysis/sent_json_label/"+from_name+"-sent.json", "a") as f:
		f.write(close_tag)
		f.close()