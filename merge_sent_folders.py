import os
import shutil
import random

dir = "C:/python37/research_project/enron email/"

### We are only interested in the 'sent' and 'sent_items'. 
#We will be moving the content of sent folder into sent_items. 
#So that we can focus only on one folder for sent messages
###

interest_folders = ['sent', 'sent_items']   				

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

#Work with the directories
for root, dirs, files in os.walk(dir, topdown=True):
	for folder in dirs:
		#this will get the base name of the folders. (e.g: allen-p/, arnold-j/, allen-p/inbox, allen-p/sent,... )
		f_path = os.path.basename(os.path.join(root, folder))  

		#this will compare if the base name has (/sent or /sent_items as basis), 
		#as well as check if the bases is something like ('/enron email/allen-p/').
		#particularly, we want only directory structure like: '../allen-p/sent' and '../allen-p/sent_items'

		if f_path in interest_folders and f_path not in users_folder_list:
			x = os.path.join(root)

			#print(x)
			
			#then we add '\sent' to the pruned directory tree structure. That will be the source path
			src_path = x + '\sent'
			#print(src_path)

			#do smae for the destination path; '\sent_items'
			dest_path = x + '\sent_items'
			#print(dest_path)

			#check if there  exist the sent folder before process is executed
			isDirExist = os.path.exists(src_path)

			if isDirExist:

				#Here, we will rename all the files in the sent folder. 
				#The reason is that most filename in sent already exist in sent_items. We do not want them over-written
				c = 20000

				#We list the files in source directory
				for filename in os.listdir(src_path):
					src_file = os.path.join(src_path, filename)

					#check if files exist in the directory
					isFileExist = os.path.exists(src_file)

					if isFileExist:

						#get a new nae. Just a strigified integer
						new_file_name = str(c)

						#create a new file name and path
						new_src_file_name = os.path.join(src_path, new_file_name)

						#rename the initial filename to another one. Usually just an increamented number process
						os.rename(src_file, new_src_file_name)

						c += 1

						#We move each renamed files to the destination folder: the '\sent_items' folder 
						shutil.move(new_src_file_name, dest_path)
					
					