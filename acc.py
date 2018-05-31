import os
import time

# import re
# import matplotlib.pyplot as plt
# import numpy as np

for i in range(55, 60):
	print (i)
	start = time.time()
	os.system(f'python parse.py outm21/parser{i+1:03} brat_dev.conllx bratdevout.conllx')
	end = time.time()
	print(f'time: {end-start:,.2f} secs')
	os.system(f'python conllx2brat.py bionlp09_shared_task_development_data_rev1/ --conllx bratdevout.conllx')
	end = time.time()
	print(f'time: {end-start:,.2f} secs')
	os.system("./a2-evaluate.pl -g gold-dev/ -s -p bionlp09_shared_task_development_data_rev1/*.t1")
	end = time.time()
	print(f'time: {end-start:,.2f} secs')

print ("=================================================")

for i in range(55, 60):
	print (i)
	start = time.time()
	os.system(f'python parse.py outm22/parser{i+1:03} brat_dev.conllx bratdevout.conllx')
	end = time.time()
	print(f'time: {end-start:,.2f} secs')
	os.system(f'python conllx2brat.py bionlp09_shared_task_development_data_rev1/ --conllx bratdevout.conllx')
	end = time.time()
	print(f'time: {end-start:,.2f} secs')
	os.system("./a2-evaluate.pl -g gold-dev/ -s -p bionlp09_shared_task_development_data_rev1/*.t1")
	end = time.time()
	print(f'time: {end-start:,.2f} secs')

print ("=================================================")

for i in range(55, 60):
	print (i)
	start = time.time()
	os.system(f'python parse.py outm23/parser{i+1:03} brat_dev.conllx bratdevout.conllx')
	end = time.time()
	print(f'time: {end-start:,.2f} secs')
	os.system(f'python conllx2brat.py bionlp09_shared_task_development_data_rev1/ --conllx bratdevout.conllx')
	end = time.time()
	print(f'time: {end-start:,.2f} secs')
	os.system("./a2-evaluate.pl -g gold-dev/ -s -p bionlp09_shared_task_development_data_rev1/*.t1")
	end = time.time()
	print(f'time: {end-start:,.2f} secs')