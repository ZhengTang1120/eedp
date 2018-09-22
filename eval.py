import sys
import os
import time

# import re
# import matplotlib.pyplot as plt
# import numpy as np
import subprocess
import shlex
from multiprocessing import Pool

def evaluate(i):
	
	os.system(f'python parse.py out13_2/parser{i:03} brat_dev13.conllx bratdevout13.conllx')
	os.system(f'python conllx2brat.py BioNLP-ST-2013_GE_devel_data_rev3/ --conllx bratdevout13.conllx')
	os.system(f'./a2-evaluate.pl -g BioNLP-ST-2013_GE_devel_data_rev3 -s -p BioNLP-ST-2013_GE_devel_data_rev3_out/*.a2')
	# output = subprocess.run(f'./a2-evaluate.pl -g BioNLP-ST-2013_GE_devel_data_rev3/ -s -p 2013_GE_devel_data_rev3_out/*.a2', stdout=subprocess.PIPE, shell = True).stdout.decode("utf-8")

	# with open(f'results_{i:03}.txt', 'w') as f:
	# 	print(output, file=f)




if __name__ == '__main__':
	# n_procs = int(sys.argv[1])
	# start = int(sys.argv[2])
	# end = int(sys.argv[3]) + 1
	# starttime = time.time()
	# with Pool(n_procs) as pool:
	# 	pool.map(evaluate, range(start, end))
	# endtime = time.time()
	# print(f'time: {endtime-starttime:,.2f} secs')
	for i in range(1,50):
		evaluate(i)
