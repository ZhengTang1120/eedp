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
	
	os.system(f'python parse.py outmp21/parser{i:03} brat_dev.conllx bratdevout{i:03}.conllx')
	os.system(f'python conllx2brat.py bionlp09_shared_task_development_data_rev1/ --conllx bratdevout{i:03}.conllx --outfolder {i:03}')
	output = subprocess.run(f'./a2-evaluate.pl -g gold-dev/ -s -p {i:03}/*.t1', stdout=subprocess.PIPE, shell = True).stdout.decode("utf-8")

	with open(f'results_{i:03}.txt', 'w') as f:
		print(output, file=f)


if __name__ == '__main__':
	n_procs = int(sys.argv[1])
	start = int(sys.argv[2])
	end = int(sys.argv[3]) + 1
	starttime = time.time()
	with Pool(n_procs) as pool:
		pool.map(evaluate, range(start, end))
	endtime = time.time()
	print(f'time: {endtime-starttime:,.2f} secs')