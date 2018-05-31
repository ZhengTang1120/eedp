import os

for i in range(30, 60):
	os.system(f'python parse.py --sentence outm21/parser{i+1:03} genia/test.conllx geniatestout2.conllx')
	os.system("./eval.pl -g genia/test.conllx -s geniatestout2.conllx  -q")

# lm = 0
# um = 0
# for line in open("genia/r.txt"):
# 	if "Labeled" in line:
# 		l = float(line.split()[-2])
# 		if lm < l:
# 			lm = l
# 	if "Unlabeled" in line:
# 		u = float(line.split()[-2])
# 		if um < u:
# 			um = u

# print (lm)