f = file("train.dat")
cnt = 0
for l in f:
	if (cnt < 10):
		cnt += 1
	else:
		break
	lines = l.strip().split()[0:11]
	print lines
