import sys
f1 = open(sys.argv[1], 'r')
f2 = open('Q1.txt', 'w')
content = f1.readline().split()
candidate = []
count = []
for word in content:
    if word not in candidate:
    	candidate.append(word)
    	count.append(1)
    else:
    	target = candidate.index(word)
    	count[target] += 1
for num in range(len(candidate)-1):
	f2.write('%s %d %d\n'%(candidate[num], num, int(count[num])))
f2.write('%s %d %d'%(candidate[len(candidate)-1], len(candidate)-1, int(count[len(candidate)-1])))