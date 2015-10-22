from numpy import *
#import re
#import textwrap
from decimal import *

lst = [0.01, 0.05, 0.25, 1, 5, 25, 100]
cross = zeros((7, 4))
cross[:,0] = lst[:]
i=0
for s in lst:
	cross[i,1:4] = 1-loadtxt('cp_cross_lambda/cross_val_lambda_'+str(s)+'.txt')
	i += 1

print cross
savetxt('all_data.txt',cross, fmt ="%s")
