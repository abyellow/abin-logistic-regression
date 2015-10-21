from numpy import *
#import re
#import textwrap
from decimal import *

lst = [0.01, 0.05, 0.25, 1, 5, 25, 100]
cross = zeros((7, 4))
cross[:,0] = lst[:]
i=0
for s in lst:
	cross[i,1:4] = 1-loadtxt('cross_val_lambda_'+str(s)+'.txt')
	i += 1
#print float(cross[1,1])
print cross

#ut0 = re.compile(r'(\d)0+$')

#thelist = textwrap.dedent(
 #       '\n'.join(ut0.sub(r'\1', "%20f" % x) for x in cross)).splitlines()


savetxt('all_data.txt',cross, fmt ="%s")
