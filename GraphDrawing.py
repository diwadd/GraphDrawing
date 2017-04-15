import random

class GraphDrawing:
    def plot(self, NV, edges):
        ret = []
        random.seed(123)
        for i in range(2 * NV):
            ret.append(random.randrange(701))
        return ret

# -------8<------- end of solution submitted to the website -------8<-------

import sys
N = int(raw_input())
E = int(raw_input())
edges = []
for i in range(E):
    edges.append(int(raw_input()))

gd = GraphDrawing()
ret = gd.plot(N, edges)
print len(ret)
for num in ret:
    print num
    sys.stdout.flush()
