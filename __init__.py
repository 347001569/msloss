import numpy as np
from loss import ms_loss
a=[[1,0,0,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]
b=[[0.12,0.32,0.12,0.11],[0.2,0.123,0.22,0.01],[0.11,0.1,0.2,0.3],[0.12,0.3,0.33,0.11]]
loss= ms_loss(a,b)
print(loss)






