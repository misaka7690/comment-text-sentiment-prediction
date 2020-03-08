'''
@Author: misaka7690
@Description: give prediction considering both body and headline
'''

import numpy as np

headline_pred = np.recfromtxt('headline_pred')
body_pred = np.recfromtxt('body_pred')

# headline 占比 0.2,body_pred 占比 0.8
pred = headline_pred*0.2 + body_pred*0.8
pred = pred * 100
np.savetxt(r'pred',pred,fmt='%d')
