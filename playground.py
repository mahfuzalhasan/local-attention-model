import numpy as np

hist = [[223, 45, 90], 
        [3, 98, 0], 
        [1, 0, 29]]

hist = np.asarray(hist)

sum_1 = hist.sum(1)
sum_0 = hist.sum(0)


iou = np.diag(hist)/(sum_1 + sum_0 - np.diag(hist))

print(sum_0, sum_1)
print(iou)