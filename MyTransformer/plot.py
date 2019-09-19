import matplotlib.pyplot as plt
import numpy as np

X = np.array([-0.4865, -0.7781, -1.1008, -1.3526, -1.6596])
y = np.array([-3.0969, -3.0969, -3.3979, -3.5229, -3.6990])

plt.xlim((-1.8,-0.4))
plt.ylim((-3.8,-2.9))
plt.plot(X,y)
plt.show()