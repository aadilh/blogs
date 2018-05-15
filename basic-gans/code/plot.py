import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from training_data import *
sb.set()


# data = sample_data(n=256)
#
# plt.scatter(data[:,0], data[:,1])

data = pd.read_csv('loss_logs.csv')

plt.plot('Iteration','Discriminator Loss',data=data)
plt.plot('Iteration','Generator Loss',data=data)

plt.legend()
plt.title('Training Losses')
plt.tight_layout()
# plt.savefig('../plots/dataset.png')
plt.savefig('../plots/training_loss.png')
plt.show()
