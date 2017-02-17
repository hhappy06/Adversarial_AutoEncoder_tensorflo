import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

_TESTING_OUTPUT_FILE_ = './result/test.res'
_TESTING_OUTPUT_IMG_ = './result/test.png'
_N_LABEL_ = 10

def show_test_result():
	data = []
	with open(_TESTING_OUTPUT_FILE_, 'r') as input_data:
		for item in input_data:
			item.replace('\n', '')
			item_str = item.split(',')
			item_data = [float(item_str[0]), float(item_str[1]), int(item_str[2])]
			data.append(item_data)
	data = np.array(data)
	plt.figure()
	color = cm.rainbow(np.linspace(0,1,_N_LABEL_))
	for l, c in zip(range(10), color):
		ix = np.where(data[:,2]==l)
		plt.scatter(data[ix,0], data[ix, 1], c=c, label=l, s=8, linewidth=0)
	plt.savefig(_TESTING_OUTPUT_IMG_)
	plt.close()

if __name__ == '__main__':
	show_test_result()