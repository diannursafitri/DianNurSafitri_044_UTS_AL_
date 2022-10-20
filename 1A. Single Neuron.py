#Dian Nur Safitri
#21091397044

#inisialisasi library
import numpy as np

#input layer feature 10
inputs = [1.5,3.1,4.5,3.6,5,2,8.2,7,1.7,9.1]
weights = [3.5,5.1,8.6,5.2,8,7,3.5,9,1.9,5.7]

#neuron 1
bias = 3

#menampilkan keluaran
outputs = np.dot(weights, inputs) + bias
print(outputs)