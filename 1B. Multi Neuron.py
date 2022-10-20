#Dian Nur Safitri
#21091397044

#inisialisasi library
import numpy as np

#input layer feature 10
inputs = [1,3,2,2.4,6,1.3,5,2.2,1.10,2.2]
weights = [
    [0.5,0.3,-0.5,2.2,0.3,2.2,0.20,-0.18,0.30,0.6],
    [0.8,-0.80,0.26,-0.5,1.0,0.7,0.6,0.30,-0.15,0.3],
    [-0.35,-0.27,0.17,0.87,0.20,1.14,0.71,-0.03,0.29,-1.0],
    [0.21,-0.5,-0.4,0.1,2.03,0.17,0.7,0.10,-2.0,3.2],
    [0.5,0.9,-0.15,0.6,0.50,3.0,-0.15,0.15,-0.28,0.28]
]

#neuron 5
biases = [3,5,0.9,2,0.7]

#menampilkan keluaran
outputs = np.dot (weights, inputs) + biases
print(outputs)