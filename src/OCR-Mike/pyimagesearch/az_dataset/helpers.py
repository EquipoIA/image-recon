# import the necessary packages
from tensorflow.keras.datasets import mnist
import numpy as np

#cargar la base de datos de letras
def load_az_dataset(datasetPath):
	# inicialisar arreglos de datos y etiquetas
	data = []
	labels = []
	# ciclo sobre las filas de la base de datos
	for row in open(datasetPath):
		# asignar imagen y etiqueta
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# convertir imagenes de un arreglo a una matriz
		image = image.reshape((28, 28))
		# agregar imagenes y etiquetas a sus arreglos
		data.append(image)
		labels.append(label)
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")
	return (data, labels)

#cargar la base de datos de numeros que ya esta incluida en keras
def load_mnist_dataset():
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	#juntar datos de entrenamiento y de prueba
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	return (data, labels)

	