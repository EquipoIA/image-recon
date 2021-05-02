# linea en terminal para usarlo
# python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model

import matplotlib
matplotlib.use("Agg")
from pyimagesearch.models import ResNet
from pyimagesearch.az_dataset import load_mnist_dataset
from pyimagesearch.az_dataset import load_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# argumentos para llamar a la funcion en al terminal
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True,
	help="path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output training history file")
args = vars(ap.parse_args())

# definir numero de epochs, taza de aprendixaje y tamaño de muestra
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# cargar las dos bases de datos 
print("[INFO] loading datasets...")
(azData, azLabels) = load_az_dataset(args["az"])
(digitsData, digitsLabels) = load_mnist_dataset()

#redefinir etiquetas de letras, par ano confundirse con las de numeros
azLabels += 10
#juntar bases
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

#cambiar tamaño de imagenes y cambiar intensidad de los pixeles de 0 a 1
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")
data = np.expand_dims(data, axis=-1)
data /= 255.0

#transformar etiquetas en vectores
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# tomar en cuenta asimetria en etiquetas
classTotals = labels.sum(axis=0)
classWeight = {}

# calcular pesos de cada clase
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

#dividir los datos en entrenamiento(80%) y prueba(20%)
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

# generador de imagenes para aumentar tamaño de datos
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

# inicializa y compila red neuronal
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
	(64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# entrenar red neuronal
print("[INFO] training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	class_weight=classWeight,
	verbose=1)

# nombres de etiquetas
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# evaluar la red neuronal
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# guardar el modelo creado
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

# graficar historial de entrenamiento
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

#imprimir una imagen con muestra
images = []

# seleccionar muestras aleatorias de los datos
for i in np.random.choice(np.arange(0, len(testY)), size=(64,)):
	# hacer las predicciones
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]

	#extraer imagen correspondiente de los datos
	image = (testX[i] * 255).astype("uint8")
	color = (0, 255, 0)

	# si al prediccion es incorrecta, ponerla en rojo
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)

	# agregar prediccion y cambiar tamaño para verlas mejor
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)

	# agregar la imagen al arreglo
	images.append(image)

# construir un montaje con el arreglo de imagenes
montage = build_montages(images, (96, 96), (8, 8))[0]
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)