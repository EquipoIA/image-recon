from tensorflow.keras.models import load_model
import numpy as np
import cv2
model = load_model('handwriting.model')

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

	# add the image to our list of output images
	images.append(image)

# construct the montage for the images
montage = build_montages(images, (96, 96), (8, 8))[0]

# show the output montage
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)