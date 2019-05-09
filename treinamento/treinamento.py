import os
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
import itertools
from models import AlexNet, LeNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH = '/content/'

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
rmsprop = RMSprop(lr=1e-4)
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)

def geraGrafico(history, approach, architecture, optimizer, activation):
	history_dict = history.history
	loss_values = history_dict['loss']
	val_loss_values = history_dict['val_loss']

	epochs_x = range(1, len(loss_values) + 1)

	fig = plt.figure(figsize=(10,10))
	plt.subplot(2,1,1)
	plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
	plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
	plt.title('Training and validation Loss and Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	#plt.legend()
	plt.subplot(2,1,2)
	acc_values = history_dict['acc']
	val_acc_values = history_dict['val_acc']
	plt.plot(epochs_x, acc_values, 'bo', label='Training acc')
	plt.plot(epochs_x, val_acc_values, 'b', label='Validation acc')
	#plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Acc')
	plt.legend()
	fig.savefig('figures/'+ approach + '-' + architecture + '-' + optimizer + '-' + activation + '.png')

dict_training = {
    'approach': ['approach1','approach2','approach3'],
    'architecture': ['alexnet', 'lenet'],
    'optimizer': ['sgd', 'rmsprop', 'adam'],
    #'patience': [10,20,35],
    'activation': ['relu', 'elu', 'selu', 'lrelu']
}

params = list(itertools.product(*dict_training.values()))

for app,arch,opt,act in params:
	train_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)
	validation_datagen = ImageDataGenerator(rescale=1./255)

	train_path = PATH + app + '/train'
	validation_path = PATH + app + '/validation'
	test_path = PATH + app + '/test'

	train_generator = train_datagen.flow_from_directory(train_path,
		class_mode='binary',
		color_mode='grayscale')

	validation_generator = validation_datagen.flow_from_directory(validation_path,
		class_mode='binary',
		color_mode='grayscale')

	test_generator = test_datagen.flow_from_directory(test_path,
		class_mode='binary',
		color_mode='grayscale')

	print("## CRIANDO O MODELO ##")

	if (arch == 'alexnet'):
		model = AlexNet(img_width=256, img_height=256, img_depth=256, activation=act)
	else:
		model = LeNet(img_width=256, img_height=256, img_depth=256, activation=act)

	if (opt = 'sgd'):
		optimizer = sgd
	elif (opt = 'rmsprop'):
		optimizer = rmsprop
	else:
		optimizer = adam

	print("## COMPILANDO O MODELO ##")

	model.model.compile(loss='binary_crossentropy',
		optimizer=optimizer,
		metrics=['acc'])

	print("## TREINANDO O MODELO ##")

	history = model.model.fit_generator(
		train_generator,
		epochs=200,
		steps_per_epoch=len(train_generator),
		validation_data=validation_generator,
		validation_steps=len(validation_generator),
		callbacks=EarlyStopping(monitor='val_acc', patience=10),
		verbose=False)

	print("##GERANDO O GRÁFICO##")

	geraGrafico(history, app, arch, opt, act)

	print("## TESTA O MODELO ##")

	results = model.model.predict_generator(test_generator,
		steps=len(test_generator))

	y_pred = results.copy()
	y_pred = [1 if i >= 0.5 else 0 for i in results]

	y_true = test_generator.classes

	print("Acurácia: ", accuracy_score(y_true, y_pred))
	if (app = 'approach2'):
		print("F-score (macro): ", f1_score(y_true, y_pred, average='macro'))
	else:
		print("F-score (micro): ", f1_score(y_true, y_pred, average='micro'))

	print("Matriz de confusão: ")
	print(confusion_matrix(y_true, y_pred))

