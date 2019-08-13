import os
import sys
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

print("COMEÇOU!!!")

## Pass argument with the number of the training when call on bash

PATH = '../data/'
N_TRAINING = int(sys.argv[1])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
rmsprop = RMSprop(lr=1e-4)
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)

def geraGrafico(history, approach, architecture, optimizer, patience,  activation):
	history_dict = history.history
	loss_values = history_dict['loss']
	val_loss_values = history_dict['val_loss']

	epochs_x = range(1, len(loss_values) + 1)

	model_name = approach + '-' + architecture + '-' + optimizer + '-' + str(patience) + '-' + activation

	fig = plt.figure(figsize=(10,10))
	plt.subplot(2,1,1)
	plt.plot(epochs_x, loss_values, 'bo', label='Loss do treinamento')
	plt.plot(epochs_x, val_loss_values, 'b', label='Loss da validação')
	plt.title('Loss e Acurácia do modelo ' + model_name)
	plt.xlabel('Épocas')
	plt.ylabel('Loss')
	plt.legend(loc=1)
	plt.subplot(2,1,2)
	acc_values = history_dict['acc']
	val_acc_values = history_dict['val_acc']
	plt.plot(epochs_x, acc_values, 'bo', label='Acurácia do treino')
	plt.plot(epochs_x, val_acc_values, 'b', label='Acurácia da validação')
	#plt.title('Training and validation accuracy ' + model_name)
	plt.xlabel('Épocas')
	plt.ylabel('Acurácia')
	plt.legend(loc=4)
	fig.savefig('figures/'+ model_name + '.png')

dict_training = {
    'approach': ['new_approach'],
    'architecture': ['alexnet', 'lenet'],
    'optimizer': ['sgd', 'rmsprop', 'adam'],
    'patience': [5,10,15],
    'activation': ['relu', 'elu', 'selu', 'lrelu']
}

params = list(itertools.product(*dict_training.values()))
params = params[12*N_TRAINING:(12*N_TRAINING)+12]

for app,arch,opt,pat,act in params:
	nome_modelo = app + '-' + arch + '-' + opt + '-' + str(pat) + '-' + act
	train_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)
	validation_datagen = ImageDataGenerator(rescale=1./255)

	train_path = PATH + app + '/train'
	validation_path = PATH + app + '/validation'
	test_path = PATH + app + '/test'

	train_generator = train_datagen.flow_from_directory(train_path,
		class_mode='binary',
		color_mode='grayscale',
		seed=42,
		shuffle=True,
		batch_size=32)

	validation_generator = validation_datagen.flow_from_directory(validation_path,
		class_mode='binary',
		color_mode='grayscale',
		seed=42,
		shuffle=True,
		batch_size=32)

	test_generator = test_datagen.flow_from_directory(test_path,
		class_mode='binary',
		color_mode='grayscale',
		shuffle=False,
		batch_size=1,
		seed=42)

	STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
	STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
	STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

	print("## CRIANDO O MODELO ##")

	if (arch == 'alexnet'):
		model = AlexNet(img_width=256, img_height=256, img_depth=1, activation=act)
	else:
		model = LeNet(img_width=256, img_height=256, img_depth=1, activation=act)

	if (opt == 'sgd'):
		optimizer = sgd
	elif (opt == 'rmsprop'):
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
		steps_per_epoch=STEP_SIZE_TRAIN,
		validation_data=validation_generator,
		validation_steps=STEP_SIZE_VALID,
		callbacks=[EarlyStopping(monitor='val_acc', patience=pat)],
		verbose=False)

	print("## GERANDO O GRÁFICO ##")

	geraGrafico(history, app, arch, opt, pat, act)

	print("## TESTA O MODELO ##")
	print("MODELO: " + nome_modelo)

	results = model.model.predict_generator(test_generator,
		steps=len(test_generator))

	y_pred = results.copy()
	y_pred = [1 if i >= 0.5 else 0 for i in results]

	y_true = test_generator.classes

	print("Acurácia: ", accuracy_score(y_true, y_pred))
	print("F-score: ", f1_score(y_true, y_pred))

	print("Matriz de confusão: ")
	print(confusion_matrix(y_true, y_pred))

	model.save_weights('weights/' + nome_modelo + '.h5')

