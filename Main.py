import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback

from CNN_Model import BinaryModel
from DataLoader import DataLoader

import json#


############################# Parameters ############################
test_trained_model     = False
load_previous_weights  = False

samples_to_train  = 3000#78468#3000 #max: 78468
samples_to_val    = 250#11219#250  #max: 11219
samples_to_test   = 2000#22433#2000 #max: 22433
epochs = 2
batch_size = 128
image_shape = (128, 128, 3)
model_learn_rate = 0.001
model_architecture = 'dense121'

#decrease resource usage:
idle_time_on_batch = 0.1
idle_time_on_epoch = 20
#####################################################################




print('##### Loading Data #####')
################################################ Load Data ################################################
data_loader = DataLoader(batch_size=batch_size,
                        img_shape=image_shape,
                        ntrain=samples_to_train,
                        nval=samples_to_val,
                        ntest=samples_to_test,
                        undersample=True,
                        augment_data=True,
                        shuffle=True,
                        plot_distribuition=False)

train_data = data_loader.load_train_generator()
val_data   = data_loader.load_validation_generator()
test_data  = data_loader.load_test_generator()


################################################ Create NN model ################################################
if not test_trained_model:
	print('##### Building NN Model #####')
	model = BinaryModel(model2load=model_architecture,
	                    percent2retrain=0.6,
	                    image_dimensions=image_shape,
	                    n_classes=3).get_model()


	#if load_previous_weights == True:
	#	print('Loading Model Weights')
	#	model.load_weights("model_weights.hdf5")
"""
	METRICS = [
		keras.metrics.TruePositives(name='tp'),
		keras.metrics.FalsePositives(name='fp'),
		keras.metrics.TrueNegatives(name='tn'),
		keras.metrics.FalseNegatives(name='fn'), 
		keras.metrics.BinaryAccuracy(name='accuracy'),
		keras.metrics.Precision(name='precision'),
		keras.metrics.Recall(name='recall'),
		keras.metrics.AUC(name='auc'),
	]"""

	optimizer = Adam(lr=model_learn_rate,
	                 beta_1=0.9,
	                 beta_2=0.999,
	                 epsilon=1e-08,
	                 decay=0.0,
	                 amsgrad=False)

	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])
"""
https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
https://dsbook.tistory.com/64
https://wordbe.tistory.com/entry/ML-Cross-entropyCategorical-Binary%EC%9D%98-%EC%9D%B4%ED%95%B4
"""


	learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
	                                            patience=5,
	                                            verbose=1,
	                                            factor=0.5,
	                                            min_lr=0.00000001)

	early_stop = EarlyStopping(monitor="val_loss",
	                           mode="min",
	                           patience=12)

	checkpoint = ModelCheckpoint('model_weights_sigmoid_3.hdf5',
	                             monitor='val_accuracy',#val_loss to val_acc
	                             verbose=1,
	                             save_best_only=True,
	                             mode='max',# according to monitor : val_accuracy
	                             #save_weights_only=True
								)

	# sleep after each batch and epoch (prevent laptop from melting) (sleeps for x sec)(remove for faster training)
	idle = LambdaCallback(on_epoch_end=lambda batch, logs: time.sleep(idle_time_on_epoch), on_batch_end=lambda batch,logs: time.sleep(idle_time_on_batch))

	# save model to json file
	with open("model.json", "w") as json_model:
		json_model.write(model.to_json())


	print('##### Training Model #####')
	########################################## Train Model ###############################################
	model.summary()
	#34113, 1950 : train 기준 normal 및 abnormal이 34113개, Car~이 1950개 
	class_w={[1, 0, 0] : 0.02, [0, 1, 0] : 0.02, [0, 0, 1] : 0.96}
	history = model.fit_generator(generator=train_data,
	                              validation_data=val_data,
	                              epochs=epochs,
	                              steps_per_epoch=len(train_data),
	                              verbose=2,
								  class_weight=class_w,
	                              callbacks=[learning_rate_reduction, early_stop, checkpoint, idle],
	                              # use_multiprocessing=True,
	                              workers=2
	                              )

	############################# Check Loss and Accuracy graphics over training ########################
	fig, ax = plt.subplots(2, 1, figsize=(6, 6))
	ax[0].plot(history.history['loss'], label="TrainLoss")
	ax[0].plot(history.history['val_loss'], label="ValLoss")
	ax[0].legend(loc='best', shadow=True)
	plt.show()
	plt.savefig('originLoss.png')

	fig, ax = plt.subplots(2, 1, figsize=(6, 6))
	ax[1].plot(history.history['acc'], label="TrainAcc")
	ax[1].plot(history.history['val_acc'], label="ValAcc")
	ax[1].legend(loc='best', shadow=True)
	plt.show()
	plt.savefig('originAcc.png')

	plt.plot(history.history['loss'], label='trainLoss')
	plt.plot(history.history['val_loss'], label='testLoss')
	plt.legend()
	plt.show()
	plt.savefig("changeLoss.png")

	plt.plot(history.history['acc'], label='trainAcc')
	plt.plot(history.history['val_acc'], label='testAcc')
	plt.legend()
	plt.show()
	plt.savefig("changeAcc.png")


"""
#  ************pretrained model 사용하는 경우*************
else: # if use_trained_model:
	print('##### Loading NN Model #####')
	from keras.models import model_from_json

	with open('model.json', 'r') as json_model:
		model = model_from_json(json_model.read())

	print('Loading Model Weights')
	model.load_weights("model_weights.hdf5")

	optimizer = Adam(lr=model_learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['acc'])
"""



print('##### Evaluating Model on Test Data #####')
################################# Evaluate model on Test Data ############################
test_score = model.evaluate_generator(test_data, verbose=2)
print('\nModel Accuracy: ', test_score[1])

print('\nParameters used:',
	'\ntrain_samples:   ',samples_to_train,
	'\nepochs:          ',epochs,
	'\nbatch_size:      ',batch_size,
	'\ninit_learn_rate: ',model_learn_rate)


print('##### Plotting Confusion Matrix #####')
predict_out = model.predict_generator(test_data, verbose=2)
test_predict = (predict_out > 0.5).astype(np.int8)

conf_matrix = confusion_matrix(y_true=data_loader.test_data[1], y_pred=test_predict)

sns.heatmap(conf_matrix, annot=True, cmap='Blues', cbar=False, square=True, xticklabels=['Normal','Abnormal'], yticklabels=['Normal','Abnormal'])
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()