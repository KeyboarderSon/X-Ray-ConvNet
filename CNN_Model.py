from keras.models import Model

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, concatenate
from keras.applications import densenet


class BinaryModel:
	def __init__(self, model2load='custom', percent2retrain=1/4, image_dimensions=(128,128,3), n_classes=1):
		self.input_dim  = image_dimensions
		self.n_classes  = n_classes
		self.model      = self.select_model(model2load, percent2retrain)


	def select_model(self, model2load, percent2retrain):
		'Selects the desired model to be loaded'
		if 0>percent2retrain>1:
			raise Exception('Invalid train percentage chosen! Value must be between 0-1')
		elif model2load == 'dense121':
			return self.dense_net121(percent2retrain)
		else:
			raise Exception ('No valid net has been chosen! Choose dense121')


	#include_top=False로 하면 맨 위에 분류 층이 포함되지 않은 네트워크 로드
	#-> 특징 추출에 이상적
	def dense_net121(self, percent2retrain):
		'Returns a Densenet121 architecture NN'
		dense_model = densenet.DenseNet121(input_shape=self.input_dim,
                                           weights='imagenet',
                                           include_top=False)
		# freeze base layers
		if percent2retrain < 1:
			for layer in dense_model.layers[:-int(len(dense_model.layers)*percent2retrain)]: layer.trainable = False

		# add classification top layer
		model = Sequential()
		model.add(dense_model)
		model.add(Flatten())
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation='sigmoid'))
		#model.add(Dense(self.n_classes, activation='softmax'))

		return model



	def get_model(self):
		'Returns the created model'
		return self.model
