import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from DatasetGenerator import DataGenerator

class DataLoader:
	def __init__(self, ntrain, nval, ntest, batch_size=64, img_shape=(256,256,3), undersample=True, augment_data=True, shuffle=True, plot_distribuition=True):
		self.ntrain       = ntrain
		self.nval         = nval
		self.ntest        = ntest
		self.batch_size   = batch_size
		self.img_shape    = img_shape
		self.undersample  = undersample
		self.augment_data = augment_data
		self.shuffle      = shuffle
		self.train_data   = self.load_train_data()
		self.val_data     = self.load_validation_data()
		self.test_data    = self.load_test_data()

	def load_train_data(self):
		train_data = pd.read_csv('dataset/mytrain_3.txt', header=None, index_col=None)[0].str.split(' ', 1)
		train_labels = np.vstack(train_data.apply(lambda x: x[1]).values).astype(np.int8)[:self.ntrain]

		if self.undersample:
			t_label=[]
			for i in train_labels:
				t_label.append(i[0])
			t_label=np.array(t_label)
			normal = np.where(t_label==0)[0]
			abnormal = np.where(t_label==1)[0]
			Cardiomegaly=np.where(t_label==2)[0]
			undersampling = len(Cardiomegaly)
			samples_to_train = np.concatenate([normal[:undersampling], abnormal[:undersampling], Cardiomegaly[:len(Cardiomegaly)]])
			self.ntrain = len(samples_to_train)
			t_label = t_label[samples_to_train]
			t_label = np_utils.to_categorical(t_label, num_classes=3)
			train_images = train_data.apply(lambda x: '../preprocessed256/' + x[0]).values[samples_to_train]
		else:
			train_images = train_data.apply(lambda x: '../preprocessed256/' + x[0]).values[:self.ntrain]
		return (train_images, t_label)

	def load_validation_data(self):

		val_data = pd.read_csv('dataset/myval_3.txt', header=None, index_col=None)[0].str.split(' ', 1)
		val_labels = np.vstack(val_data.apply(lambda x: x[1]).values).astype(np.int8)[:self.nval]

		if self.undersample:
			label=[]
			for i in val_labels:
				label.append(i[0])
			label=np.array(label)
			normal = np.where(label==0)[0]
			abnormal = np.where(label==1)[0]
			Cardiomegaly=np.where(label==2)[0]

			# validation 시 normal, abnormal, Cardiomegaly의 갯수가 같은 상태에서 validation을 시행하자.
			undersampling = len(Cardiomegaly)
			samples_to_val = np.concatenate([normal[:undersampling], abnormal[:undersampling], Cardiomegaly[:undersampling]])
			self.nval  = len(samples_to_val)
			label = label[samples_to_val]

			label = np_utils.to_categorical(label, num_classes=3)
			val_images = val_data.apply(lambda x: '../preprocessed256/' + x[0]).values[samples_to_val]
		else:
			val_images = val_data.apply(lambda x: '../preprocessed256/' + x[0]).values[:self.nval]
		return (val_images, label)

	def load_test_data(self):
		test_data = pd.read_csv('dataset/mytest_3.txt', header=None, index_col=None)[0].str.split(' ', 1)
		test_labels = np.vstack(test_data.apply(lambda x: x[1]).values).astype(np.int8)[:self.ntest]

		if self.undersample:
			label=[]
			for i in test_labels:
				label.append(i[0])
			label=np.array(label)
			normal = np.where(label==0)[0]
			abnormal = np.where(label==1)[0]
			Cardiomegaly=np.where(label==2)[0]

			# validation 시 normal, abnormal, Cardiomegaly의 갯수가 같은 상태에서 validation을 시행하자.
			undersampling = len(Cardiomegaly)
			samples_to_test = np.concatenate([normal[:undersampling], abnormal[:undersampling], Cardiomegaly[:undersampling]])
			self.ntest  = len(samples_to_test)
			label = label[samples_to_test]

			label = np_utils.to_categorical(label, num_classes=3)
			test_images = test_data.apply(lambda x: '../preprocessed256/' + x[0]).values[samples_to_test]
		else:
			test_images = test_data.apply(lambda x: '../preprocessed256/' + x[0]).values[:self.ntest]
		return (test_images, label)


	def load_train_generator(self):
		return DataGenerator(*self.train_data,
		                     batch_size=min(self.batch_size, self.ntrain),
		                     image_dimensions=self.img_shape,
		                     shuffle=self.shuffle,
		                     augment=self.augment_data)

	def load_validation_generator(self):
		return DataGenerator(*self.val_data,
		                     batch_size=min(self.batch_size, self.nval),
		                     image_dimensions=self.img_shape,
		                     shuffle=False,
		                     augment=False)

	def load_test_generator(self):
		return DataGenerator(*self.test_data,
		                     batch_size=1,
		                     image_dimensions=self.img_shape,
		                     shuffle=False,
		                     augment=False)
