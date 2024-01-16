import pandas
import numpy
import matplotlib.pyplot as plt

class DataPreparation:
	def __init__(self, csv_path):
		"""
		Cette classe prend en entrée un chemin de fichier csv.
		Elle split le jeu de donnée en 2 bases 
		+ une train 75 %
		+ une test 25 %
		Ce 2 bases, la classe va les splits en 2 

		+ un vecteur x (qui contient les indexs temporels)
		+ un vecteyr y (qui contient les valeurs à prédire)
		En tout cette va extraire 4 arrays.
		x_train
		y_train
		x_test
		y_test
		"""
		self.dataset_df = pandas.read_csv(csv_path)
		self.dataset_df["month"] = pandas.to_datetime(self.dataset_df["month"])
		self.prepare_data()

	def prepare_data(self):
		number_of_rows = len(self.dataset_df)
		self.dataset_df["index_mesure"] = numpy.arange(0, number_of_rows, 1)

		dataset_train_df = self.dataset_df.iloc[ : int(number_of_rows*0.75)]
		dataset_test_df = self.dataset_df.iloc[int(number_of_rows*0.75): ]

		self.x_train = dataset_train_df[['index_mesure']].values
		self.y_train = dataset_train_df[['passengers']].values

		self.x_test = dataset_test_df[['index_mesure']].values
		self.y_test = dataset_test_df[['passengers']].values


	def show_graph(self):
		plt.figure(figsize=(15, 6))
		plt.plot(self.dataset_df["month"], self.dataset_df["passengers"], "o:")
		plt.show()