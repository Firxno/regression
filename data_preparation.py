import pandas
import numpy
import matplotlib.pyplot as plt

class DataPreparation:
    def __init__(self, csv_path):
        self.dataset_df = pandas.read_csv(csv_path)
        self.prepare_data()

    def prepare_data(self):
        number_of_rows = len(self.dataset_df)
        self.dataset_df["index_mesure"] = numpy.arange(0, number_of_rows, 1)

        self.dataset_df["Years"] = pandas.to_datetime(self.dataset_df["Years"])
        self.dataset_df["Years_name"] = self.dataset_df["Years"].dt.strftime('%B')
        boolean_df = pandas.get_dummies(self.dataset_df["Years_name"])

        self.dataset_df = self.dataset_df.join(boolean_df)

        self.dataset_train_df = self.dataset_df.iloc[:int(number_of_rows*0.75)]
        self.dataset_test_df = self.dataset_df.iloc[int(number_of_rows*0.75):]

        self.x_train = self.dataset_train_df[['index_mesure'] + list(boolean_df.columns)].values
        self.y_train = self.dataset_train_df[['Sales']].values

        self.x_test = self.dataset_test_df[['index_mesure'] + list(boolean_df.columns)].values
        self.y_test = self.dataset_test_df[['Sales']].values

    def show_graph(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.dataset_df["Years"], self.dataset_df["Sales"], "o:")
        
        plt.show()
