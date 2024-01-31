from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy

class Regression:
    def __init__(self, data_preparation_object):
        self.data_preparation_object = data_preparation_object
        self.model = LinearRegression()

        self.model.fit(data_preparation_object.x_train, data_preparation_object.y_train)

        y_train_predicted = self.model.predict(data_preparation_object.x_train)
        mean_train_absolute_error = numpy.mean(numpy.abs(y_train_predicted - data_preparation_object.y_train))
        print(f"sur le jeu de train : {mean_train_absolute_error=:.2f}")

        y_test_predicted = self.model.predict(data_preparation_object.x_test)
        mean_test_absolute_error = numpy.mean(numpy.abs(y_test_predicted - data_preparation_object.y_test))
        print(f"sur le jeu de test : {mean_test_absolute_error=:.2f}")

        self.show_model_predictions(y_train_predicted, y_test_predicted)

    def show_model_predictions(self, y_train_predicted, y_test_predicted):
        plt.figure(figsize=(15, 6))

        conf = 0.95
        residuals = y_test_predicted - self.data_preparation_object.y_test
        std_residuals = numpy.std(residuals)
        n = len(y_test_predicted)
        margin_of_error = 2.5 * (std_residuals / numpy.sqrt(n))

        min_interval = (y_test_predicted - margin_of_error).ravel()
        max_interval = (y_test_predicted + margin_of_error).ravel()       
        plt.plot(self.data_preparation_object.dataset_train_df["Years"], self.data_preparation_object.y_train, "bo:", label="TimeSeries Data")
        plt.plot(self.data_preparation_object.dataset_train_df["Years"], y_train_predicted, "b", label="fitted Adiitve Data")
        plt.plot(self.data_preparation_object.dataset_test_df["Years"], self.data_preparation_object.y_test, "ro:", label="True Future Data")
        plt.plot(self.data_preparation_object.dataset_test_df["Years"], y_test_predicted, "r", label="Forecasted Additive Model Data")

        plt.fill_between(self.data_preparation_object.dataset_df['Years'][len(self.data_preparation_object.x_train):],
        min_interval, max_interval,
        color="lightgray", alpha=0.9, label="95% Confidence Interval")
        
        plt.legend()
        plt.show()
