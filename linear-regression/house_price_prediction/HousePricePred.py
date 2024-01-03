import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class HousePredictPricer:
    def __init__(self):
        self.data = pd.read_csv('HousingData.csv')
        self.data.head()
        self.clean_the_cols()
        self.X_COLS = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B']
        self.Y_COLS = ['MEDV']

    def clean_the_cols(self):
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                self.data[column].fillna(self.data[column].mean(), inplace=True)

        self.data.isnull().sum()

    def get_significant_cols(self):
        correlation_with_medv = self.data.corr()['MEDV'].sort_values(ascending=False)
        significant_columns = correlation_with_medv[abs(correlation_with_medv) >= 0.5].index.tolist()
        significant_columns.remove('MEDV')
        print(significant_columns)
        return significant_columns

    def init_the_model(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.data[self.get_significant_cols()], self.data[self.Y_COLS], test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        mse = mean_squared_error(Y_test, prediction)
        print("Mean Squared Error:", mse)

        r2 = r2_score(Y_test, prediction)
        print("R-squared:", r2)



if __name__ == '__main__':
    HousePredictPricer().init_the_model()