from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def arima_model(train_data, test_data, column_name="TSLA"):
    logger.info("ARIMA model training and prediction")
    try:
        # Select the specified column for modeling
        train_series = train_data[column_name].values
        test_series = test_data[column_name].values
        
        logger.info("Parameter selection")
        arima_model = auto_arima(train_series, seasonal=False, trace=True)
        
        logger.info("Training the ARIMA model")
        model = ARIMA(train_series, order=arima_model.order)
        arima_fitted = model.fit()
        
        logger.info("Forecasting on test set")
        forecast = arima_fitted.forecast(len(test_series))
        print(f"The forecast of the test data is: {forecast}")
        
        return arima_fitted , forecast
    except Exception as e:
        logger.error(f"Error occurred while ARIMA model training and forecasting: {e}")