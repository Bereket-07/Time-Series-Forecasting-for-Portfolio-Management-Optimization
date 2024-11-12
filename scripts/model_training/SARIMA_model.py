from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def sarima_model(train_data, test_data, column_name="TSLA"):
    logger.info("SARIMA model training and prediction")
    try:
        # Select the specified column for modeling
        train_series = train_data[column_name].values
        test_series = test_data[column_name].values
        
        logger.info("Parameter selection using auto_arima")
        sarima_model = auto_arima(train_series, seasonal=True, m=12, trace=True)

        logger.info(f"Best SARIMA Model Parameters: Order: {sarima_model.order}, Seasonal Order: {sarima_model.seasonal_order}")
        
        logger.info("Training the SARIMA model")
        sarima_fitted = SARIMAX(train_series, 
                                order=sarima_model.order, 
                                seasonal_order=sarima_model.seasonal_order).fit()
        
        logger.info("Forecasting on test set")
        forecast = sarima_fitted.forecast(len(test_series))
        print(f"The forecast of the test data is: {forecast}")
        
        return sarima_fitted, forecast
    except Exception as e:
        logger.error(f"Error occurred while SARIMA model training and forecasting: {e}")
        return None, None
