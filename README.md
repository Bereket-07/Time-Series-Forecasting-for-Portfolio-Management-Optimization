# Time-Series-Forecasting-for-Portfolio-Management-Optimization

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Folder Organization](#folder-organization)
- [Setup](#setup)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Overview: Key Functionalities


## 1. Project Overview
- **Objective**: Build a machine learning system for optimizing a sample investment portfolio using time series     forecasting models.
1. **Data Collection**: 
  - Collect historical daily closing prices for assets including Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) using yfinance.
  - Clean and organize data, addressing outliers, seasonality, and trends for accurate forecasting.
2. **Forecasting Models**: 
  - Develop and implement time series models such as ARIMA, SARIMA, and LSTM for predicting future prices of TSLA, BND, and SPY.
  - Use the models to generate 12-month forecasted prices for each asset.
3. **Portfolio Optimization**: 
  - Create a portfolio allocation strategy by computing daily returns and a covariance matrix.
  - Optimize the portfolio by adjusting asset weights to maximize returns while minimizing risks based on forecasted prices.
  - Use the Sharpe Ratio to identify the best risk-adjusted allocation
4. **Model Evaluation and Analysis**: 
  - Measure performance of forecast models using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
  - Analyze portfolio performance by computing expected return, volatility, and Value at Risk (VaR).
  - Visualize cumulative returns and risk-return trade-offs for optimal asset allocation.

# Tools & Libraries Used

1. **Programming Language**: [![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=yellow)](https://www.python.org/)
2. **Data Manipulation**: [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
3. **Numerical Computing**: [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
4. **Data Visualization**: [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)
5. **Time Series Modeling**: [![Statsmodels](https://img.shields.io/badge/Statsmodels-2C2D72?style=flat&logo=python&logoColor=white)](https://www.statsmodels.org/)
6. **Deep Learning Framework**: [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
7. **Statistical Analysis**: [![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
8. **Machine Learning Library**: [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
9. **Logging**: [![Logging](https://img.shields.io/badge/Logging-4B8BBE?style=flat&logo=python&logoColor=yellow)](https://docs.python.org/3/howto/logging.html)
10. **Portfolio Optimization**: [![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-333333?style=flat&logo=python&logoColor=white)](https://pyportfolioopt.readthedocs.io/en/latest/)
11. **Database**: [![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white)](https://www.mysql.com/)
12. **Web Framework**: [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
13. **Environment Management**: [![Pip](https://img.shields.io/badge/Pip-005A8B?style=flat&logo=pypi&logoColor=white)](https://pip.pypa.io/en/stable/)
14. **Version Control**: [![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)](https://git-scm.com/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/)
15. **Code Formatting & Linting**: [![Black](https://img.shields.io/badge/Black-000000?style=flat&logo=python&logoColor=white)](https://github.com/psf/black)
16. **Continuous Integration (CI)**: [![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)

## Folder Organization

```

📁.github
└──
    └── 📁workflows
         └── 📃unittests.yml
└── 📁data
└── 📁notebooks
         └── 📃__init__.py
         └── 📓eda.ipynb
         └── 📓future_forecasting.ipynb
         └── 📓model_training_and_predictions.ipynb
         └── 📓portfolio.ipynb
└── 📁scripts
         └── 📁model_training
                └── 📁Forecast_future_market_trends
                        └── 📃predict_future_market_trends.py
                └── 📃ARIMA_model.py
                └── 📃evalutaion_metrics.py
                └── 📃LSTM_model.py
                └── 📃SARIMA_model.py
         └── 📁optmize_portfolio
                └── 📃optimize_portfolio.py
         └── 📃collection_and_cleaning.py
         └── 📃time_series_forecasting_model
└── 💻src
    └── 📁dashboard-div
                    └── 📝app.py
                    └── 📝setup.py
└── ⌛tests
         └── 📃__init__.py

└── 📜.gitignore
└── 📰README.md
└── 🔋requirements.txt
└── 📇templates.py

```


### **Usage**

These modules are designed to be used in conjunction with each other to streamline the data analysis process, from data preparation and cleaning to in-depth analysis and model creation.

- **💻src**: The main source code of the project, including the Streamlit dashboard and other related files.

  - **📁dashboard-div**: Holds the code for the dashboard.
    - **📝app.py**: Main application file for the dashboard.
    - **📝README.md**: Documentation specific to the dashboard component.

- **⌛tests**: Contains test files, including unit and integration tests.

  - \***\*init**.py\*\*: Initialization file for the test module.

- **📜.gitignore**: Specifies files and directories to be ignored by Git.

- **📰README.md**: The main documentation for the entire project.

- **🔋requirements.txt**: Lists the Python dependencies required to run the project.

- **📇templates.py**: Contains templates used within the project, possibly for generating or processing data.

## Setup

1. Clone the repo

```bash
git clone https://github.com/Bereket-07/Time-Series-Forecasting-for-Portfolio-Management-Optimization.git
```

2. Change directory

```bash
cd Time-Series-Forecasting-for-Portfolio-Management-Optimization
```

3. Install all dependencies

```bash
pip install -r requirements.txt
```

4. change directory to run the Flask app locally.

```bash
cd src
```

5. Start the Flask app

```bash
uvicorn dashboard-div.app:app --reload                           
```
6. go to the front end then run the html

## Contributing

We welcome contributions to this project! To get started, please follow these guidelines:

### How to Contribute

1. **Fork the repository**: Click the "Fork" button at the top right of this page to create your own copy of the repository.
2. **Clone your fork**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
3. **Create a new branch**: Create a new branch for your feature or bugfix.
   ```bash
   git checkout -b feature/your-feature
   ```
4. **Make your changes**: Implement your feature or fix the bug. Ensure your code adheres to the project's coding standards and style.
5. **Commit your changes**: Commit your changes with a descriptive message.
   ```bash
   git add .
   git commit -m 'Add new feature or fix bug'
   ```
6. **Push your branch**: Push your branch to your forked repository.
   ```bash
   git push origin feature/your-feature
   ```
7. **Create a Pull Request**: Go to the repository on GitHub, switch to your branch, and click the `New Pull Request` button. Provide a detailed description of your changes and submit the pull request.

## Additional Information

- **Bug Reports**: If you find a bug, please open an issue in the repository with details about the problem.

- **Feature Requests**: If you have ideas for new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License

### Summary

The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). It is a simple and easy-to-understand license that places very few restrictions on reuse, making it a popular choice for open source projects.

By using this project, you agree to include the original copyright notice and permission notice in any copies or substantial portions of the software.
