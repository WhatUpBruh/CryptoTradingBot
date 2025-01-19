import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta

# ticker = yf.Ticker("BTC-USD")
# data = ticker.history(period="6mo", interval="1h")


def calculate_sma(data, period):
    delta = data['Close']
    SMA = delta.rolling(window=period).mean()
    return SMA

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    # Calculate the multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate the initial EMA (SMA of the first 'period' prices)
    ema = data['Close'].ewm(alpha=multiplier).mean()
    return ema

def macd_line(data):
    EMA_12 = calculate_ema(data, 12)
    EMA_26 = calculate_ema(data, 26)
    MACD_line = EMA_12.sub(EMA_26)
    return MACD_line

def signal_line(data):
    MACD_line = macd_line(data)
    multiplier = 2 / (9 + 1)
    signal_line = MACD_line.ewm(alpha=multiplier).mean()
    return signal_line

def calculate_macd_hist(data):
    MACD_line = macd_line(data)
    Signal_line = signal_line(data)
    MACD_hist = MACD_line.sub(Signal_line)

    return MACD_hist

def calculate_bollinger_bands(data):
    # Calculate the SMA and the standard deviation
    SMA = calculate_sma(data, 10)
    std = data['Close'].rolling(window=20).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = SMA + 2 * std
    lower_band = SMA - 2 * std

    return upper_band, lower_band

def calculate_stochastic_oscillator(data):
    # Calculate the lowest and highest price in the last 14 days
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()

    # Calculate the %K and %D
    k = 100 * (data['Close'] - low_14) / (high_14 - low_14)
    d = k.rolling(window=3).mean()

    return k, d

def calculate_on_balance_volume(data):
    # Calculate the On Balance Volume (OBV)
    obv = data['Volume'].copy()
    obv[data['Close'] > data['Close'].shift(1)] = data['Volume']
    obv[data['Close'] < data['Close'].shift(1)] = -data['Volume']
    obv = obv.cumsum()

    return obv

# Calculates the trend of the stock
def trend_analysis(df):
    # Determine trend using ADX
    df['Trend_ADX'] = 'Sideways'  # Default
    # trend_adx = pd.DataFrame()
    # trend_adx = 'Sideways'
    # trend_adx.loc[(df['ADX'] > 20) & (df['DI_positive'] > df['DI_negative']), 'Trend_ADX'] = 'Uptrend'
    # trend_adx.loc[(df['ADX'] > 20) & (df['DI_positive'] < df['DI_negative']), 'Trend_ADX'] = 'Downtrend'    
    df.loc[(df['ADX'] > 20) & (df['DI_positive'] > df['DI_negative']), 'Trend_ADX'] = 'Uptrend'
    df.loc[(df['ADX'] > 20) & (df['DI_positive'] < df['DI_negative']), 'Trend_ADX'] = 'Downtrend'

    # Determine trend using SMA
    df['SMA_trend'] = 'Sideways'  # Default
    df.loc[df['SMA_10'] > df['SMA_50'], 'SMA_trend'] = 'Uptrend'
    df.loc[df['SMA_10'] < df['SMA_50'], 'SMA_trend'] = 'Downtrend'
    # sma_trend = pd.DataFrame()
    # sma_trend = 'Sideways'
    # sma_trend.loc[df['SMA_10'] > df['SMA_50'], 'SMA_trend'] = 'Uptrend'
    # sma_trend.loc[df['SMA_10'] < df['SMA_50'], 'SMA_trend'] = 'Downtrend'

    #Determining trend using ROC
    df['ROC_trend'] = 'Sideways'  # Default
    df.loc[df['ROC'] > 0, 'ROC_trend'] = 'Uptrend'
    df.loc[df['ROC'] < 0, 'ROC_trend'] = 'Downtrend'

    # If the trend is sideways, it sets the column sideways to 1 and the rest(uptrend and downtrend) to 0
    # Predicting the average trend
    df['Average_trend'] = 'Sideways'  # Default 
    df.loc[((df['Trend_ADX'] == "Uptrend") & (df['SMA_trend'] == "Uptrend")) | ((df['Trend_ADX'] == "Uptrend") & (df['ROC_trend'] == "Uptrend")) | ((df['SMA_trend'] == "Uptrend") & (df['ROC_trend'] == "Uptrend")), 'Average_trend'] = 'Uptrend'
    df.loc[((df['Trend_ADX'] == "Downtrend") & (df['SMA_trend'] == "Downtrend")) | ((df['Trend_ADX'] == "Downtrend") & (df['ROC_trend'] == "Downtrend")) | ((df['SMA_trend'] == "Downtrend") & (df['ROC_trend'] == "Downtrend")), 'Average_trend'] = 'Downtrend'


    # del df['Trend_ADX']
    # del df['SMA_trend']
    # del df['ROC_trend'] 
    # del df['Average_trend']

# Makes the testing/prediction data for the model
def trainingPrep(Stock_ticker):

    # Getting the stock data
    ticker = yf.Ticker(Stock_ticker)
    df = ticker.history(start="2024-12-04", end="2025-01-06", interval="1h")

    #Changing the price data to a 1 dimensional array
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()

    #Calculating the indicators for trend analysis
    df['ADX'] = ta.trend.adx(high, low, close, window=14) #should be greater than 20
    df['DI_positive'] = ta.trend.adx_pos(high, low, close, window=14) #should be greater than DI_negative for positive trend and vice versa
    df['DI_negative'] = ta.trend.adx_neg(high, low, close, window=14)
    df['SMA_10'] = calculate_sma(df, 10) #should be greater than SMA_50 for positive trend and vice versa
    df['SMA_50'] = calculate_sma(df, 50)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100 # >0 indicates positive trend and vice versa
    df['SMA'] = calculate_sma(df, 10)
    df['RSI'] = calculate_rsi(df)
    df['MACD_line'] = macd_line(df)
    df['Signal_line'] = signal_line(df)
    df['MACD_hist'] = calculate_macd_hist(df)
    df['K'], df['D'] = calculate_stochastic_oscillator(df)
    df['OBV'] = calculate_on_balance_volume(df)


    # Determine trend using ADX
    df['Trend_ADX'] = 'Sideways'  # Default
    df.loc[(df['ADX'] > 20) & (df['DI_positive'] > df['DI_negative']), 'Trend_ADX'] = 'Uptrend'
    df.loc[(df['ADX'] > 20) & (df['DI_positive'] < df['DI_negative']), 'Trend_ADX'] = 'Downtrend'

    # Determine trend using SMA
    df['SMA_trend'] = 'Sideways'  # Default
    df.loc[df['SMA_10'] > df['SMA_50'], 'SMA_trend'] = 'Uptrend'
    df.loc[df['SMA_10'] < df['SMA_50'], 'SMA_trend'] = 'Downtrend'

    #Determining trend using ROC
    df['ROC_trend'] = 'Sideways'  # Default
    df.loc[df['ROC'] > 0, 'ROC_trend'] = 'Uptrend'
    df.loc[df['ROC'] < 0, 'ROC_trend'] = 'Downtrend'

    # If the trend is sideways, it sets the column sideways to 1 and the rest(uptrend and downtrend) to 0
    # Predicting the average trend
    df[['Average_trend', 'Sideways', 'Uptrend', 'Downtrend']] = ['Sideways', 1, 0, 0]  # Default 
    df.loc[((df['Trend_ADX'] == "Uptrend") & (df['SMA_trend'] == "Uptrend")) | ((df['Trend_ADX'] == "Uptrend") & (df['ROC_trend'] == "Uptrend")) | ((df['SMA_trend'] == "Uptrend") & (df['ROC_trend'] == "Uptrend")), ['Average_trend', 'Sideways', 'Uptrend', 'Downtrend']] = ['Uptrend', '0', '1', '0']
    df.loc[((df['Trend_ADX'] == "Downtrend") & (df['SMA_trend'] == "Downtrend")) | ((df['Trend_ADX'] == "Downtrend") & (df['ROC_trend'] == "Downtrend")) | ((df['SMA_trend'] == "Downtrend") & (df['ROC_trend'] == "Downtrend")), ['Average_trend', 'Sideways', 'Uptrend', 'Downtrend']] = ['Downtrend', '0', '0', '1']
    
    df['Sideways'] = df['Sideways'].shift(-1)  # Shift price to create future target
    df['Uptrend'] = df['Uptrend'].shift(-1)  # Shift price to create future target
    df['Downtrend'] = df['Downtrend'].shift(-1)  # Shift price to create future target


    # Removing unwanted columns
    del df['Dividends']
    del df['Stock Splits']
    del df['Trend_ADX']
    del df['SMA_trend']
    del df['ROC_trend'] 
    del df['Average_trend']
    
    # Removing first 51 rows to avoid NaN values from SMA 50
    df = df.drop(df.index[1:73])
    df  = df.iloc[:-2]
    
    # Converting to CSV filetype
    df.to_csv('training_Data.csv')

    print("Training data has been prepared")

# Makes the training data for the model
def testingPrep(Stock_ticker):

    # Getting the stock data
    ticker = yf.Ticker(Stock_ticker)
    df = ticker.history(start="2024-09-04", end="2024-10-10", interval="1h")

    #Changing the price data to a 1 dimensional array
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()

    #Calculating the indicators for trend analysis
    df['ADX'] = ta.trend.adx(high, low, close, window=14) #should be greater than 20
    df['DI_positive'] = ta.trend.adx_pos(high, low, close, window=14) #should be greater than DI_negative for positive trend and vice versa
    df['DI_negative'] = ta.trend.adx_neg(high, low, close, window=14)
    df['SMA_10'] = calculate_sma(df, 10) #should be greater than SMA_50 for positive trend and vice versa
    df['SMA_50'] = calculate_sma(df, 50)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100 # >0 indicates positive trend and vice versa
    df['SMA'] = calculate_sma(df, 10)
    df['RSI'] = calculate_rsi(df)
    df['MACD_line'] = macd_line(df)
    df['Signal_line'] = signal_line(df)
    df['MACD_hist'] = calculate_macd_hist(df)
    df['K'], df['D'] = calculate_stochastic_oscillator(df)
    df['OBV'] = calculate_on_balance_volume(df)
    df['Sideways'] = np.nan
    df['Uptrend'] = np.nan
    df['Downtrend'] = np.nan

    # Deleting unwanted columns
    del df['Dividends']
    del df['Stock Splits']

    # Removing first 51 rows to avoid NaN values from SMA 50
    df = df.drop(df.index[1:73])
    df  = df.iloc[:-2]

    # Converting to CSV filetype
    df.to_csv('prediction_Data.csv')

    print("Testing data has been prepared")








trainingPrep("BTC-USD")
testingPrep("BTC-USD")

# "BTC-USD"
# data = pd.DataFrame()
# data = trend_analysis()