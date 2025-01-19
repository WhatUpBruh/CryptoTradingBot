import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

dat = yf.Ticker("MSFT")



# Historic price of that stock, RSI, SMA

def calculate_sma(data, period=10):
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
    SMA = calculate_sma(data)
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

def calculate_average_directional_index(data):
    # Calculate the True Range (TR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    TR = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate the Directional Movement (DM)
    plus_DM = data['High'] - data['High'].shift(1)
    plus_DM[plus_DM < 0] = 0
    minus_DM = data['Low'].shift(1) - data['Low']
    minus_DM[minus_DM < 0] = 0

    # Calculate the 14-day ATR and the 14-day +DI and -DI
    ATR = TR.rolling(window=14).mean()
    plus_DI = 100 * (plus_DM.rolling(window=14).mean() / ATR)
    minus_DI = 100 * (minus_DM.rolling(window=14).mean() / ATR)

    # Calculate the Average Directional Index (ADX)
    DX = 100 * (plus_DI - minus_DI).abs() / (plus_DI + minus_DI)
    ADX = DX.rolling(window=14).mean()

    return plus_DI, minus_DI, ADX



def plot_chart(data):
    data['RSI'] = calculate_rsi(data)
    data['SMA'] = calculate_sma(data)
    data['EMA'] = calculate_ema(data, 20)
    data['MACD'] = macd_line(data)
    data['MACD_Signal'] = signal_line(data)
    data['MACD_HIST'] = calculate_macd_hist(data)
    data['K'] = K
    data['D'] = D
    data['OBV'] = calculate_on_balance_volume(data)
    data['Plus_DI'] = Plus_DI
    data['Minus_DI'] = Minus_DI
    data['ADX'] = ADX
    # data['Plus_DI', 'Minus_DI', 'ADX'] = calculate_average_directional_index(data)
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    print(data[['RSI', 'SMA', 'EMA', 'MACD', 'Upper_Band', 'Lower_Band']])


    # Plotting the charts in the same window
    plt.figure(figsize=(20, 20))
    # First subplot (Close Prices, RSI, SMA, EMA)
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Prices')
    plt.plot(data['SMA'], label='SMA', linestyle=':')
    plt.plot(data['EMA'], label='20-day EMA')
    plt.plot(data['Upper_Band'], label='Upper Bollinger Band')
    plt.plot(data['Lower_Band'], label='Lower Bollinger Band')
    plt.title('Stock Prices and Indicators')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    # Second subplot (MACD, MACD Signal, MACD Histogram)
    plt.subplot(3, 1, 2)
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['MACD_Signal'], label='MACD Signal', color='red')
    plt.bar(data.index, data['MACD_HIST'], label='MACD Histogram', color='gray')
    plt.title('MACD and MACD Histogram')
    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.legend()

    # Third subplot (RSI)
    plt.subplot(3, 1, 3)
    plt.plot(data['RSI'], label='RSI', linestyle='--')
    plt.axhline(y=30, color='r', label='RSI 30')
    plt.axhline(y=70, color='g', label='RSI 70')
    plt.title('Stock RSI')



    plt.tight_layout()
    plt.show()


# Fetch live data for a stock (e.g., Apple Inc.)
ticker = yf.Ticker("BTC-USD")
data = ticker.history(start="2024-12-04", end="2025-01-01", interval="1h")

# Calculating resulting dataframes
K, D = calculate_stochastic_oscillator(data)
Plus_DI, Minus_DI, ADX = calculate_average_directional_index(data)


data['RSI'] = calculate_rsi(data)
data['SMA'] = calculate_sma(data)
data['EMA'] = calculate_ema(data, 20)
data['MACD'] = macd_line(data)
data['MACD_Signal'] = signal_line(data)
data['MACD_HIST'] = calculate_macd_hist(data)
data['K'] = K
data['D'] = D
data['OBV'] = calculate_on_balance_volume(data)
data['Plus_DI'] = Plus_DI
data['Minus_DI'] = Minus_DI
data['ADX'] = ADX
# data['Plus_DI', 'Minus_DI', 'ADX'] = calculate_average_directional_index(data)
data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
print(data[['RSI', 'SMA', 'EMA', 'MACD', 'Upper_Band', 'Lower_Band']])



# Plotting the Chart: 
plot_chart(data)




# Calculate RSI
# data['RSI'] = calculate_rsi(data)
# data['SMA'] = calculate_sma(data)
# data['EMA'] = calculate_ema(data, 20)
# data['MCAD'] = calculate_mcad(data)
# print(data[['RSI', 'SMA', 'EMA', 'MCAD']])




# Print the historical price data




# Information about stock's current price

        # current = dat.analyst_price_targets.get("current")
        # low = dat.analyst_price_targets.get("low")
        # high = dat.analyst_price_targets.get("high")
        # mean = dat.analyst_price_targets.get("mean")
        # median = dat.analyst_price_targets.get("median")

        # print(current)
        # print(low)
        # print(high)
        # print(mean)
        # print(median)


# Obtaining the data in CVS format



