def add_features(data):
    # Create moving averages and daily returns
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Return'] = data['Close'].pct_change()

    # Drop rows with missing values
    data.dropna(inplace=True)

    return data

if __name__ == "__main__":
    from preprocess import load_and_preprocess
    data = load_and_preprocess('data/DELL_daily_data.csv')
    data = add_features(data)
    print(data.head())
