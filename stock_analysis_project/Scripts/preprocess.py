import pandas as pd

def load_and_preprocess(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Convert 'Date' column to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    return data

if __name__ == "__main__":
    # Call the function with the path to the dataset
    data = load_and_preprocess('data/DELL_daily_data.csv')
    print(data.head())
