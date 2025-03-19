import pandas as pd  

# Extract data
def extract_data(file_path):
    """Reads a CSV file and returns a DataFrame."""
    data = pd.read_csv(file_path)
    return data         

def transform_data(df):
    """Clean and prepare the data."""
    
    print("Before transformation:", df.columns)
    
    df['date'] = pd.to_datetime(df['date'])
    # Rename columns to be detected by Prophet
    # Prophet expects ds and y
    df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
    
    print("After transformation:", df.columns)
    
    return df

def load_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Clean Data saved to {output_file}")


if __name__ == '__main__':
    file_path = 'ecommerce_data.csv'
    output_file = 'clean_ecommerce_data.csv'
    df = extract_data(file_path)  # Extract data
    clean_df = transform_data(df)  # Transform data
    load_data(clean_df, output_file)
