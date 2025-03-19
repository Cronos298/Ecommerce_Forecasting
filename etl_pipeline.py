import pandas as pd  # Import pandas for data handling

# Extract data
def extract_data(file_path):
    """Reads a CSV file and returns a DataFrame."""
    data = pd.read_csv(file_path)
    return data         

def transform_data(df):
    """Clean and prepare the data."""
    
    print("Before transformation:", df.columns)  # Print original column names
    
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    # Remove duplicate rows, if any
    df.drop_duplicates(inplace=True)
    # Fill any missing values (if any) with the previous valid value
    df.ffill(inplace=True)
    # Rename columns to fit Prophet's expected names:
    # Prophet expects 'ds' for date and 'y' for the target value (sales)
    df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
    
    print("After transformation:", df.columns)  # Print transformed column names
    
    return df

def load_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Clean Data saved to {output_file}")


# Test the transform function
if __name__ == '__main__':
    file_path = 'ecommerce_data.csv'
    output_file = 'clean_ecommerce_data.csv'
    df = extract_data(file_path)  # Extract data
    clean_df = transform_data(df)  # Transform data
    load_data(clean_df, output_file)  # Display the first few rows
