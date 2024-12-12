from pathlib import Path
import gzip
import shutil

# Define the directory containing the .gz files
directory = Path('./Human_features/raw')

# Loop through all .gz files in the directory
for file in directory.glob('*.gz'):
    output_filepath = file.with_suffix('')  # Remove the .gz extension
    
    try:
        # Open the .gz file and decompress it
        with gzip.open(file, 'rb') as f_in:
            with open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Delete the original .gz file
        file.unlink()  
        
    except Exception as e:
        print(f"Error processing {file.name}: {e}")