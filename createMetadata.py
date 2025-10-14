import os
import pandas as pd

def create_metadata(directories, output_csv):
    metadata = []  # List to store metadata rows

    for directory in directories:
        if os.path.isdir(directory):
            folder_name = os.path.basename(directory)  # Get the folder name (dx)
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):  # Only process files
                    metadata.append({"image_id": file, "dx": folder_name})
        else:
            print(f"Directory '{directory}' does not exist.")

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(metadata)
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

# Example usage
directories = [
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED/FINALactinickeratosis', 
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED/FINALbenignkeratosis', 
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED/FINALdermatofibroma', 
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED/FINALmelanoma',
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED/FINALvascularlesion',
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET/FINALnevus',
    '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET/FINALbasalcell'  # Add as many directories as needed
]
output_csv = '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED/ROTATEDmetadata.csv'
create_metadata(directories, output_csv)
