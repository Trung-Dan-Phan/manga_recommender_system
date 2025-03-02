import os

import pandas as pd
from loguru import logger

# Folder where batch files are stored
BATCH_FOLDER = "./data/raw"  # Update this path

# Output folder where the final file will be saved
OUTPUT_FOLDER = "./data/processed"
OUTPUT_FILENAME = "user_manga_list.csv"


def concatenate_csv_batches(
    batch_folder: str, batch_start_name: str, output_folder: str, output_filename: str
):
    """
    Concatenates multiple CSV batch files into a single CSV file
    and saves it in a specified directory.

    Args:
        batch_folder (str): Path to the folder containing batch CSV files.
        batch_start_name (str): Prefix of the batch file names.
        output_folder (str): Path to the folder where the final concatenated file will be saved.
        output_filename (str): Name of the final CSV file.

    Returns:
        None
    """
    # Get all batch CSV files with the specified prefix and ending
    csv_files = sorted(
        [
            f
            for f in os.listdir(batch_folder)
            if f.startswith(batch_start_name) and f.endswith(".csv")
        ]
    )

    if not csv_files:
        logger.warning("No batch files found in the folder.")
        return

    logger.info(f"Found {len(csv_files)} batch files. Starting concatenation...")

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(batch_folder, file)
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            logger.info(f"Loaded {file} with {len(df)} rows.")
        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")

    # Concatenate all dataframes
    final_df = pd.concat(dataframes, ignore_index=True)

    # Drop potential duplicates
    final_df.drop_duplicates(inplace=True)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save final concatenated CSV
    output_path = os.path.join(output_folder, output_filename)
    final_df.to_csv(output_path, index=False)

    logger.info(
        f"Concatenation complete! Final file saved at {output_path} with {len(final_df)} rows."
    )


if __name__ == "__main__":
    batch_start_name = "user_manga_list"
    concatenate_csv_batches(
        batch_folder=BATCH_FOLDER,
        batch_start_name=batch_start_name,
        output_folder=OUTPUT_FOLDER,
        output_filename=OUTPUT_FILENAME,
    )
