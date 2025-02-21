import pandas as pd
from utils.bigquery_utils import write_data_to_bigquery

if __name__ == "__main__":
    # Fetch data from folder
    manga_data = pd.read_csv('./data/processed/manga_data.csv')
    user_statistics = pd.read_csv('./data/processed/user_statistics.csv')
    user_manga_list = pd.read_csv('./data/processed/user_manga_list.csv')
    
    # Upload to BigQuery
    write_data_to_bigquery(dataset_id='manga_dataset', table_id='manga_collection_raw', df=manga_data)
    write_data_to_bigquery(dataset_id='users_dataset', table_id='user_statistics_raw', df=user_statistics)
    write_data_to_bigquery(dataset_id='users_dataset', table_id='user_manga_list_raw', df=user_manga_list)
