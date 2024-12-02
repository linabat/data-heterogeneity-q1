import sys
import json
import os

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from etl import (
    retrieve_adult_data,
    gmm_adults,
    retrieve_covid_data,
    gmm_covid,
    plot_pca_gmm_covid,
    kmeans_adults,
    retrieve_features,
    run_waterbirds, 
    run_census, 
    
)
    
    
if __name__ == '__main__':
    args = sys.argv[1:]
    if 'census_model' in args: 
        print("census")

    if 'waterbirds_features' in args:
        with open("config/waterbirds_extract_features.json", "r") as file:
            config = json.load(file)
        retrieve_features(**config)

    if 'waterbirds_model' in args:
        print("wb")


    # COME BACK AND UPDATE
    if 'gmm_adults' in args: 
        with open('config/data-params.json') as fh:
            data_params = json.load(fh)
        
        gmm_adults(data_params["gmm_adult_ts"])

    # COME BACK AND UPDATE
    if 'gmm_covid' in args:
        with open('config/data-params.json') as fh:
            data_params = json.load(fh)
        
        gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"]
    )
        
    # COME BACK AND UPDATE
    if 'plot_gmm_covid' in args: 
        with open('config/data-params.json') as fh:
            data_params = json.load(fh)
        plot_pca_gmm_covid()

    # COME BACK AND UPDATE
    if 'kmeans_adults' in args:
        with open('config/data-params.json') as fh:
            data_params = json.load(fh)
        kmeans_adults()
