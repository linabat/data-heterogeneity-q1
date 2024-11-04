import sys
import json

from etl import (
    retrieve_adult_data,
    gmm_adults,
    retrieve_covid_data,
    gmm_covid,
    plot_pca_gmm_covid
)

def main(): 
    # Load data parameters from JSON configuration file
    with open('etl/data-params.json') as fh:
        data_params = json.load(fh)

    # Run Gaussian Mixture Model functions with specified parameters
    gmm_adults(data_params["gmm_adult_ts"])
    gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"]
    )

    # Plot the results of the GMM on COVID data
    plot_pca_gmm_covid()
    
if __name__ == '__main__':
    main()
