import sys
import json

from etl import (
    retrieve_adult_data,
    gmm_adults,
    retrieve_covid_data,
    gmm_covid,
    plot_pca_gmm_covid,
    kmeans_adults
)

def main(targets): 
    # Load data parameters from JSON configuration file
    with open('etl/data-params.json') as fh:
        data_params = json.load(fh)
    
    if 'gmm_adults' in targets: 
        gmm_adults(data_params["gmm_adult_ts"])
    
    if 'gmm_covid' in targets: 
        gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"]
    )
    
    if 'plot_gmm_covid' in targets: 
        plot_pca_gmm_covid()
        
    if 'kmeans_adults' in targets:
        kmeans_adults()
    
    
if __name__ == '__main__':
    main()
