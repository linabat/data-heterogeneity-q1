# data-heterogeneity-q1

The focus of this quarter's work has been looking at clustering methods for latent variables. We started off by looking looking at the common clustering methods, K-Means Clustering and Gaussian Mixture Models, on the UCI Adult dataset and Kaggle's COVID-19 dataset. From there, we implemented a more complex model that uses a neural network to cluster the data we had to find latent, unknown, variables. We applied this algorithm to 4 datasets - 3 tabular and 1 imagedataset. Note: in order to run this datasets in a timely manner, you will likely need a GPU as their are quite large. 

Once you've cloned this repo, you will be able to acheive the same results as us. 
But first, you must install the dependencies. To install the dependencies, run the following command from the root directory: pip install -r requirements.txt

Let's start by looking at the results for the waterbirds dataset. This dataset comes from [this repo](https://github.com/kohpangwei/group_DRO). Below are the steps to get this code working for this dataset and retrieve the same results as we did
1. To download the waterbirds dataset, [click this link]. Place the .tar.gz file in the cloned repo and save the path to this file.
2. Go into the `config` folder and past the relative path for where you saved the .tar.gz file into your repo - will be the value of the key`tar_file_path` in the json value.
3. In your terminal, in the repo directory run `python run.py download_wb_data` -> This will create a folder in your repo called `waterbirds_data`, where all the image data for the waterbirds dataset is stored
4. To retrieve the features need in order to run the model, in terminal, run `python run.py waterbirds_features` -> This will create a folder `waterbirds_features` that will store s `features.npy` file.
5. Now that everything has been processed for the 




Building the project stages using run.py.
    - To run the Gaussian Model Mixture model on the adults dataset, run python run.py gmm_adults
     - To run the Gaussian Model Mixture model on the covid dataset, run python run.py gmm_covid
     - To plot the Gaussian Model Mixture model clusters and original clusters on the covid dataset, run python run.py plot_gmm_covid
     - To run the K-Means clustering and visualize the clusters on the adults dataset, run python kmeans_adults