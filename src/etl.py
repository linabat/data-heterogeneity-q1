from ucimlrepo import fetch_ucirepo 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.mixture import GaussianMixture 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

### Retrieving Datasets 
def retrieve_adult_data():
    """
    This function is used to retrieve UCI's adult dataset
    """
    # Fetch dataset 
    adult = fetch_ucirepo(id=2) 

    # Data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
    
    # Cleaning target values
    y.replace("<=50K.", "<=50K", inplace=True)
    y.replace(">50K.", ">50K", inplace=True)
    
    # Group the original dataset for how it was 
    full_data = pd.concat([X,y], axis = 1)

    # Breaking up the groups so can  do undersampling for greather than group
    less_than = full_data[full_data["income"]=="<=50K"]
    greater_than = full_data[full_data["income"]==">50K"]

    # Conducting Undersampling Here 
    greater_than_count = greater_than.shape[0]
    less_than_under = less_than.sample(greater_than_count)


    under_sampled_data = pd.concat([greater_than, less_than_under], axis=0)
    under_sampled_data["lower_income_bool"] = under_sampled_data["income"] == "<=50K"

    y = under_sampled_data["lower_income_bool"]
    X = under_sampled_data.drop(columns=["income", "lower_income_bool"])
    
    return X, y

def retrieve_covid_data(covid_fp, replace_num): 
    """
    This function is used to retrieve the covid dataset
    """
    covid = pd.read_csv(covid_fp)
    # Cleaning up column names
    covid.columns = covid.columns.str.strip().str.lower()

    # Creating boolean column which is what will be predicted
    covid["died_bool"] = covid["date_died"] != "9999-99-99"

    # Replacing all 98 values with 97 so there is only one number that indicates whethe
    # the value is missing
    covid.replace(replace_num, 97, inplace=True)
    
    covid.drop(columns=["clasiffication_final", "date_died"], inplace=True)
    
    return covid 

### GMM
def gmm_adults(gmm_adult_ts): 
    """
    This function prints out a classification report for the Gaussian Mixture Model that
    is used to identify 2 clusters to predict whether someone will have an income greater than 
    or less than 50,000
    """
    # Retrieving data for model 
    X,y = retrieve_adult_data()
    X = pd.get_dummies(X, drop_first=True)
    
    # standardizing features 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = gmm_adult_ts)

    gmm = GaussianMixture(n_components = 2)

    gmm.fit(X_train)

    y_pred = gmm.predict(X_test)

    mapped_y_pred = [0 if label == y_test.mode()[0] else 1 for label in y_pred]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return 

def gmm_covid(covid_fp, replace_num, gmm_covid_ts): 
    """
    This function outputs a classification report for the the Gaussian Mixture model for 
    covid dataset - am only looking at it's ability to identify 2 groups. 
    """
    covid = retrieve_covid_data(covid_fp, replace_num)

    class_0 = covid[covid["died_bool"] == False]
    class_1 = covid[covid["died_bool"] == True]
    class_1_count = class_1.shape[0]

    class_0_under = class_0.sample(class_1_count)

    # Equal numbers of died and not died in this datasets
    covid_under = pd.concat([class_0_under, class_1], axis=0)

    # Separate the target variable
    y = covid_under["died_bool"]
    X = covid_under.drop(columns=["died_bool"])

    # Convert categorical variables to dummy/indicator variables
    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified train-test split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=gmm_covid_ts, random_state=42
    )

    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_train)

    # Predict on test data
    y_pred = gmm.predict(X_test)

    # Map predictions to 0 or 1 based on the most common label in y_test
    mapped_y_pred = [0 if label == y_test[0] else 1 for label in y_pred]

    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, mapped_y_pred))

    print("Classification Report:\n", classification_report(y_test, mapped_y_pred))
    
    return X_scaled

def plot_pca_gmm_covid():
    """
    This function is used to plot compare the two group that the GMM identifies 
    to the 2 original groups.
    """
    X_scaled = gmm_covid()
    # Perform PCA to reduce the dataset to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Predict clusters using GMM
    y_cluster = gmm.predict(X_scaled)

    # Create a DataFrame with PCA results, GMM clusters, and original class labels
    pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    pca_df['GMM Cluster'] = y_cluster
    pca_df['Original Class'] = y  # Assuming `y_sample` is the original target label

    # Plot side-by-side comparison of GMM clusters and original classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot GMM Clusters
    sns.scatterplot(x='PCA1', y='PCA2', hue='GMM Cluster', data=pca_df, palette='Set1', ax=ax1, alpha=0.7)
    ax1.set_title('GMM Clusters')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.legend(title='GMM Cluster')

    # Plot Original Classes
    sns.scatterplot(x='PCA1', y='PCA2', hue='Original Class', data=pca_df, palette='Set2', ax=ax2, alpha=0.7)
    ax2.set_title('Original Classes')
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    ax2.legend(title='Original Class')

    plt.tight_layout()
    plt.show()
    
    return 


### KMeans
def kmeans_adults(): 
    X, y = retrieve_adult_data()
    data = pd.concat([X,y], axis=1)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].apply(LabelEncoder().fit_transform)
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=0)
    data['cluster'] = kmeans.fit_predict(data)
    
    score = silhouette_score(data[numeric_cols], data['cluster'])
    print(f'Silhouette Score for {k} clusters: {score}')
    
    # If you need dimensionality reduction (for datasets with >2 features)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(data)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['cluster'], palette='viridis', s=50)

    # Mark cluster centers
    centers = kmeans.cluster_centers_
    centers_reduced = pca.transform(centers)  # Reduce dimensions for plotting
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', s=200, alpha=0.6, label='Centers')
    plt.title("K-means Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    