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

from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, 
    MaxPooling2D, BatchNormalization, Dropout
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam
from keras.initializers import Constant

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from tensorflow.keras.regularizers import Regularizerfrom folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomUniform
import tensorflow as tf
import random

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image

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


# Image Pre-Processing 
# ===========================
# Feature Extraction
# ===========================
def create_feature_extractor():
    """
    Model that extracts features from images using a pre-trained ResNet50
    """
    base_model = ResNet50(weights='imagenet', include_top=False)
    extractor = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    return extractor

def process_images_in_batches(dataset_path, metadata_df, batch_size=32):
    """
    Process images in batches to extract features while minimizing memory usage
    """
    extractor = create_feature_extractor()
    total_images = len(metadata_df)
    features = np.zeros((total_images, 2048)) # Resizing images to 224x224 pixels to match 
    # ResNet50's input size
    
    for i in tqdm(range(0, total_images, batch_size)):
        batch_files = metadata_df['img_filename'].iloc[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_files:
            img = image.load_img(os.path.join(dataset_path, img_path), 
                               target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            batch_images.append(img_array)
        
        batch_images = np.vstack(batch_images)
        batch_images = preprocess_input(batch_images)
        
        batch_features = extractor.predict(batch_images, verbose=0)
        features[i:i+len(batch_files)] = batch_features
        
        del batch_images
        del batch_features
    
    return features # a 2D numpy array where each row represents the extracted features for an image
    

# Census Pre-Processing

def load_census_data(data_processor):
    """
    Load and process census data.
    @param data_processor: ACSIncome, ACSEmployment, ACSPublicCoverage 
    """
    states = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA',
        'MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NH','NJ','NM','NY','NV','OH','OK','OR','PA','PR',
        'RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
    
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    all_data = pd.DataFrame()

    for state in states: 
        try:
            state_data = data_source.get_data(states=[state], download=True)     
            state_features, _, _ = data_processor.df_to_numpy(state_data)
            state_features_df = pd.DataFrame(state_features)
            state_features_df['ST'] = state
            all_data = pd.concat([all_data, state_features_df], ignore_index=True)
        except Exception as e: 
            continue

    # Encode state labels
    label_encoder = LabelEncoder()
    states_from_df = all_data["ST"].to_numpy()
    all_data['ST_encoded'] = label_encoder.fit_transform(all_data['ST'])
    labels = all_data["ST_encoded"].to_numpy()
    features = all_data.drop(columns=["ST", "ST_encoded"]).to_numpy()

    return features, labels, states_from_df



# Clustering method - Parjanya's code 

# ===========================
# Helper Classes
# ===========================
class ClipConstraint(Constraint):
    """
    Clips model weights to a given range [min_value, max_value].
    Normalizes weights along a specific axis to exurethey sum to1
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, weights):
        w = tf.clip_by_value(weights, self.min_value, self.max_value)
        return w / tf.reduce_sum(w, axis=1, keepdims=True)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class VarianceRegularizer(Regularizer):
    """
    Custom regularizer for maximum weight variance.
    Purpose: encourage uniformity among weights -- improve generalization or stability
    """
    
    def __init__(self, factor=0.01):
        self.factor = factor
    
    def __call__(self, x):
        variances = tf.math.reduce_variance(x, axis=1)
        max_variance = tf.reduce_max(variances)
        return self.factor * max_variance
    
    def get_config(self):
        return {'factor': self.factor}

# ===========================
# Model Definition and Training
# ===========================
def lr_schedule(epoch):
    """Defines the learning rate schedule."""
    if epoch < 20:
        return 1e-4
    else:
        return 1e-5

def get_model_z(X,s,n_z,model_name,epochs=20,verbose=1,var_reg=0.0):
    """
    Defines and trains a clustering model. 
    """
    #Shuffle X and s together
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1) # added this in here isntead
    
    X = X[indices]
    s = s[indices]
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(s.shape[1], activation='linear'),
        Dense(n_z, activation='softmax', name='target_layer'),
        Dense(s.shape[1], activation='linear', use_bias=False,kernel_initializer=RandomUniform(minval=0, maxval=1),kernel_constraint=ClipConstraint(0, 1), kernel_regularizer=VarianceRegularizer(factor=var_reg)), 
    ])
    optimizer = Adam(learning_rate=1e-3) # an adaptive learning rate optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint_path = model_name

    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=verbose
    )
    print(model.get_weights()[-1])

    model.fit(
        X,
        s,
        batch_size=1024,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[model_checkpoint_callback, lr_scheduler]    
    )
    
    best_model = load_model(model_checkpoint_path, custom_objects={'ClipConstraint': ClipConstraint,'VarianceRegularizer': VarianceRegularizer})
    return best_model

def pzx(X,best_model,arg_max=True):
    """
    Predict cluster assignments
    """
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    if(arg_max):
        p = np.argmax(p,axis=1)
    return p

def set_seed(seed_num): 
    # Set random seed for reproducibility
    seed = seed_num
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def retrieve_features(dataset_path, features_path):
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)

    features = process_images_in_batches(dataset_path, metadata_df)
    np.save(features_path, features)

def visualization_images(p_value, y_value, dataset_path):
        """
        Visualizes a 2x2 grid of images based on specified filtering conditions.
        """
        selected_indices = np.random.choice(len(file_paths[(p==p_value)&(np.squeeze(y)==y_value)]), 4, replace=False)
        selected_file_paths = file_paths[(p==p_value)&(np.squeeze(y)==y_value)][selected_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        for ax, file_path in zip(axes.flatten(), selected_file_paths):
            img = Image.open('data/Waterbirds_dataset/waterbird_complete95_forest2water2/' + file_path)
            ax.imshow(img)
            ax.axis('off')  
        
        plt.tight_layout()
    
        # Save the plot as a PNG file
        plt.savefig(f"selected_images_{p_value}_{y_value}.png", dpi=300)
    

# ===========================
# Waterbirds Path
# ===========================
def run_waterbirds(dataset_path, features_path, output_csv_path, num_clusters=2, num_epochs=60, model_path='best_model_inc.h5', , num_var_reg=0, seed_num=0): 
    set_seed(seed_num)
    
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)
    s = metadata_df['split'].values 
    s=(s+1)//2

    features = np.load(features_path)

    # Make s1 as one hot of s
    s1 = to_categorical(s)
        
    best_model = get_model_z(features, s1, num_clusters, model_path, epochs=num_epochs, var_reg=num_var_reg)
    
    # for function pzx - want to save these each to a column 
    p1_fl = pzx(features, best_model, arg_max=False)
    p1_tr = pzx(features, best_model, arg_max=True)

    file_paths = metadata_df['img_filename'].values

    
    y = metadata_df['y']
    place = metadata_df['place'] # check if y and place match and output into a txt file


    p1_fl_df = pd.DataFrame(
        p1_fl,
        columns=['p1_fl_cluster_0', 'p1_fl_cluster_1']
    )

    p1_tr_df = pd.DataFrame(
        p1_tr,
        columns=['p1_tr']
    )

    # Combine everything into a DataFrame
    p_y_place_df = pd.concat(
        [
            p1_fl_df,
            p1_tr_df, 
            pd.DataFrame({
                "y": y,
                "place": place
            })
        ],
        axis=1
    )

    # Save the DataFrame to a CSV file
    output_csv = output_csv_path
    p_y_place_df.to_csv(output_csv, index=False)


    # Doing all the combination to see what it should be
    visualization_images(1, 0, dataset_path)
    visualization_images(1, 1, dataset_path)
    visualization_images(0, 1, dataset_path)
    visualization_images(0, 0, dataset_path)



# ===========================
# Census Data Set
# ===========================

    
def run_census(data_processor, dataset_path, features_path, output_csv_path, num_clusters=4, num_epochs=60, model_path='best_census_model_inc.h5', , num_var_reg=0, seed_num=0):
    
    set_seed(seed_num)
    features, labels, states_from_df = load_census_data(data_processor)
    s1 = to_categorical(labels)
    p1_tr = pzx(features, best_model, arg_max=True)

    processor_type = ""
    if data_processor == ACSIncome: 
        processor_type = "income"

    elif data_processor == ACSEmployment: 
        processor_type = "employment" 

    elif data_processor == ACSPublicCoverage: 
        processor_type = "public_coverage" 

    else: 
        processor_type = "not_identified" 

    cluster_state_df = pd.DataFrame({
        "p1_tr":p1_tr,
        "states":states_from_df, 
        "type": "public_coverage"
        }
        )

    output_csv = output_csv_path
    cluster_state_df.to_csv(output_csv, index=False)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    