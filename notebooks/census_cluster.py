from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage
import pandas as pd
import numpy as np
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

def lr_schedule(epoch):
    if epoch < 20:
        return 1e-4
    else:
        return 1e-5


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



def get_model_z(X,s,n_z,model_name,epochs=20,verbose=1,var_reg=0.0):
    """
    Defines and trains a clustering model. 
    """
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    
    #Shuffle X and s together
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
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
        # changing this from True to False
        save_best_only=False,
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

def pzxs(X,s,best_model,arg_max=True):
    """
    Compute conditional probabilities - probability of cluster z given x and additional
    informations
    """
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    psz_matrix = best_model.get_weights()[-1]
    p2 = p*psz_matrix[:,s.astype(int)].T
    p2 = p2/np.reshape(np.sum(p2,axis=1),(np.sum(p2,axis=1).shape[0],1))

    if(arg_max):
        p2 = np.argmax(p2,axis=1)
    return p2


def main():
    """Main function to run the entire process."""
    # Set random seed for reproducibility
    seed = 0
    print(f"Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load data
    features, labels, states_from_df = load_census_data(ACSIncome)  # will need to change this based off of source we're looking at 
    s1 = to_categorical(labels)
    

    best_model = get_model_z(features, s1, 4, 'best_census_model_inc.h5', epochs=40, var_reg=0)

    # for function pzx
    p1_fl = pzx(features, best_model, arg_max=False)
    p1_tr = pzx(features, best_model, arg_max=True)

    cluster_state_df = pd.DataFrame({
        "p1_tr":p1_tr,
        "states":states_from_df, 
        "type": "public_coverage"
        }
        )

    output_csv = "cluster_state_data.csv"
    cluster_state_df.to_csv(output_csv, index=False)
    

    try:
        np.save("p1_fl_pzx_predictions.npy", p1_fl)
        print("Cluster assignments saved to 'p1_fl_pzx_predictions.npy'")
    except Exception as e:
        print(f"Error saving predictions: {e}")

    try:
        np.save("p1_tr_pzx_predictions.npy", p1_tr)
        print("Cluster assignments saved to 'p1_tr_pzx_predictions.npy'")
    except Exception as e:
        print(f"Error saving predictions: {e}")

    
if __name__ == "__main__":
    main()
