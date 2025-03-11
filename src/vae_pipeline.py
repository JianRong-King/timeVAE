import os, warnings
import numpy as np
import time
import os
import torch
import scipy.io as scio


files = {
    'MFP': ['Training.mat']
}

filename = "MFP"
file = files[filename]
filepath = "./Data/CVACaseStudy/"

if filename == "MFP":
    data = scio.loadmat(filepath + file[0])
    data_t1 = data["T1"]
    data_t2 = data["T2"]
    data_t3 = data["T3"]


# Split the data into training, validation and test sets: 0,6, 0.2, 0.2
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 1 - train_ratio - valid_ratio 

# full data

new_data_train_t1 = data_t1[:int(data_t1.shape[0] * train_ratio)] 
new_data_train_t2 = data_t2[:int(data_t2.shape[0] * train_ratio)]
new_data_train_t3 = data_t3[:int(data_t3.shape[0] * train_ratio)]

new_data_valid_t1 = data_t1[int(data_t1.shape[0] * train_ratio) : int(data_t1.shape[0] * (train_ratio+valid_ratio))] 
new_data_valid_t2 = data_t2[int(data_t2.shape[0] * train_ratio) : int(data_t2.shape[0] * (train_ratio+valid_ratio))]
new_data_valid_t3 = data_t3[int(data_t3.shape[0] * train_ratio) : int(data_t3.shape[0] * (train_ratio+valid_ratio))]

new_data_test_t1 = data_t1[int(data_t1.shape[0] * (train_ratio+valid_ratio)) : data_t1.shape[0]] 
new_data_test_t2 = data_t2[int(data_t2.shape[0] * (train_ratio+valid_ratio)) : data_t2.shape[0]]
new_data_test_t3 = data_t3[int(data_t3.shape[0] * (train_ratio+valid_ratio)) : data_t3.shape[0]]

# Combine the 3 datasets into each of the training, validation and testing sets
training_set = np.vstack((new_data_train_t1,new_data_train_t2,new_data_train_t3))
validation_set = np.vstack((new_data_valid_t1,new_data_valid_t2,new_data_valid_t3))
testing_set = np.vstack((new_data_test_t1,new_data_test_t2,new_data_test_t3))



idx_var_remove = list(set([-1])) # remove output_vars and last variable, ensure that both are different with set()
pressure_var = 5 # 4th index

X_train = np.delete(training_set, idx_var_remove, axis=1)
Y_train = training_set[:, pressure_var - 1]

X_valid = np.delete(validation_set, idx_var_remove, axis=1)
Y_valid = validation_set[:, pressure_var -1]

X_test = np.delete(testing_set, idx_var_remove, axis=1)
Y_test = testing_set[:, pressure_var -1]



def get_norm_param(X,Y):
    x = X
    y = Y

    keys = ['x_min','x_max','y_min','y_max','x_mean','x_std','y_mean','y_std']
    norm_param = {}
    for key in keys: norm_param[key] = []
    
    norm_param['x_min']  = np.min(x, axis=0)
    norm_param['x_max']  = np.max(x, axis=0)
    norm_param['y_min']  = np.min(y, axis=0)
    norm_param['y_max']  = np.max(y, axis=0)
    norm_param['x_mean'] = np.mean(x, axis=0)
    norm_param['x_std']  = np.std(x, axis=0)
    norm_param['y_mean'] = np.mean(y, axis=0)
    norm_param['y_std']  = np.std(y, axis=0)

    return norm_param

norm_param_train = get_norm_param(X_train,Y_train)

def normalize(X,Y,norm_param,method):
    x = X
    y = Y

    if method == 'minmax':
        X_norm = (x - norm_param['x_min']) / (norm_param['x_max'] - norm_param['x_min'])
        Y_norm = (y - norm_param['y_min']) / (norm_param['y_max'] - norm_param['y_min'])


    elif method == 'standardize':
        X_norm = (x - norm_param['x_mean']) / norm_param['x_std']
        Y_norm = (y - norm_param['y_mean']) / norm_param['y_std']

    else:
        raise TypeError("Normalization Method Not Known")

    return X_norm, Y_norm

norm_method = "minmax"
X_train_norm, Y_train_norm = normalize(X_train,Y_train,norm_param_train,norm_method)
X_valid_norm, Y_valid_norm = normalize(X_valid,Y_valid,norm_param_train,norm_method)
X_test_norm, Y_test_norm = normalize(X_test,Y_test,norm_param_train,norm_method)


import torch 
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self, X, Y, his_length, pred_length, pred_mode):
        self.x = X
        self.y = Y
        self.his_length = his_length 
        self.pred_length = pred_length
        self.mode = pred_mode

    def __getitem__(self,index):

        x = self.x[index : index + self.his_length]
        # print(x.shape)

        if self.mode == 'current_step':
            y = self.y[index + self.his_length - 1]
            y = torch.Tensor([y])

        elif self.mode == 'multi_step':
            y = self.y[index + self.his_length : index + self.his_length + self.pred_length]
            y = torch.Tensor(y)

        else:
            raise TypeError('Prediction Model is not Known')
        
        y = torch.ones_like(y)


        return torch.Tensor(x)#, y
    
    def __len__(self):
        if self.mode == 'current_step':
            # print(len(self.x) - self.his_length + 1)
            return len(self.x) - self.his_length + 1

        elif self.mode == 'multi_step':
            return len(self.x) - self.his_length - self.pred_length + 1
        
        else: 
            raise TypeError('Prediction Model is not Known')
        
        


pred_mode = 'current_step' # current_step/multi_step
his_length = 85 # L
pred_length = 1 

train_dataset = MyDataset(X_train_norm, Y_train_norm, his_length, pred_length, pred_mode)
valid_dataset = MyDataset(X_valid_norm, Y_valid_norm, his_length, pred_length, pred_mode)
test_dataset = MyDataset(X_test_norm, Y_test_norm, his_length, pred_length, pred_mode)

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)





from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)
from visualize import plot_samples, plot_latent_space_samples, visualize_and_save_tsne


def run_vae_pipeline(dataset_name: str, vae_type: str):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    # # read data
    # data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    # split data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=1000,
        verbose=1,
    )

    # ----------------------------------------------------------------------------------
    # Save scaler and model
    model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    # save scaler
    save_scaler(scaler=scaler, dir_path=model_save_dir)
    # Save vae
    save_vae_model(vae=vae_model, dir_path=model_save_dir)

    # ----------------------------------------------------------------------------------
    # Visualize posterior samples
    x_decoded = get_posterior_samples(vae_model, scaled_train_data)
    plot_samples(
        samples1=scaled_train_data,
        samples1_name="Original Train",
        samples2=x_decoded,
        samples2_name="Reconstructed Train",
        num_samples=5,
    )
    # ----------------------------------------------------------------------------------
    # Generate prior samples, visualize and save them

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
    # Plot prior samples
    plot_samples(
        samples1=prior_samples,
        samples1_name="Prior Samples",
        num_samples=5,
    )

    # visualize t-sne of original and prior samples
    visualize_and_save_tsne(
        samples1=scaled_train_data,
        samples1_name="Original",
        samples2=prior_samples,
        samples2_name="Generated (Prior)",
        scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
        save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
        max_samples=2000,
    )

    # inverse transformer samples to original scale and save to dir
    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    save_data(
        data=inverse_scaled_prior_samples,
        output_file=os.path.join(
            os.path.join(paths.GEN_DATA_DIR, dataset_name),
            f"{vae_type}_{dataset_name}_prior_samples.npz",
        ),
    )

    # ----------------------------------------------------------------------------------
    # If latent_dim == 2, plot latent space
    if hyperparameters["latent_dim"] == 2:
        plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))

    # ----------------------------------------------------------------------------------
    # later.... load model
    loaded_model = load_vae_model(vae_type, model_save_dir)

    # Verify that loaded model produces same posterior samples
    new_x_decoded = loaded_model.predict(scaled_train_data)
    print(
        "Preds from orig and loaded models equal: ",
        np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    )

    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    # check `/data/` for available datasets
    dataset = "sine_subsampled_train_perc_20"

    # models: vae_dense, vae_conv, timeVAE
    model_name = "timeVAE"

    run_vae_pipeline(dataset, model_name)
