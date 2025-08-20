
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pybalmorel import IncFile, Balmorel, MainResults
from GeneralHelperFunctions import get_combined_obj_value
import random  
import os

# ignore warnings 
import warnings
warnings.filterwarnings("ignore")


class ScenarioGenerator(nn.Module):
    def __init__(self, X_tensor=None, input_shape=(24, 72), latent_dim=32, device='cpu', lr=0.0005, seed=42, scale=True, logger=None):
        """
        A non-conditional autoencoder-based Scenario Generator.
        Removes all usages of condition tensors and condition dimensions.
        """
        super(ScenarioGenerator, self).__init__()

        self.seq_len, self.input_dim = input_shape
        self.flat_dim = self.seq_len * self.input_dim
        self.latent_dim = latent_dim
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.total_loss_history = []
        self.pretrain_loss_history = []
        self.obj_values = []
        self.seed = seed
        self.reconstruction_loss_total = None
        self.new_scenarios = None
        self.new_scenarios_list = []
        self.X_original = None
        self.reconstruction_scenarios = None
        self.scale = scale
        self.logger = logger

        torch.manual_seed(self.seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(self.seed)
        random.seed(self.seed)

        # Encoder (no condition concatenation)
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        # Decoder (takes only latent vector)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.flat_dim),
        )

        self.optimizer_ = torch.optim.Adam(self.parameters(), lr=lr)
        self.X_tensor = X_tensor
        self.to(self.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon.view(-1, self.seq_len, self.input_dim), z

    def cae_loss(self, recon_x, x):
        """Reconstruction loss function"""
        return nn.MSELoss()(recon_x, x)
    
    # def cae_loss(self, recon_x, x_scaled):
    #     """
    #     MSE in original (unscaled, un-logged) space, fully differentiable.
    #     recon_x, x_scaled: (B, seq_len, input_dim)
    #   based on my original code
    #   """
    #     if self.scale:
    #         # inverse StandardScaler in torch: x_log = x_scaled * scale + mean
    #         recon_log = recon_x * self.scaler_scale + self.scaler_mean
    #         x_log     = x_scaled * self.scaler_scale + self.scaler_mean
    #         # inverse log1p: expm1 in torch
    #         recon_orig = torch.expm1(recon_log)
    #         x_orig     = torch.expm1(x_log)
    #     else:
    #         recon_orig = recon_x
    #         x_orig     = x_scaled

    #     return nn.MSELoss()(recon_orig, x_orig)

    def generate_kday_blocks(self, df, k_days=1):
        """
        Groups the input DataFrame into non-overlapping k-day sequences (each day = 24 hours).
        Returns only X (no conditions).
        """
        df = df.copy()
        df.fillna(0, inplace=True)

        df['year'] = df['WY'].astype(int)
        df['week'] = df['SSS'].str.extract(r'S(\d+)').astype(int)
        df['hour'] = df['TTT'].str.extract(r'T(\d+)').astype(int)

        df['day_in_week'] = ((df['hour'] - 1) // 24) + 1
        df['block_in_week'] = ((df['day_in_week'] - 1) // k_days) + 1

        group = df.groupby(['year', 'week', 'block_in_week'])
        df = group.filter(lambda x: len(x) == 24 * k_days)

        feature_cols = df.columns.difference(['WY', 'SSS', 'TTT', 'year', 'week', 'hour', 'day_in_week', 'block_in_week'])

        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found. Ensure there are non-constant features.")

        print(f"Using {len(feature_cols)} features for {k_days}-day blocks.")

        # empty df to later collect generated scenarios as tidy table
        self.empty_df = pd.DataFrame(columns=feature_cols)

        samples = []
        for (_, _, _), grp in df.groupby(['year', 'week', 'block_in_week']):
            X_block = grp[feature_cols].values
            samples.append(X_block)

        X = np.stack(samples)  # (N_blocks, k*24, num_features)
        return X

    def load_and_process_data(self, file_path, k_days=1):
        print("-" * 121)
        print("Loading and processing data from file:", file_path)

        df = pd.read_csv(file_path)
        X = self.generate_kday_blocks(df, k_days=k_days)
        self.X_original = X.copy()

        if self.scale:
            #X_log = np.log1p(X)
            X_log = X
            self.scaler = StandardScaler()
            X_log_scaled = self.scaler.fit_transform(X_log.reshape(-1, X_log.shape[-1])).reshape(X_log.shape)
            
            self.scaler_mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device).view(1, 1, -1)
            self.scaler_scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device).view(1, 1, -1)
        else:
            X_log_scaled = X

        self.X_tensor = torch.tensor(X_log_scaled, dtype=torch.float32)

    def pretrain(self, epochs=20, batch_size=32):
        if self.X_tensor is None:
            raise ValueError("X_tensor is not set. Please load or provide the data before training.")

        dataset = TensorDataset(self.X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        log = (self.logger.info if self.logger else print)

        self.train()
        log("-" * 121)
        log("Pretraining the model...")
        log("*" * 74)
        log(f"Pretraining on {len(loader)} batches with batch size {batch_size} for {epochs} epochs.")

        for epoch in range(epochs):
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                self.optimizer_.zero_grad()
                recon_x, _ = self(x_batch)
                loss = self.cae_loss(recon_x, x_batch)
                loss.backward()
                self.optimizer_.step()
                total_loss += loss.item()
            avg_loss = total_loss
            self.pretrain_loss_history.append(avg_loss)
            log(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    def generate_scenario(self, batch_size=32, n_scenarios=10):
        if self.X_tensor is None:
            raise ValueError("X_tensor is not set. Please load or provide the data before generating scenarios.")

        dataset = TensorDataset(self.X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        self.empty_df.drop(self.empty_df.index, inplace=True)
        self.empty_df.drop(columns=['scenario_id', 'time_step'], inplace=True, errors='ignore')

        all_z_batches = []
        all_recon_batches = []
        recon_loss_total = []

        for (x_batch,) in loader:
            x_batch = x_batch.to(self.device)
            self.optimizer_.zero_grad()

            recon_x, z = self(x_batch)
            recon_loss = self.cae_loss(recon_x, x_batch)
            recon_loss_total.append(recon_loss)

            all_z_batches.append(z.cpu().detach().numpy())
            all_recon_batches.append(recon_x.cpu().detach().numpy())

        self.reconstruction_loss_total = torch.stack(recon_loss_total).sum()

        all_z_batches = np.concatenate(all_z_batches, axis=0)
        all_recon_batches = np.concatenate(all_recon_batches, axis=0)
        all_recon_batches = all_recon_batches.reshape(-1, self.seq_len, self.input_dim)

        if self.scale:
            #all_recon_batches = np.expm1(self.scaler.inverse_transform(all_recon_batches.reshape(-1, self.input_dim)).reshape(all_recon_batches.shape))
            all_recon_batches = self.scaler.inverse_transform(all_recon_batches.reshape(-1, self.input_dim)).reshape(all_recon_batches.shape)

        self.reconstruction_scenarios = all_recon_batches

        # KMeans over latent space only
        kmeans_z = KMeans(n_clusters=n_scenarios, random_state=self.seed)
        kmeans_z.fit(all_z_batches)

        centroids_z = torch.tensor(kmeans_z.cluster_centers_, dtype=torch.float32).to(self.device)
        self.new_scenarios = self.decoder(centroids_z)
        self.new_scenarios = self.new_scenarios.view(-1, self.seq_len, self.input_dim).cpu().detach().numpy()

        if self.scale:
            #self.new_scenarios = np.expm1(self.scaler.inverse_transform(self.new_scenarios.reshape(-1, self.new_scenarios.shape[-1])).reshape(self.new_scenarios.shape))
            self.new_scenarios = self.scaler.inverse_transform(self.new_scenarios.reshape(-1, self.new_scenarios.shape[-1])).reshape(self.new_scenarios.shape)

        self.new_scenarios_list.append(self.new_scenarios)

        # Build tidy DataFrame with scenario_id and time_step
        new_rows = []
        for i in range(self.new_scenarios.shape[0]):
            for j in range(self.new_scenarios.shape[1]):
                new_row = pd.Series(self.new_scenarios[i, j, :], index=self.empty_df.columns)
                new_row['scenario_id'] = i + 1
                new_row['time_step'] = j + 1
                new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)
        self.empty_df = pd.concat([self.empty_df, new_df], ignore_index=True)

        self.empty_df['scenario_id'] = self.empty_df['scenario_id'].astype(int)
        self.empty_df['time_step'] = self.empty_df['time_step'].astype(int)

        cols = ['scenario_id', 'time_step'] + [c for c in self.empty_df.columns if c not in ['scenario_id', 'time_step']]
        self.empty_df = self.empty_df[cols]

        return self.new_scenarios, self.empty_df

    def update(self, obj_value, epoch=None):
        
        log = (self.logger.info if self.logger else print)

        if epoch == 0:
            log("-" * 121)
            log("Updating model with the feedback...")
            log("*" * 74)

        self.obj_values.append(obj_value)
        policy_loss = obj_value
        total_batch_loss = self.reconstruction_loss_total + policy_loss

        self.optimizer_.zero_grad()
        total_batch_loss.backward()
        self.optimizer_.step()

        self.total_loss_history.append(total_batch_loss.item())
        prefix = f'Epoch {epoch+1} - ' if epoch is not None else ''
        log(f"{prefix}Total Loss: {total_batch_loss.item():.6f} | Reconstruction Loss: {self.reconstruction_loss_total.item():.6f} | Policy Loss: {policy_loss.item():.6f}")

    def plot_loss_history(self):
        plt.plot(self.pretrain_loss_history, label='Pretrain Loss')
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.plot(self.total_loss_history, label='Total Loss')
        plt.plot(self.obj_values, label='Objective Values', linestyle='--')
        plt.title("Total Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_scenarios(self, selecting_feature_index=0):
        if self.new_scenarios is None:
            raise ValueError("No new scenarios generated. Please call generate_scenario() first.")

        plt.figure(figsize=(12, 6))
        for i in range(self.X_original.shape[0]):
            plt.plot(self.X_original[i, :, selecting_feature_index], alpha=0.1)

        for i in range(self.new_scenarios.shape[0]):
            plt.plot(self.new_scenarios[i, :, selecting_feature_index], alpha=0.7)

        plt.title("Generated Scenario vs Original Scenario")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer_.state_dict(),
            'total_loss_history': self.total_loss_history,
            'pretrain_loss_history': self.pretrain_loss_history,
            'obj_values': self.obj_values,
            'new_scenarios_list': self.new_scenarios_list,
            'X_original': self.X_original,
            'X_tensor': self.X_tensor,
            'reconstruction_loss_total': self.reconstruction_loss_total,
            'scaler': self.scaler if self.scale else None,
            'empty_df': self.empty_df,
            'reconstruction_scenarios': self.reconstruction_scenarios if hasattr(self, 'reconstruction_scenarios') else 0
        }, file_path)

        print("-" * 121)
        print(f'Model and additional attributes saved to {file_path}')

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_loss_history = checkpoint['total_loss_history']
        self.pretrain_loss_history = checkpoint['pretrain_loss_history']
        self.obj_values = checkpoint['obj_values']
        self.new_scenarios_list = checkpoint['new_scenarios_list']
        self.X_original = checkpoint['X_original']
        self.X_tensor = checkpoint['X_tensor']
        self.reconstruction_loss_total = checkpoint['reconstruction_loss_total']
        self.scaler = checkpoint['scaler'] 
        self.empty_df = checkpoint['empty_df']
        self.reconstruction_scenarios = checkpoint.get('reconstruction_scenarios', None)
        self.to(self.device)
        
        print(f'Model and additional attributes loaded from {file_path}')

    def plot_reconstruction_scenarios(self, features_to_plot, sample_index=0):
        num_features = len(features_to_plot)
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        for i, feature in enumerate(features_to_plot):
            if i >= len(axes):
                break
            axes[i].plot(self.X_original[sample_index, :, feature], label='Original', alpha=0.5)
            axes[i].plot(self.reconstruction_scenarios[sample_index, :, feature], label='Reconstructed', alpha=0.7)
            axes[i].set_title(f'Feature {feature}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()
        print("Reconstruction scenarios plotted for sample index:", sample_index)
        print("Features plotted:", features_to_plot)
        print("Original data shape:", self.X_original.shape)
        print("Reconstruction scenarios shape:", self.reconstruction_scenarios.shape)
        print("Sample index:", sample_index)
        print("Number of features:", num_features)
        
def convert_to_incfiles(new_scenarios_df: pd.DataFrame,
                        scenario: str,
                        scenario_folder: str,
                        balmorel_model_path: str = 'Balmorel',
                        gams_system_directory: str | None = None,
                        overwrite_data_load: bool = False):
    
    
    # Get parameters and temporal resolution
    parameters = new_scenarios_df.columns[2:] # pop scenario id and time_step
    balmorel_season_time_index = [f"S{row['scenario_id']:02d} . T{row['time_step']:03d}" for i,row  in new_scenarios_df.loc[:, ['scenario_id', 'time_step']].iterrows()]
    balmorel_season_index = [f"S{season:02d}" for season in new_scenarios_df.loc[:, 'scenario_id'].values]
    balmorel_term_index = [f"T{term:03d}" for term in new_scenarios_df.loc[:, 'time_step'].values]
    
    # Load Balmorel input data for descriptions and set names
    model = Balmorel(balmorel_model_path, gams_system_directory=gams_system_directory)
    model.load_incfiles(scenario, overwrite=overwrite_data_load)
    
    placeholder_parameter = ''
    for parameter in parameters:

        # Get elements and parameter_name
        elements = parameter.split('|')
        parameter_name = elements.pop(0)
        
        # Skip if we already processed this parameter
        if placeholder_parameter == parameter_name:
            continue
        else:
            placeholder_parameter = parameter_name
        
        # Find all values of this parameter
        idx=new_scenarios_df.columns.str.find(parameter_name + '|') == 0
        
        # Get sets, values and explanation
        sets = model.input_data[scenario][parameter_name].domains_as_strings
        text = model.input_data[scenario][parameter_name].text
        table_or_param = 'PARAMETER' if (len(sets) == 1) else 'TABLE'
        if table_or_param == 'PARAMETER':
            prefix = f"""{table_or_param} {parameter_name}({','.join(sets)}) "{text}" \n/\n"""
            suffix = '\n/;'
        else:
            prefix = f"""{table_or_param} {parameter_name}({','.join(sets)}) "{text}" \n"""
            suffix = '\n;'
        
        df = new_scenarios_df.loc[:, idx]
        df.columns = df.columns.str.replace(parameter_name+'|', '', n=1).str.replace('|', ' . ')

        if 'SSS' in sets and 'TTT' in sets:
            df.index = balmorel_season_time_index   
            df.index.name = 'ST'
            df = df.pivot_table(index='ST', aggfunc='mean')
            df.index.name = ''
            df = df.T
            
        elif 'SSS' in sets:
            df.index = balmorel_season_index
            df.index.name = 'S'
            df = df.pivot_table(index='S', aggfunc='mean')
            df.index.name = ''
            df = df.T
            
        else:
            df = df.mean().T
            if table_or_param != 'PARAMETER':
                prefix = prefix.replace('TABLE', 'PARAMETER')
                prefix += "/\n"
                suffix = "\n/;"
        
        if parameter_name != 'HYDROGEN_DH2':
            file = IncFile(
                name=parameter_name,
                prefix=prefix,
                body=df.to_string(),
                suffix=suffix,
                path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
            )
        else:
            file = IncFile(
                name='HYDROGEN_DH2',
                prefix='',
                suffix='',
                body="\n".join([f"HYDROGEN_DH2('{year}', '{region}') = HYDROGEN_DH2('{year}', '{region}') + {df.loc[f'{year} . {region}']};" for year, region in df.index.str.split(' . ', expand=True)]),
                path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
            )
            
        if parameter_name == 'DE_VAR_T':
            file.suffix += "\n* Flat profiles for industry and datacenter"
            file.suffix += "\nDE_VAR_T(RRR,'PII',SSS,TTT)=1;"
            file.suffix += "\nDE_VAR_T(RRR,'DATACENTER',SSS,TTT)=1;"
            
        file.save()
        
    # Define temporal resolution
    IncFile(
        name='S',
        prefix="SET S(SSS)  'Seasons in the simulation'\n/\n",
        body=', '.join(np.unique(balmorel_season_index)),
        suffix='\n/;',
        path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
    ).save()
    IncFile(
        name='T',
        prefix="SET T(TTT)  'Time periods within a season in the simulation'\n/\n",
        body=','.join(np.unique(balmorel_term_index)),
        suffix='\n/;',
        path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
    ).save()

        
def pretrain(epochs: int, days: int = 1, n_scenarios: int = 4, latent_dim: int = 64, batch_size: int = 256, learning_rate: float = 5e-4, seed: int = 42, n_features=72, logger=None, scenario_folder: str = 'operun'):
    """
    Pretrain the scenario generator.
    
    Parameters
    ----------
    epochs : int
        Number of pretraining epochs.
    days : int, default 1
        Length of each block in days (input_shape uses days*24).
    n_scenarios : int, default 4
        How many scenarios to generate after pretraining.
    latent_dim : int, default 64
        Latent space dimensionality for the ScenarioGenerator.
    batch_size : int, default 256
        Batch size used in pretraining and generation.
    learning_rate : float, default 1e-3
        Optimizer learning rate for the ScenarioGenerator.
    seed : int, default 42
        Random seed for reproducibility.
    n_features : int, default 72
        Amount of features in training data, 72 for all weather years for small-system, 33 for one weather year and no yearly data.
    logger : bool, default None
        Whether to log or print (None = False)
    scenario_folder : str, default operun
        Which scenario folder to run
    """
    
    log = (logger.info if logger else print)


    model = ScenarioGenerator(input_shape=(days*24, n_features), latent_dim=latent_dim, lr=learning_rate, seed=seed, logger=logger)
    model.load_and_process_data('Pre-Processing/Output/genmodel_input.csv', k_days=days)
    
    log(f"Pretraining for {epochs} epochs, batch_size={batch_size}, lr={learning_rate}, seed={seed}")
    
    model.pretrain(epochs=epochs, batch_size=batch_size)
    
    log("Pretraining done; generating initial scenarios")
    
    # create new incfiles
    new_scenarios, new_scenarios_df = model.generate_scenario(batch_size=batch_size, n_scenarios=n_scenarios)
    
    log("Converting scenarios to INC files")
    
    convert_to_incfiles(new_scenarios_df, 'base', scenario_folder, gams_system_directory=os.getenv('GAMS_DIR','/appl/gams/47.6.0'), overwrite_data_load=True)

    # model.save_model(f'Pre-Processing/Output/{scenario}_model.pth')

    return model

def train(model: ScenarioGenerator, scenario: str, epoch_string: str, n_scenarios: int=4, batch_size: int = 256, logger=None, scenario_folder: str = 'operun'):
    
    # Get the objective value
    results = MainResults([f'MainResults_{scenario}_{runtype}_{epoch_string}.gdx' for runtype in ['capacity', 'dispatch']],
                          paths=[f'Balmorel/{scenario_folder}/model'])
    obj_value, capital_costs, operational_costs = get_combined_obj_value(results, return_capex_opex_dfs=True)

    log = (logger.info if logger else print)
    
    log("read objective values from capacity and dispatch runs")
    log(f"Capital costs:\n%s"%capital_costs.to_string())
    log(f"Operational costs:\n%s"%operational_costs.to_string())
    log(f'Loss value: {obj_value} M€')

    # update the model with the objective value
    model.update(obj_value, epoch=int(epoch))
    
    # create new incfiles
    new_scenarios, new_scenarios_df = model.generate_scenario(batch_size=batch_size, n_scenarios=n_scenarios)
    
    log("running convert_to_incfiles")
    convert_to_incfiles(new_scenarios_df, 'base', scenario_folder, gams_system_directory=os.getenv('GAMS_DIR','/appl/gams/47.6.0'))
    
    return model