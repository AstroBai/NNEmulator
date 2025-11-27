# NNEmulator

This is a template for a neural network emulator based on Torch.

Several options can be configured in the `config.yaml` file.

A very clear illustration of how data flow through a neural network (I forgot where I found this GIF, please let me know if you know who created it):

![illustration](readme_files/neural_network.GIF)

## Requirements
- PyTorch
- scikit-learn
- PyYAML
- joblib
- numpy
- scipy

## Usage

After configuring the `config.yaml` file, you can train the model:

```python
emu = ANN_Emu(config_path='config.yaml')
train_loader, trial_loader = emu.get_data_loaders()
emu.train_model(train_loader, trial_loader)
```

After executing the above code, the trained model will be saved in the directory specified in the `config.yaml` file.

Then you can use the trained model to make predictions:

```python
# test on trials
trial_X_np = emu.Trial_X_original.numpy()
trial_Y_np = emu.Trial_Y_original.numpy()
predictions = emu.predict(trial_X_np, xs=emu.xs)
```

You don't need to train the model again if you have already saved it. You can directly load the trained model:

```python
emu = ANN_Emu(config_path='config.yaml')
predictions = emu.predict(X, xs=emu.xs)
```

Note here `X` stands for the input parameters you want to make predictions on, while `xs` is the x-axis values for the output functions, e.g., the wavenumbers $k$ for a spectrum.

## Configuration

The emulator is controlled entirely by a YAML configuration file with three main sections:

- DataParam — how data is loaded and transformed  
- ANNParam — architecture, training strategy, optimization settings  
- OutputParam — where to save trained models and diagnostics  

Each entry is described precisely below.

### 1. DataParam — Dataset and Preprocessing

```yaml
DataParam:
  Training_ParamPath: /path/to/train_params.npy
  Training_DataPath: /path/to/train_data.npy
  Trial_ParamPath: /path/to/trial_params.npy
  Trial_DataPath: /path/trial_data.npy
  xs_Path: /path/to/x-axis.npy
  Do_Log: False
```

#### Training_ParamPath
- Shape: `(N_train, N_params)`
- Each row is one set of cosmological/physical parameters.
- Input to the network.

#### Training_DataPath
- Shape: `(N_train, N_outputs)`
- Output vector for each input parameter set.  
- Could be a function sampled on a grid (e.g., power spectrum).

#### Trial_ParamPath
- Same format as training parameters but used as a validation/evaluation set.

#### Trial_DataPath
- Paired outputs for the trial parameters.

#### xs_Path
- Shape: `(N_outputs,)`
- The x-axis associated with the output spectra.
- Used for CubicSpline interpolation when the user queries the emulator at new coordinates.

#### Do_Log
- If `True`, the emulator uses:
  \[
  y_\text{train} \rightarrow \log(y + 10^{-10})
  \]
- Helps stabilize training when outputs vary by orders of magnitude.
- Automatically exponentiated back during prediction.

### 2. ANNParam — Network Architecture and Training Strategy

```yaml
ANNParam:
  Device: 'cpu'
  Scale_Params: True
  Use_PCA: True
  N_PCA_Components: 16
  S_Hidden: [24, 20]
  Activation: 'silu'
  Positive_Output: False
  Step_LR: True
  Initial_LR: 0.01
  Step_Size: 100
  Gamma: 0.1
  Max_Epochs: 100000
  Early_Stopping: True
  Patience: 10
  Min_Delta: 1e-6
  Dropout: 0
  Weight_Decay: 0
  Plot_Loss: True
```

#### Device
- `'cuda'` uses GPU if available.  
- `'cpu'` forces CPU.

Note: GPU is not always faster than CPU for small datasets.

#### Scale_Params
- If `True`: uses `MinMaxScaler` to map each input parameter to [0, 1].
- Recommended for stable training.

#### Use_PCA
- Applies PCA to the outputs:
  - Reduces output dimensionality from N_outputs → N_PCA_Components
  - De-noises training targets
  - Reduces model size and training time

#### N_PCA_Components
- Number of principal components retained.
- Should be chosen to capture > 99% variance of the output.

#### S_Hidden
- List defining hidden layer widths.
- Example: `[24, 20]`  
  → Two layers: 24 neurons → 20 neurons → output layer.

#### Activation
Available options:
- `'relu'`
- `'tanh'`
- `'sigmoid'`
- `'gelu'`
- `'silu'` (recommended)
- `'linear'` (no activation)

#### Positive_Output
- If `True`, applies Softplus to enforce strictly positive output.

#### Learning Rate Schedule
- Initial_LR: starting LR for Adam.
- Step_Size: epochs between LR drops.
- Gamma: multiplicative factor (e.g., 0.1 → 10× reduction).

#### Max_Epochs
- Hard cap on training duration.

#### Early_Stopping
- Stops training when the loss stops improving.

#### Patience
- Number of evaluation windows to wait before stopping.

#### Min_Delta
- Minimum improvement threshold.  
- If loss reduction per reporting step is < `Min_Delta`, patience is accumulated.

#### Dropout
- Fraction of units dropped during training `(0.0–1.0)`.
- Use small values like 0.05–0.1 for regularization.

#### Weight_Decay
- L2 penalty in Adam, e.g., `1e-5`.
- Helps prevent overfitting on small datasets.

#### Plot_Loss
- If `True`, saves `loss_curve.png` to visualize training & trial loss evolution.

### 3. OutputParam — Paths and Saving Behavior

```yaml
OutputParam:
  Model_SavePath: //directory/to/your/model
```

#### Model_SavePath
Directory where the following outputs will be stored:

- `ann_emu.pth` (trained model weights)
- `input_scaler.pkl` (MinMaxScaler)
- `pca.pkl` (PCA transform, if used)
- `loss_curve.png` (if enabled)

This folder is created automatically if it does not exist.