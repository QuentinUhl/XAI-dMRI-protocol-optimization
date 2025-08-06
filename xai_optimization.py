####################################################################################################################################################
#%%
# 1. Imports
####################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib.patches import Ellipse
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import graymatter_swissknife as gmsk
import time
import shap
from sklearn.model_selection import train_test_split
from ipywidgets import interact
import ipywidgets as widgets

####################################################################################################################################################
# Toggles
####################################################################################################################################################

# Define the random seed
seed = 12
np.random.seed(seed)

# Turn this to True if you want to use the normalized parameters to learn
use_normalized_parameters = True

# Noise type: Rician
rice_noise_toggle = False
gaussian_noise_toggle = True
no_noise_toggle = False
assert rice_noise_toggle+gaussian_noise_toggle+no_noise_toggle == 1, "Only one noise type should be selected"

# Turn this to True if you want to plot the protocol
plot_toggle = False

# Define the C2 protocol
b = np.array([1, 2.5, 5, 7.35, 1, 2.5, 5, 7.5, 10, 1, 2.5, 5, 7.5, 10, 12.5])
big_delta = np.array([12, 12, 15, 20, 27, 27, 27, 27, 27, 45, 45, 45, 45, 45, 45])
small_delta = 5
td = big_delta - small_delta/3
num_inputs = len(b)
nb_directions = np.array([20, 30, 32, 34, 20, 30, 32, 34, 44, 20, 30, 32, 34, 44, 64])
# Define the acquisition parameters
acq_param = gmsk.models.parameters.acq_parameters.AcquisitionParameters(b, big_delta, small_delta)

# Define the NEXI model and parameter limits
model = 'Nexi'
gmsk_model = gmsk.models.NEXI.nexi.Nexi
gmsk_model_lower = 'nexi'
signal_function = gmsk_model.get_signal
parameter_names = ['tex', 'Di', 'De', 'f']
parameter_names_lower = [name.lower() for name in parameter_names]
parameter_limits = np.array([[1, 70], [1.7, 3.5], [0.5, 1.5], [0.15, 0.8]])
model_dim = 4

# Define the size of the datasets
num_samples = 1000000

# Define the training, validation and test sizes
# 80% training, 10% validation, 10% test
training_percentage = 0.8
valid_over_test_percentage = 0.5 # 0.5 means 50% of the remaining data will be used for validation and 50% for testing
training_size = int(num_samples * training_percentage)
validation_size = int(num_samples * training_percentage * valid_over_test_percentage)
test_size = num_samples - training_size - validation_size

# Number of CPU cores to use for parallel processing
n_core = -2

# XGBoost features
n_estimators=128
n_jobs=16
max_depth=8
multi_strategy="one_output_per_tree"
subsample=0.6
device='gpu'

####################################################################################################################################################
# Load sigma distribution
####################################################################################################################################################

# Load the CSV file (index in the column 0)
experimental_sigma = np.array(np.load(r'./C2_complex_all_sigma.npz')['sigma'])
# Take only the sigma that are inside the 2.5 - 97.5 percentile to avoid <1e-10 and >1 sigmas that are not realistic
experimental_sigma_2p5_percentile = np.percentile(experimental_sigma, 2.5)
experimental_sigma_97p5_percentile = np.percentile(experimental_sigma, 97.5)
experimental_sigma_inside_ci = experimental_sigma[
    (experimental_sigma >= experimental_sigma_2p5_percentile) & (
                experimental_sigma <= experimental_sigma_97p5_percentile)]
experimental_sigma = experimental_sigma_inside_ci

plt.figure(figsize=(6, 4))
plt.hist(experimental_sigma_inside_ci, bins=300, range=(0, 0.15))
plt.xlabel('Experimental sigma')
plt.ylabel('Number of samples')
plt.title('Distribution of experimental sigma')
plt.xlim(0, 0.1)
plt.show()

####################################################################################################################################################
# Parameters
####################################################################################################################################################

# Set the random seed for reproducibility
rng = np.random.default_rng(seed)
# Simulate random parameters in the parameter space
target_parameters = rng.uniform(parameter_limits[:, 0], parameter_limits[:, 1], (num_samples, model_dim))
# Pick num_samples random sigma values
sigma = rng.choice(experimental_sigma, num_samples)

# Plot the histograms of the parameters
fig, axs = plt.subplots(2, 2, figsize=(7, 7), dpi=300)
fig.suptitle('Distribution of ground truth parameters')
axs[0, 0].hist(target_parameters[:, 0], bins=50)
axs[0, 0].set_title(r'$t_{ex}$')
axs[0, 1].hist(target_parameters[:, 1], bins=50)
axs[0, 1].set_title(r'$D_i$')
axs[1, 0].hist(target_parameters[:, 2], bins=50)
axs[1, 0].set_title(r'$D_e$')
axs[1, 1].hist(target_parameters[:, 3], bins=50)
axs[1, 1].set_title(r'$f$')

plt.tight_layout()
plt.show()

####################################################################################################################################################
#%%
# 2. Create the train, test and validation Datasets
####################################################################################################################################################

# Compute the signal for all parameters
target_signals = Parallel(n_jobs=n_core)(
    delayed(signal_function)(target_parameters[irunning], acq_param) for irunning in
    range(num_samples))
target_signals = np.array(target_signals)
# Check the shape of the signal is (num_samples, num_inputs)
assert target_signals.shape == (
num_samples, num_inputs), f"Signal shape is {target_signals.shape}, expected {(num_samples, num_inputs)}"

# Generate random Rice noise from a normal distribution
# Gaussian case
if gaussian_noise_toggle:
    resulting_sigma = np.divide(1, np.sqrt(nb_directions))[None, :] * sigma[:, None]
    target_signals_noisy = target_signals + np.random.randn(*target_signals.shape) * resulting_sigma
# Rice noise case
elif rice_noise_toggle:
    target_signals_noisy = np.zeros_like(target_signals)
    for k, nb_dir_k in enumerate(nb_directions):
        signal_snr_bval_k_in_multiple_directions = np.zeros((target_signals.shape[0], nb_dir_k))
        for i in range(nb_dir_k):
            dist1 = np.random.randn(target_signals.shape[0])
            dist2 = np.random.randn(target_signals.shape[0])
            signal_snr_bval_k_dir_i = np.sqrt((target_signals[:, k] + dist1 * sigma) ** 2
                                                + (dist2 * sigma) ** 2)
            signal_snr_bval_k_in_multiple_directions[:, i] = signal_snr_bval_k_dir_i
        target_signals_noisy[:, k] = np.sum(signal_snr_bval_k_in_multiple_directions, axis=-1)/nb_dir_k
# No noise case
elif no_noise_toggle:
    target_signals_noisy = target_signals
else:
    raise ValueError("No noise type selected")

# Create train, validation and test datasets
def norm_p(parameters, parameter_limits=parameter_limits):
    return (parameters - parameter_limits[:, 0]) / (parameter_limits[:, 1] - parameter_limits[:, 0])
def denorm_p(parameters, parameter_limits=parameter_limits):
    return parameters * (parameter_limits[:, 1] - parameter_limits[:, 0]) + parameter_limits[:, 0]
normed_target_parameters = norm_p(target_parameters, parameter_limits)

# Split the data into training, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(target_signals_noisy, normed_target_parameters, 
                                                    test_size=(1-training_percentage), 
                                                    random_state=seed)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, 
                                                    test_size=(1-valid_over_test_percentage),
                                                    random_state=seed)

####################################################################################################################################################
# Plot the distribution of the parameters in a histogram
####################################################################################################################################################

fig, axs = plt.subplots(2, 2, figsize=(14, 7), dpi=300)
fig.suptitle('Distribution of ground truth parameters for the training set')
axs[0, 0].hist(y_train[:, 0], bins=50)
axs[0, 0].set_title(r'$t_{ex}$')
axs[0, 1].hist(y_train[:, 1], bins=50)
axs[0, 1].set_title(r'$D_i$')
axs[1, 0].hist(y_train[:, 2], bins=50)
axs[1, 0].set_title(r'$D_e$')
axs[1, 1].hist(y_train[:, 3], bins=50)
axs[1, 1].set_title(r'$f$')
plt.tight_layout()
plt.show()

####################################################################################################################################################
#%%
# 3. Train and test XGBoost model to get its error
####################################################################################################################################################

# Train a regressor on the full protocol
# Create the regressor
reg = xgb.XGBRegressor(
    tree_method="hist",  # Use GPU acceleration, or use "hist" for CPU
    n_estimators=n_estimators,
    n_jobs=n_jobs,
    max_depth=max_depth,
    multi_strategy=multi_strategy,
    subsample=subsample,
    device=device
)
# Time the training
start = time.time()
reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
end = time.time()
print(f"Training time: {end - start:.7f}s")

# 4. Test the model
# Predict the parameters
start = time.time()
y_hat = reg.predict(X_test)
end = time.time()
# Compute the parametric RMSE
parametric_scores_single_inference = np.sqrt(np.square(denorm_p(y_test) - denorm_p(y_hat))).mean(axis=0)

print(f"RMSE on the full protocol (narrow parametric ranges):")
print(f"tex: {parametric_scores_single_inference[0]:.4f} ms")
print(f"Di: {parametric_scores_single_inference[1]:.4f} µm²/ms")
print(f"De: {parametric_scores_single_inference[2]:.4f} µm²/ms")
print(f"f: {parametric_scores_single_inference[3]:.4f}")

print(f"Evaluation/test time: {end - start:.7f}s")



####################################################################################################################################################
# %%
# 4. Feature selection Using SHAP
####################################################################################################################################################

# Initialize the lists to store the results
n_feat = X_test.shape[1]
scores = []
parametric_scores = []
n_per_threshold = []
successive_b = []
successive_td = []
successive_support = []

# Loop initialization
select_X_train = X_train
select_X_test = X_test
n_select_feat = n_feat
select_b = b
select_td = td
successive_b.append(select_b)
successive_td.append(select_td)

# First step : add the MSE of the full model
selection_model = xgb.XGBRegressor(
                                    tree_method="hist", 
                                    n_estimators=n_estimators,
                                    n_jobs=n_jobs,
                                    max_depth=max_depth,
                                    multi_strategy=multi_strategy,
                                    subsample=subsample,
                                    device=device
                                    )
selection_model.fit(select_X_train, y_train)
predictions = selection_model.predict(select_X_test)
rmse = np.sqrt(np.square(y_test - predictions).mean())
scores.append(rmse)
parametric_scores.append(np.sqrt(np.square(denorm_p(y_test) - denorm_p(predictions))).mean(axis=0))
n_per_threshold.append(n_feat)
# Update SHAP values
X_set_to_explain = select_X_test
explainer = shap.TreeExplainer(selection_model)
shap_values = explainer.shap_values(X_set_to_explain)
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))
lower_abs_shap = np.argmin(mean_abs_shap_values)
support = [True for i in range(n_feat)]
successive_support.append(np.copy(support))
support[lower_abs_shap] = False
# Remove the feature with the lowest SHAP value: loop until only 2 features remain
while n_select_feat > 2:
    print(f"Number of features: {n_select_feat}")
    select_X_train = select_X_train[:, support]
    select_b = select_b[support]
    select_td = select_td[support]
    successive_b.append(select_b)
    successive_td.append(select_td)
    successive_support.append(np.copy(support))
    # train model
    selection_model = xgb.XGBRegressor(
                                        tree_method="hist", 
                                        n_estimators=n_estimators,
                                        n_jobs=n_jobs,
                                        max_depth=max_depth,
                                        multi_strategy=multi_strategy,
                                        subsample=subsample,
                                        device=device
                                        )
    selection_model.fit(select_X_train, y_train)
    # eval model
    # select_X_test = selection.transform(select_X_test)
    select_X_test = select_X_test[:, support]
    predictions = selection_model.predict(select_X_test)
    rmse = np.sqrt(np.square(y_test - predictions).mean())
    scores.append(rmse)
    parametric_scores.append(np.sqrt(np.square(denorm_p(y_test) - denorm_p(predictions))).mean(axis=0))
    n_select_feat = select_X_train.shape[1]
    n_per_threshold.append(n_select_feat)
    print(f"n={n_select_feat}, RMSE: {rmse}")
    # Update SHAP values
    X_set_to_explain = select_X_test
    explainer = shap.TreeExplainer(selection_model)
    shap_values = explainer.shap_values(X_set_to_explain)
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))
    lower_abs_shap = np.argmin(mean_abs_shap_values)
    support = [True for i in range(n_select_feat)]
    support[lower_abs_shap] = False
parametric_scores = np.array(parametric_scores)


# Compute the ranks of the features based on their presence in the successive indexes
indexes = np.array([i for i in range(len(b))])
concatenated_successive_indexes = None
successive_indexes = [indexes]
for support in successive_support[1:]:
    current_indexes = successive_indexes[-1][support]
    successive_indexes.append(current_indexes)
    concatenated_successive_indexes = np.concatenate((concatenated_successive_indexes, current_indexes), axis=0) if concatenated_successive_indexes is not None else current_indexes
# rank is given by the number of times a feature appears in concatenated_successive_indexes
ranks = len(b) - np.array([np.sum(concatenated_successive_indexes == i) for i in range(len(b))])


# Plot the protocol with the ranks
colors = sns.color_palette("Paired", 12)
# Create a figure and axis
fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
# Set limits for better spacing around the circles
x_lim_min = 0
x_lim_max = 14
y_lim_min = 8
y_lim_max = 50
# Plot circles and place numbers inside them
for i, (x_val, y_val, rank_val) in enumerate(zip(b, big_delta, ranks), start=1):
    if rank_val <= 8:
        # Determine the width and height of the ellipse
        radius = 1.2
        width = radius  # Major axis for x-direction
        height = radius * (y_lim_max - y_lim_min) / (x_lim_max - x_lim_min)  # Adjust for y-direction
        edgecolor = 'green'
        fontcolor = 'white'
        # Plot the ellipse
        ellipse = Ellipse((x_val, y_val), width, height, edgecolor=colors[3], facecolor=colors[2])
        ax.add_patch(ellipse)
        # Place the number inside the circle (en gras)
        if rank_val==2:
            ax.text(x_val, y_val, str('1*'), color='black', ha='center', va='center', fontsize=14,
                    fontweight='bold')
        else:
            ax.text(x_val, y_val, str(int(rank_val)), color='black', ha='center', va='center', fontsize=14,
                    fontweight='bold')
    else:
        # Determine the width and height of the ellipse
        radius = 1.1
        width = radius  # Major axis for x-direction
        height = radius * (y_lim_max - y_lim_min) / (x_lim_max - x_lim_min)  # Adjust for y-direction
        edgecolor = 'blue'
        fontcolor = 'white'
        # Plot the ellipse
        ellipse = Ellipse((x_val, y_val), width, height, edgecolor=colors[1], facecolor='none')
        ax.add_patch(ellipse)
        # Place the number inside the circle
        ax.text(x_val, y_val, str(int(rank_val)), color='black', ha='center', va='center', fontsize=14,
                fontweight='bold')
ax.set_xticks([1, 2.5, 5, 7.5, 10, 12.5])
ax.set_yticks([12, 15, 20, 27, 45])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel('b-value (ms/µm²)', fontsize=16)
ax.set_ylabel('Δ (ms)', fontsize=16)
ax.set_xlim(x_lim_min, x_lim_max)
ax.set_ylim(y_lim_min, y_lim_max)
plt.savefig(f'./C2_complex_xgb_optimization.png', dpi=300)
plt.show()



####################################################################################################################################################
#%%
# 5. (Additional) Slider plot: protocol depending on the number of features selected
####################################################################################################################################################

def plot_protocol(n_features):
    # Erase the previous plot
    plt.clf()
    # # Increase figure size
    # plt.figure(figsize=(6, 4))
    # Two subplots, on the right the protocol and on the left the MSE vs number of features with a circle on the selected threshold
    plt.subplot(231)
    # use numpy to find the index of n_features in n_per_threshold (it is in the dataset)
    nb_feat_index = np.where(np.array(n_per_threshold) == n_features)[0][0]
    plt.plot(n_per_threshold, scores)
    # if nb_analyzed_features-n_features >= 0:
    plt.plot(n_per_threshold[nb_feat_index], scores[nb_feat_index], 'o')
    plt.xlabel('Number of features')
    plt.ylabel('RMSE')
    plt.title('RMSE vs number of features')

    # Plot the corresponding protocol
    lengths_of_successive_prot = np.array([len(b) for b in successive_b])
    corresponding_index = np.where(lengths_of_successive_prot == n_features)[0][0]
    plt.subplot(234)
    plt.plot(successive_b[corresponding_index], successive_td[corresponding_index], 'o')
    plt.title('Protocol')
    plt.xlabel('b (ms/µm²)')
    plt.ylabel(r'$\Delta$ (ms)')
    plt.xlim(0, 13)
    plt.ylim(8, 50)

    # Plot the parameter errors in 4 subplots
    plt.subplot(232)
    plt.plot(n_per_threshold, parametric_scores[:,0])
    plt.plot(n_per_threshold[nb_feat_index], parametric_scores[nb_feat_index][0], 'o')
    plt.title(r'$t_{ex}$')
    plt.subplot(233)
    plt.plot(n_per_threshold, parametric_scores[:,1])
    plt.plot(n_per_threshold[nb_feat_index], parametric_scores[nb_feat_index][1], 'o')
    plt.title(r'$D_i$')
    plt.subplot(235)
    plt.plot(n_per_threshold, parametric_scores[:,2])
    plt.plot(n_per_threshold[nb_feat_index], parametric_scores[nb_feat_index][2], 'o')
    plt.title(r'$D_e$')
    plt.subplot(236)
    plt.plot(n_per_threshold, parametric_scores[:,3])
    plt.plot(n_per_threshold[nb_feat_index], parametric_scores[nb_feat_index][3], 'o')
    plt.title(r'$f$')
    plt.tight_layout()
    plt.show()


# While sliding the threshold, the protocol will change based on the features selected (using the support from the SelectFromModel)
interact(plot_protocol, n_features=widgets.IntSlider(min=2, max=len(b), step=1, value=0))

#%%