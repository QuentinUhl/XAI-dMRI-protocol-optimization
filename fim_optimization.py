####################################################################################################################################################
# %%
# 1. Imports
####################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from joblib import Parallel, delayed
import graymatter_swissknife as gmsk
from ipywidgets import interact
import ipywidgets as widgets



####################################################################################################################################################
# Toggles
####################################################################################################################################################

# Define the random seed
seed = 301
np.random.seed(seed)

# Turn this to True if you want to use the normalized parameters to learn
use_normalized_parameters = True

# Noise type: Gaussian (Condition to use the FIM)
gaussian_noise_toggle = True
no_noise_toggle = False
assert gaussian_noise_toggle+no_noise_toggle == 1, "Only one noise type should be selected"
assert gaussian_noise_toggle, "Only gaussian noise type should be selected here. No noise option is currently disabled but can be re-implemented from this file."

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
# signal_function = gmsk_model.get_signal  # Not used in this script
jacobian_function = gmsk_model.get_jacobian
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

# Taking into account the number of directions for each acquisition
# Gaussian case
if gaussian_noise_toggle:
    resulting_sigma = np.divide(1, np.sqrt(nb_directions))[None, :] * sigma[:, None]
# No noise case
if no_noise_toggle:
    resulting_sigma = np.zeros((num_samples, num_inputs))

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
#%% 2. Compute the Jacobian for each acquisition and each parameter ahead of time
####################################################################################################################################################


sigma_dir_corr = resulting_sigma
parameters = target_parameters

n_examples = sigma_dir_corr.shape[0]  # Number of samples, 1000000
n_acq = sigma_dir_corr.shape[1]  # Number of acquisitions, 15 for Nexi (b_i, td_i) for i until 15
n_param = parameters.shape[1]  # Number of features, 4 for Nexi (tex, di, de, f)

# First step : Compute the determinant of the Fisher Information Matrix (FIM) for all parameters
acq_param = gmsk.models.parameters.acq_parameters.AcquisitionParameters(b, td, small_delta)

# Compute the Jacobian for all parameters
jacobian = Parallel(n_jobs=-2)(
    delayed(jacobian_function)(parameters[irunning], acq_param) for irunning in
    range(n_examples))
jacobian = np.array(jacobian)
# Check the shape of the jacobian is (1000000, 15, 4)
assert jacobian.shape == (
n_examples, n_acq, n_param), f"Jacobian shape is {jacobian.shape}, expected {(n_examples, n_acq, n_param)}"

####################################################################################################################################################
#%% 3. Run the optimization pipeline
####################################################################################################################################################

# Optimizing using the determinant of the FIM, using the proper definition of the FIM, even though tex is bigger than the rest of the parameters
scores = []
# parametric_scores = []
successive_n_acq = []
successive_b = []
successive_td = []
successive_sub_fim = []
successive_support = []
successive_det_fim_excluding_param_k = []
successive_dfepk_indices = []
# Loop initialization
select_n_acq = n_acq
select_b = b
select_delta = td
sub_fim = np.zeros((n_acq, n_param, n_param))
for i in range(n_acq):
    jacobian_over_sigma = np.divide(1, sigma_dir_corr)[:, i, None] * jacobian[:, i, :]
    sub_i_fim = np.sum(jacobian_over_sigma[:, None, :] * jacobian_over_sigma[:, :, None], axis=0)
    sub_i_fim /= n_examples
    sub_fim[i, :, :] = sub_i_fim
# Compute the determinant of the FIM for all parameters
fim_total = np.sum(sub_fim, axis=0)
det_fim = np.linalg.det(fim_total)
select_det_fim = det_fim
support = [True for _ in range(n_acq)]
# Update the lists
scores.append(select_det_fim)
successive_support.append(np.copy(support))
successive_n_acq.append(n_acq)
successive_b.append(select_b)
successive_td.append(select_delta)
while select_n_acq > 1:
    # print(f"Number of features: {select_n_acq}")
    # Convert the support into the index where the value is True
    support_index = np.where(support)[0]
    # Compute the determinant of the FIM for all parameters excluding the k-th parameter
    det_fim_excluding_param_k = np.zeros(len(support_index))
    for det_fim_k_index, k in enumerate(support_index):
        select_k_support = support.copy()
        select_k_support[k] = False
        select_k_fim = np.sum(sub_fim[select_k_support, :, :], axis=0)
        det_fim_excluding_param_k[det_fim_k_index] = np.linalg.det(select_k_fim)
    select_det_fim = np.max(det_fim_excluding_param_k)
    index_param_to_exclude_giving_highest_det_fim = support_index[np.argmax(det_fim_excluding_param_k)]
    successive_det_fim_excluding_param_k.append(det_fim_excluding_param_k)
    successive_dfepk_indices.append([select_n_acq - 1] * select_n_acq)
    support[index_param_to_exclude_giving_highest_det_fim] = False
    select_n_acq = np.sum(support)
    select_b = b[support]
    select_delta = td[support]
    # Append the support to the successive support
    scores.append(select_det_fim)
    successive_support.append(np.copy(support))
    successive_n_acq.append(select_n_acq)
    successive_b.append(select_b)
    successive_td.append(select_delta)
# Concatenate all the arrays in successive_det_fim_excluding_param_k. Do the same for successive_dfepk_indices
successive_det_fim_excluding_param_k = np.concatenate(successive_det_fim_excluding_param_k)
successive_dfepk_indices = np.concatenate(successive_dfepk_indices)
successive_support = np.array(successive_support)
ranks = 16 - successive_support.sum(axis=0)
kept_pairs = (ranks <= 8)

#Plot the final ranking and the protocol
colors = sns.color_palette("Paired", 12)
ranks = 16 - successive_support.sum(axis=0)
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
        ellipse = Ellipse((x_val, y_val), width, height, edgecolor=colors[7], facecolor=colors[6])
        ax.add_patch(ellipse)
        # Place the number inside the circle (en gras)
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
plt.savefig(f'./C2_complex_fim_optimization.png', dpi=300)
plt.show()


####################################################################################################################################################
# %% Simple figure & Elbow technique
####################################################################################################################################################
# Plot the determinant of the FIM vs number of features
num_features = np.array(successive_n_acq)
log_det_values = np.log10(scores)

# Normalize data for elbow detection
x_norm = (num_features - num_features.min()) / (num_features.max() - num_features.min())
y_norm = (log_det_values - log_det_values.min()) / (log_det_values.max() - log_det_values.min())
# Compute distance from line connecting first and last point
start = np.array([x_norm[0], y_norm[0]])
end = np.array([x_norm[-1], y_norm[-1]])
line_vec = end - start
line_vec /= np.linalg.norm(line_vec)
# Calculate distances from the line
distances = []
for x, y in zip(x_norm, y_norm):
    point = np.array([x, y])
    vec = point - start
    proj = np.dot(vec, line_vec) * line_vec
    orth = vec - proj
    distances.append(np.linalg.norm(orth))
# Select the elbow point
elbow_index = int(np.argmax(distances))
elbow_feature_count = num_features[elbow_index]

# Plot the results
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(successive_n_acq, np.log10(scores))
plt.scatter(successive_dfepk_indices, np.log10(successive_det_fim_excluding_param_k), marker='+', color='red')
plt.axvline(x=elbow_feature_count, color='green', linestyle='--')
# Add a text box with the elbow feature count
plt.text(elbow_feature_count + 0.5, np.log10(scores[elbow_index]) - 1.2,
         f'Elbow at {elbow_feature_count} features', color='green', fontsize=12, ha='left', va='bottom')
plt.xlabel('Number of features')
plt.ylabel('Log$_{10}$ of the determinant of the FIM')
plt.title('Determinant of the FIM vs number of features')
plt.gca().set_axisbelow(True)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.grid(axis='y')
plt.show()

####################################################################################################################################################
# %%
# SHAP: Plot the protocol depending on the threshold, using a slider
####################################################################################################################################################


def plot_protocol(n_acquisitions):
    # Erase the previous plot
    plt.clf()
    # Two subplots, on the right the protocol and on the left the MSE vs number of features with a circle on the selected threshold
    plt.subplot(121)
    # use numpy to find the index of n_acquisitions in successive_n_acq (it is in the dataset)
    nb_feat_index = np.where(np.array(successive_n_acq) == n_acquisitions)[0][0]
    plt.plot(successive_n_acq, np.log10(scores))
    plt.scatter(successive_dfepk_indices, np.log10(successive_det_fim_excluding_param_k), color='red')
    plt.plot(successive_n_acq[nb_feat_index], np.log10(scores[nb_feat_index]), 'o')
    plt.xlabel('Number of features')
    plt.ylabel('Log$_{10}$ of the determinant of the FIM')
    plt.title('Determinant of the FIM vs number of features')

    # Plot the corresponding protocol
    lengths_of_successive_prot = np.array([len(b) for b in successive_b])
    corresponding_index = np.where(lengths_of_successive_prot == n_acquisitions)[0][0]
    plt.subplot(122)
    plt.plot(successive_b[corresponding_index], successive_td[corresponding_index], 'o')
    plt.title('Protocol')
    plt.xlabel('b (ms/µm²)')
    plt.ylabel(r'$\Delta$ (ms)')
    plt.xlim(0, 13)
    plt.ylim(8, 50)
    plt.tight_layout()
    plt.show()


# While sliding the threshold, the protocol will change based on the features selected
interact(plot_protocol, n_acquisitions=widgets.IntSlider(min=2, max=len(b), step=1, value=0))

# %%
