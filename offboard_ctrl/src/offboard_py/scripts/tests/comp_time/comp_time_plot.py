# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np

# # Load the CPU timing data
# with open('cpu_timing_data.pkl', 'rb') as f:
#     cpu_data = pickle.load(f)

# # Load the GPU timing data
# with open('gpu_timing_data.pkl', 'rb') as f:
#     gpu_data = pickle.load(f)

# # Extract batch sizes (assumed to be the same in both files)
# batch_sizes = cpu_data['batch_sizes']

# # Extract timing arrays
# cpu_timing_array = cpu_data['timing_array']
# gpu_timing_array = gpu_data['timing_array']

# # Extract individual timing data in milliseconds
# cpu_torch_times = cpu_timing_array[:, 0] * 1000  # Convert to milliseconds
# cpu_numpy_times = cpu_timing_array[:, 1] * 1000  # Convert to milliseconds
# gpu_torch_times = gpu_timing_array[:, 0] * 1000  # Convert to milliseconds
# gpu_numpy_times = gpu_timing_array[:, 1] * 1000  # Convert to milliseconds

# # Format the numbers to 4 significant digits
# cpu_torch_times = np.around(cpu_torch_times, decimals=4)
# cpu_numpy_times = np.around(cpu_numpy_times, decimals=4)
# gpu_torch_times = np.around(gpu_torch_times, decimals=4)
# gpu_numpy_times = np.around(gpu_numpy_times, decimals=4)

# # Create figure and gridspec layout
# fig = plt.figure(figsize=(14, 6))
# gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 

# # Plotting the data on the left subplot
# ax1 = fig.add_subplot(gs[0])
# ax1.plot(batch_sizes, cpu_torch_times, label="CPU: torch")
# ax1.plot(batch_sizes, cpu_numpy_times, label="CPU: numpy")
# ax1.plot(batch_sizes, gpu_torch_times, label="GPU: torch")
# ax1.plot(batch_sizes, gpu_numpy_times, label="GPU: numpy")

# ax1.set_xlabel('Batch Sizes')
# ax1.set_ylabel('Time (milliseconds)')
# ax1.set_title('Timing Data Comparison')
# ax1.legend()
# ax1.grid(True)

# # Creating the table on the right subplot
# ax2 = fig.add_subplot(gs[1])
# ax2.axis('off')  # Turn off the axis

# # Create a table with the timing data
# table_data = [["Batch Size", "CPU:tc", "CPU:np", "GPU:tc", "GPU:np"]]
# for i in range(len(batch_sizes)):
#     row = [
#         batch_sizes[i],
#         f"{cpu_torch_times[i]:.4g}",
#         f"{cpu_numpy_times[i]:.4g}",
#         f"{gpu_torch_times[i]:.4g}",
#         f"{gpu_numpy_times[i]:.4g}"
#     ]
#     table_data.append(row)

# table = ax2.table(cellText=table_data, cellLoc='center', colLabels=None, loc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2)

# # Save the combined figure to a file
# plt.savefig('timing_data_comparison_with_table.png')

# # Optionally, clear the figure after saving
# plt.clf()


import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Load the CPU timing data
with open('cpu_timing_data_2.pkl', 'rb') as f:
    cpu_data = pickle.load(f)

# Load the GPU timing data
with open('gpu_timing_data_2.pkl', 'rb') as f:
    gpu_data = pickle.load(f)

# Extract batch sizes (assumed to be the same in both files)
batch_sizes = cpu_data['batch_sizes']

# Extract timing arrays
cpu_timing_array = cpu_data['timing_array']
gpu_timing_array = gpu_data['timing_array']

# Extract individual timing data in milliseconds
cpu_torch_times = cpu_timing_array[:, 0] * 1000  # Convert to milliseconds
cpu_numpy_times = cpu_timing_array[:, 1] * 1000  # Convert to milliseconds
gpu_torch_times = gpu_timing_array[:, 0] * 1000  # Convert to milliseconds
gpu_numpy_times = gpu_timing_array[:, 1] * 1000  # Convert to milliseconds

# Format the numbers to 4 significant digits
cpu_torch_times = np.around(cpu_torch_times, decimals=4)
cpu_numpy_times = np.around(cpu_numpy_times, decimals=4)
gpu_torch_times = np.around(gpu_torch_times, decimals=4)
gpu_numpy_times = np.around(gpu_numpy_times, decimals=4)

# Set larger font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.titlesize': 18})
plt.rcParams.update({'axes.labelsize': 16})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})
plt.rcParams.update({'legend.fontsize': 14})
# plt.rcParams.update({'table.fontsize': 14})

# Create figure and gridspec layout
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 

# Plotting the data on the left subplot
ax1 = fig.add_subplot(gs[0])
ax1.plot(batch_sizes, cpu_torch_times, label="CPU: torch")
ax1.plot(batch_sizes, cpu_numpy_times, label="CPU: numpy")
ax1.plot(batch_sizes, gpu_torch_times, label="GPU: torch")
ax1.plot(batch_sizes, gpu_numpy_times, label="GPU: numpy")

ax1.set_xlabel('Batch Sizes')
ax1.set_ylabel('Time (milliseconds)')
ax1.set_title('Timing Data Comparison')
ax1.legend()
ax1.grid(True)

# Creating the table on the right subplot
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')  # Turn off the axis

# Create a table with the timing data
table_data = [["BS", "CPU:tc", "CPU:np", "GPU:tc", "GPU:np"]]
for i in range(len(batch_sizes)):
    row = [
        batch_sizes[i],
        f"{cpu_torch_times[i]:.4g}",
        f"{cpu_numpy_times[i]:.4g}",
        f"{gpu_torch_times[i]:.4g}",
        f"{gpu_numpy_times[i]:.4g}"
    ]
    table_data.append(row)

table = ax2.table(cellText=table_data, cellLoc='center', colLabels=None, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(15)
table.scale(1.5,1.5)

# Save the combined figure to a file
plt.savefig('timing_data_comparison_with_table.png')

# Optionally, clear the figure after saving
plt.clf()
