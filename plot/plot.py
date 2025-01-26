import pandas as pd
import matplotlib.pyplot as plt

# Set font family and sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15
})

# Read the Excel file
df = pd.read_excel('/home/lh/Dowzag_2.0/plot/acc.xlsx', index_col=0)

# Create the line plot
plt.figure(figsize=(10, 4))  # Slightly smaller figure size for tighter layout

# Define different markers and line styles
markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
linestyles = ['--', ':', '-.', '--']  # different dash patterns

# Plot each row as a line
x_positions = range(len(df.columns))
for idx, (index, row) in enumerate(df.iterrows()):
    plt.plot(x_positions, row, 
             marker=markers[idx],
             linestyle=linestyles[idx],
             markersize=8,
             label=index)

# Set x-axis ticks and labels
plt.xticks(x_positions, df.columns)

# Customize the plot
plt.xlabel('Window Size')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot as PDF
plt.savefig('/home/lh/Dowzag_2.0/plot/accuracy_plot.pdf', bbox_inches='tight')
plt.show()


# Read the Excel file
df = pd.read_excel('/home/lh/Dowzag_2.0/plot/mrr.xlsx', index_col=0)

# Create the line plot
plt.figure(figsize=(10, 4))  # Smaller figure size for tighter layout

# Define different markers and line styles
markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
linestyles = ['--', ':', '-.', '--']  # different dash patterns

# Plot each row as a line
x_positions = range(len(df.columns))
for idx, (index, row) in enumerate(df.iterrows()):
    plt.plot(x_positions, row, 
             marker=markers[idx],
             linestyle=linestyles[idx],
             markersize=8,
             label=index)

# Set x-axis ticks and labels
plt.xticks(x_positions, df.columns)

# Customize the plot
plt.xlabel('Window Size')
plt.ylabel('MRR')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot as PDF
plt.savefig('/home/lh/Dowzag_2.0/plot/mrr_plot.pdf', bbox_inches='tight')
plt.show()
