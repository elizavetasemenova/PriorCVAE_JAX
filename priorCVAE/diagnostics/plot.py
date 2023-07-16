import matplotlib.pyplot as plt
import seaborn as sns

def plot_realizations(grid, realizations, title=''):
    fig, ax = plt.subplots(figsize=(4,3))
    for i in range(15):
        ax.plot(grid, realizations[i,:])
        
    ax.plot(grid, realizations.mean(axis=0), c='black')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f_{VAE}(x)$')
    ax.set_title(title)

    return fig, ax

def plot_covariance_matrix(matrix, title=''):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    sns.heatmap(matrix, annot=False, fmt='g', cmap='coolwarm', ax=ax)
    ax.set_title(title)

    # sns.heatmap(kernel, annot=False, fmt='g', cmap='coolwarm', ax=ax[1])
    # ax[1].set_title(kernel_name)

    return fig, ax