import matplotlib.pyplot as plt
import seaborn as sns


def plot_realizations(grid, realizations, title='', figsize=(4,3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid, realizations[:15].T)
    ax.plot(grid, realizations.mean(axis=0), c='black')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f_{VAE}(x)$')
    ax.set_title(title)

    return fig, ax

def plot_heatmap(matrix, title='', figsize=(8,6)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.heatmap(matrix, annot=False, fmt='g', cmap='coolwarm', ax=ax)
    ax.set_title(title)
    return fig, ax