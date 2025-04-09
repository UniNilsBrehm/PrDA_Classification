#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
from config import Config
from utils import load_hdf5_as_dict, norm_min_max
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from plotting_style import PlotStyle


def plot_grouped_boxplot(ax, groups, data_condition1, data_condition2, group_spacing=3, box_width=0.6,
                         colors=('lightblue', 'lightgreen'), jitter=0.05, scatter_color='grey', scatter_alpha=0.7,
                         scatter_size=20):
    """
    Plots a grouped boxplot with overlaid individual data points using matplotlib.

    Parameters:
        ax: axis for plotting
        groups (list of str): Labels for each group.
        data_condition1 (list of arrays): A list where each element is an array of data for condition 1 of a group.
        data_condition2 (list of arrays): A list where each element is an array of data for condition 2 of a group.
        group_spacing (float): The spacing between groups on the x-axis.
        box_width (float): The width of each box in the boxplot.
        colors (tuple): A tuple of two color strings; the first for condition 1 boxes, the second for condition 2 boxes.
        jitter (float): The standard deviation of horizontal jitter for the scatter points.
        scatter_color (str): Color for the individual data points.
        scatter_alpha (float): Transparency for the scatter points.
        scatter_size (int): Size of the scatter points.

    Returns:
        fig, ax: The matplotlib Figure and Axes objects.
    """
    # Combine the data for each group and set up x-axis positions.
    all_data = []
    positions = []  # x-axis positions for each box
    tick_positions = []  # positions for group labels

    for i, (d1, d2) in enumerate(zip(data_condition1, data_condition2)):
        base = i * group_spacing
        all_data.append(d1)
        all_data.append(d2)
        # Place condition 1 at base + 1 and condition 2 at base + 2.
        positions.append(base + 1)
        positions.append(base + 2)
        # Place the group label in the center between the two boxes.
        tick_positions.append(base + 1.5)

    # Create the boxplot with patch_artist=True to allow box face coloring.
    box = ax.boxplot(
        all_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False
    )

    # Prepare colors for each box by repeating the tuple for each group.
    box_colors = list(colors) * len(groups)

    # Customize the boxes.
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Customize whiskers and medians.
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    for median in box['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    # Overlay individual data points with horizontal jitter.
    for pos, dataset in zip(positions, all_data):
        x = np.random.normal(loc=pos, scale=jitter, size=len(dataset))
        ax.scatter(x, dataset, alpha=scatter_alpha, color=scatter_color, s=scatter_size)

    # Set x-axis ticks and labels.
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(groups)
    # ax.set_xlabel('Groups')
    # ax.set_ylabel('Values')
    # ax.set_title('Grouped Boxplot with Overlaid Data Points')
    #
    #


def score_histogram(ax, data, data_motor):
    # Histogram with percentage normalization
    ax.hist(data, bins=20, cumulative=False, density=False, alpha=0.3, color='blue', edgecolor='black',
             weights=np.ones(len(data)) / len(data))

    ax.hist(data_motor, bins=20, cumulative=False, density=False, alpha=0.3, color='red', edgecolor='black',
            weights=np.ones(len(data)) / len(data))

    # Convert y-axis to percentage format
    from matplotlib.ticker import PercentFormatter
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))


def histogram_scores(data, label, th, x_lim):
    from matplotlib.ticker import PercentFormatter
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(f'Stimulus: {label}', fontsize=14)
    h = axs[0].hist(data, bins=20, cumulative=False, density=False, histtype='step', weights=np.ones(len(data)) / len(data))
    axs[0].plot([th, th], [0, np.max(h[0])], 'r--')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlim([0, x_lim])
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))

    # Cumulative Histogram
    axs[1].hist(data, bins=20, cumulative=True, density=True, histtype='step')
    axs[1].plot([th, th], [0, 1], 'r--')
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Cumulative Frequency')
    axs[1].set_xlim([0, x_lim])
    axs[1].yaxis.set_major_formatter(PercentFormatter(1))
    return fig


def k_means_benchmark(data, n_clusters):
    from sklearn.cluster import KMeans
    sh_scores = []
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(data)
        cluster_labels = kmeans.labels_
        # cluster_centers = kmeans.cluster_centers_
        try:
            sh_scores.append(silhouette_score(data, cluster_labels))
        except ValueError:
            sh_scores.append(-1)
    return sh_scores


def compute_pca(data, labels=0, wh=False, show=True):
    pca = PCA(n_components=3, whiten=wh)
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    components = pca.components_
    # cmap = color_maps.get_cmap('PiYG', 11)  # 11 discrete colors (from pylab)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if labels is int():
        s_plot = ax.scatter(components[0], components[1], components[2])
    else:
        s_plot = ax.scatter(components[0], components[1], components[2], c=labels, cmap='tab10')
        plt.colorbar(s_plot)
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2f})')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2f})')
    ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.2f})')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    if labels is int():
        s_plot2 = ax2.scatter(components[0], components[1])
    else:
        s_plot2 = ax2.scatter(components[0], components[1], c=labels, cmap='tab10')
        plt.colorbar(s_plot2)
    ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2f})')
    ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2f})')
    if show:
        plt.show()


def compute_linkage_matrix(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


def neurons_names_per_cluster(data_frame, labels):
    data = data_frame.copy()
    clusters = np.unique(labels)
    clusters_dict = dict().fromkeys(clusters)
    for cl in clusters:
        idx = np.where(labels == cl)[0]
        roi_names = data.iloc[idx, :].index
        clusters_dict[cl] = list(roi_names)

    return clusters_dict


def agglomerative_clustering_benchmark_th(data, ths):
    sh_scores = []
    db_score = []
    ch_score = []
    n_clusters = []
    for n in ths:
        clustering = AgglomerativeClustering(
            distance_threshold=n,
            n_clusters=None,
            metric='euclidean',
            # metric='correlation',
            linkage='ward',
            # linkage='average',
        ).fit(data)
        cluster_labels = clustering.labels_
        try:
            sh_scores.append(silhouette_score(data, cluster_labels))
            db_score.append(davies_bouldin_score(data, cluster_labels))
            ch_score.append(calinski_harabasz_score(data, cluster_labels))
        except ValueError:
            sh_scores.append(0)
            db_score.append(0)
            ch_score.append(0)
        n_clusters.append(np.unique(cluster_labels).shape[0])

    return sh_scores, db_score, ch_score, n_clusters


def thresholding_scores(data, th, embedding, n_components=2):
    high_score_mask = data > th
    if n_components > 2:
        result = [embedding[high_score_mask, 0], embedding[high_score_mask, 1], embedding[high_score_mask, 2]]
    else:
        result = [embedding[high_score_mask, 0], embedding[high_score_mask, 1]]

    return result


def plot_t_sne(data, threshold, stim_col='motor_spontaneous'):
    from sklearn.manifold import TSNE

    # Initialize t-SNE (you can tweak perplexity and learning_rate)
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=1000, n_jobs=-1)
    tsne_embedding = tsne.fit_transform(data)

    # Grating 0
    grating_0 = thresholding_scores(data['grating_0'], threshold, tsne_embedding)
    grating_appears = thresholding_scores(data['grating_appears'], threshold, tsne_embedding)

    # Plot t-SNE result
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], color='black', label='All', s=10, alpha=0.2)
    plt.scatter(grating_0[0], grating_0[1], color='red', label='High Grating 0', s=15, alpha=1)
    plt.scatter(grating_appears[0], grating_appears[1], color='green', label='High Grating Appears', s=10, alpha=1)

    plt.legend()
    plt.title('t-SNE Projection of Neuron Stimulus Responses')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()


def plot_t_sne_3d(data, threshold):
    from sklearn.manifold import TSNE

    # Initialize t-SNE for 3D
    tsne = TSNE(n_components=3, perplexity=20, random_state=42, max_iter=1000, n_jobs=-1)
    tsne_embedding = tsne.fit_transform(data)

    # Apply thresholding (assumes thresholding_scores returns 3D coords)
    grating_0 = thresholding_scores(data['grating_0'], threshold, tsne_embedding, n_components=3)
    grating_appears = thresholding_scores(data['grating_appears'], threshold, tsne_embedding, n_components=3)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], tsne_embedding[:, 2],
               color='black', label='All', s=10, alpha=0.2)
    ax.scatter(grating_0[0], grating_0[1], grating_0[2],
               color='red', label='High Grating 0', s=15, alpha=1)
    ax.scatter(grating_appears[0], grating_appears[1], grating_appears[2],
               color='green', label='High Grating Appears', s=10, alpha=1)

    ax.set_title('3D t-SNE Projection of Neuron Stimulus Responses')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_pca(data, threshold, stim_col='motor_spontaneous'):
    from sklearn.decomposition import PCA
    # Initialize PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(data)

    high_score_mask = data[stim_col] > threshold

    # Plot PCA result
    plt.figure(figsize=(8, 6))
    # plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], s=10, alpha=0.7)
    plt.scatter(pca_embedding[~high_score_mask, 0], pca_embedding[~high_score_mask, 1],
                color='blue', label='Low Score', s=10, alpha=0.6)
    plt.scatter(pca_embedding[high_score_mask, 0], pca_embedding[high_score_mask, 1],
                color='red', label='High Score', s=10, alpha=0.8)
    plt.title('PCA Projection of Neuron Stimulus Responses')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


def plot_umap(data, threshold):
    import umap
    """
    Parameter	Default
    n_neighbors	15	Controls local vs. global structure	Lower = fine details; Higher = big picture
    min_dist	0.1	Controls closeness of points in low-D space	Lower = tighter clusters; Higher = more spread out
    n_components	2	Number of output dimensions	Use 2 for plots, 3 for 3D; higher for modeling
    metric	'euclidean'	Distance metric used	Try 'cosine', 'manhattan', etc., for different results
    random_state	None	Random seed for reproducibility	Set to integer for consistent results
"""
    reducer = umap.UMAP(
        n_neighbors=20,      # consider 30 nearest neighbors
        min_dist=0.01,       # allow tight clustering
        n_components=2,      # 2D embedding
        metric='euclidean',  # or 'cosine' for angle-based
        # metric='correlation',  # or 'cosine' for angle-based
        # random_state=42      # reproducibility
    )

    # reducer = umap.UMAP()  # You can tweak parameters here
    embedding = reducer.fit_transform(data)

    # Create a mask for high scorers
    groups = dict()
    for s_type in data:
        groups[s_type] = thresholding_scores(data[s_type], threshold, embedding)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], color='black', label='All', s=15, alpha=0.2)
    for s_type in groups:
        plt.scatter(groups[s_type][0], groups[s_type][1], label=f'High {s_type}', s=10, alpha=0.8)

    plt.legend()
    plt.title('UMAP Projection of Neuron Stimulus Responses')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


def pca_loadings(data):
    from sklearn.decomposition import PCA
    # Fit PCA
    pca = PCA(n_components=3)
    pca_embedding = pca.fit_transform(data)

    # Get loadings (components)
    loadings = pd.DataFrame(
        pca.components_.T,  # Transpose to align features with PCs
        index=data.columns,  # Stimulus regressor names
        columns=['PC1', 'PC2', 'PC3']
    )

    explained_var = pca.explained_variance_ratio_
    # Display top contributing regressors per PC
    print("PCA Loadings:")
    print(loadings)

    # Optional: Show absolute values to find strongest influences
    print("\nTop contributors to PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(5))

    print("Explained Variance Ratio:", explained_var)

    return loadings, pca_embedding, explained_var


def plot_pca_loadings2(loadings, pca_embedding, explained_var, show_rois=False, save_fig=False):
    # PCA already fit; loadings and pca_embedding already defined
    # loadings: DataFrame with PC1–PC3 as columns, rows = regressors
    # pca_embedding: numpy array of shape (360, 3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Bar plot: PC1 loadings ---
    loadings['PC1'].sort_values().plot(kind='barh', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title(f'PC1 Loadings (explained var: {explained_var[0]:.3f})')
    axes[0, 0].set_xlabel('Loading Value')

    # --- Bar plot: PC2 loadings ---
    loadings['PC2'].sort_values().plot(kind='barh', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title(f'PC2 Loadings (explained var: {explained_var[1]:.3f})')
    axes[0, 1].set_xlabel('Loading Value')

    # --- Scatter plot: neurons in PC1 vs PC2 space ---
    scatter = axes[1, 0].scatter(pca_embedding[:, 0], pca_embedding[:, 1],
                                 c=pca_embedding[:, 2], cmap='viridis', s=10)
    axes[1, 0].set_title('Neurons in PCA Space (PC1 vs PC2)')
    axes[1, 0].set_xlabel('PC1 Score')
    axes[1, 0].set_ylabel('PC2 Score')
    cbar = fig.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('PC3 Score')

    # --- Bar plot: PC3 loadings ---
    loadings['PC3'].sort_values().plot(kind='barh', ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title(f'PC3 Loadings (explained var: {explained_var[2]:.3f})')
    axes[1, 1].set_xlabel('Loading Value')

    # --- Label each neuron with its index ---
    if show_rois:
        for i in range(pca_embedding.shape[0]):
            axes[1, 0].text(pca_embedding[i, 0], pca_embedding[i, 1], str(i),
                            fontsize=6, color='black', alpha=0.6)

        # # Label neurons with high PC1 or PC2 scores
        # for i in range(pca_embedding.shape[0]):
        #     if abs(pca_embedding[i, 0]) > 2 or abs(pca_embedding[i, 1]) > 2:  # threshold example
        #         axes[1, 0].text(pca_embedding[i, 0], pca_embedding[i, 1], str(i),
        #                         fontsize=6, color='black', alpha=0.8)

    plt.tight_layout()
    if save_fig:
        save_dir = f'{Config.BASE_DIR}/figures/new_figures/pca_all_scores'
        plt.savefig(f'{save_dir}.jpg', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_pca_loadings(loadings, pca_embedding, explained_var, show_rois=False, save_fig=False):
    # PCA already fit; loadings and pca_embedding already defined
    # loadings: DataFrame with PC1–PC3 as columns, rows = regressors
    # pca_embedding: numpy array of shape (n_samples, 3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Bar plot: PC1 loadings ---
    loadings['PC1'].sort_values().plot(kind='barh', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title(f'PC1 Loadings (explained var: {explained_var[0]:.3f})')
    axes[0, 0].set_xlabel('Loading Value')

    # --- Bar plot: PC2 loadings ---
    loadings['PC2'].sort_values().plot(kind='barh', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title(f'PC2 Loadings (explained var: {explained_var[1]:.3f})')
    axes[0, 1].set_xlabel('Loading Value')

    # --- Scatter plot: neurons in PC1 vs PC2 space ---
    scatter = axes[1, 0].scatter(pca_embedding[:, 0], pca_embedding[:, 1],
                                 c=pca_embedding[:, 2], cmap='viridis', s=10)
    axes[1, 0].set_title('Neurons in PCA Space (PC1 vs PC2)')
    axes[1, 0].set_xlabel('PC1 Score')
    axes[1, 0].set_ylabel('PC2 Score')
    cbar = fig.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('PC3 Score')

    # --- Add loading vectors to PC1 vs PC2 plot ---
    arrow_scale = 10  # adjust for visual clarity
    for feature in loadings.index:
    # for feature in ['grating_0', 'bright_loom', 'moving_target', 'motor_spontaneous']:
        x = loadings.loc[feature, 'PC1']
        y = loadings.loc[feature, 'PC2']
        axes[1, 0].arrow(0, 0, x * arrow_scale, y * arrow_scale,
                        color='red', alpha=0.8, head_width=0.05, length_includes_head=True)
        axes[1, 0].text(x * arrow_scale * 1.1, y * arrow_scale * 1.1, feature,
                        color='red', ha='center', va='center', fontsize=8)

    # --- Bar plot: PC3 loadings ---
    loadings['PC3'].sort_values().plot(kind='barh', ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title(f'PC3 Loadings (explained var: {explained_var[2]:.3f})')
    axes[1, 1].set_xlabel('Loading Value')

    # --- Optional neuron labels ---
    if show_rois:
        for i in range(pca_embedding.shape[0]):
            axes[1, 0].text(pca_embedding[i, 0], pca_embedding[i, 1], str(i),
                            fontsize=6, color='black', alpha=0.6)

    plt.tight_layout()

    # --- Save or show ---
    if save_fig:
        save_dir = f'{Config.BASE_DIR}/figures/new_figures/pca_all_scores'
        plt.savefig(f'{save_dir}.jpg', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_pca_loadings_vectors(loadings, pca_embedding, explained_var, fig_name, save_fig=False, show_labels=False, show_vectors=True):
    # PCA already fit; loadings and pca_embedding already defined
    # loadings: DataFrame with PC1–PC3 as columns, rows = regressors
    # pca_embedding: numpy array of shape (n_samples, 3)

    fig, axes = plt.subplots(figsize=(10, 8))
    # --- Scatter plot: neurons in PC1 vs PC2 space ---
    scatter = axes.scatter(pca_embedding[:, 0], pca_embedding[:, 1],
                                 c=pca_embedding[:, 2], cmap='viridis', s=10)
    axes.axvline(x=0, ymin=0, ymax=1, color='black', ls='--')
    axes.axhline(y=0, xmin=0, xmax=1, color='black', ls='--')
    axes.set_xlabel(f'PC1 Score (explained var: {explained_var[0]:.3f})', fontsize=14)
    axes.set_ylabel(f'PC2 Score (explained var: {explained_var[1]:.3f})', fontsize=14)
    cbar = fig.colorbar(scatter, ax=axes)
    cbar.set_label(f'PC3 Score (explained var: {explained_var[2]:.3f})', fontsize=14)

    # --- Add loading vectors to PC1 vs PC2 plot ---
    arrow_scale = 10  # adjust for visual clarity
    for feature in loadings.index:
        x = loadings.loc[feature, 'PC1']
        y = loadings.loc[feature, 'PC2']
        if show_vectors:
            axes.arrow(0, 0, x * arrow_scale, y * arrow_scale,
                            color='red', alpha=0.8, head_width=0.10, length_includes_head=True)
        if show_labels:
            axes.text(x * arrow_scale * 1.1, y * arrow_scale * 1.1, feature,
                            color='red', ha='center', va='center', fontsize=12)

    plt.tight_layout()

    # --- Save or show ---
    if save_fig:
        save_dir = f'{Config.BASE_DIR}/figures/new_figures/{fig_name}'
        plt.savefig(f'{save_dir}.jpg', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_corr_matrix(data, data_name, save_fig=False):
    import seaborn as sns

    corr_matrix = data.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, center=0, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix of Regressors Scores')
    plt.tight_layout()

    if save_fig:
        save_dir = f'{Config.BASE_DIR}/figures/new_figures/correlation_matrix_{data_name}'
        plt.savefig(f'{save_dir}.jpg', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def get_residuals(data, drop_motor=True):
    """
    Replace the normal regressor with a motor-cleaned version by using a OLS with the motor regressor.

    Parameters
    ----------
    data
    drop_motor

    Returns data
    -------

    """
    import statsmodels.api as sm

    regressor_pairs = [
        ('grating_0', 'grating_0_motor'),
        ('grating_180', 'grating_180_motor'),
        ('grating_appears', 'grating_appears_motor'),
        ('grating_disappears', 'grating_disappears_motor'),
        ('bright_flash', 'bright_flash_motor'),
        ('dark_flash', 'dark_flash_motor'),
        ('bright_loom', 'bright_loom_motor'),
        ('dark_loom', 'dark_loom_motor'),
        ('moving_target', 'moving_target_motor'),

    ]

    for reg1, reg2 in regressor_pairs:
        if reg1 in data.columns and reg2 in data.columns:
            x = sm.add_constant(data[reg2])
            model = sm.OLS(data[reg1], x).fit()
            residuals = model.resid
            r2 = model.rsquared
            cf = model.params.iloc[1]
            print('')
            print(f'{reg1}, {reg2}: r2={r2:.3f},  slope={cf:.3f}')
            compare_histograms(data[reg1], data[reg2], residuals, reg_name=reg1)
            plot_scatter(data, reg1, reg2, model, x)

            # if reg1 == 'bright_loom':
            #     embed()
            #     exit()

            data[reg1] = residuals  # Replace original reg1 with residualized version
            if drop_motor:
                data.drop(columns=[reg2], inplace=True)  # Drop motor regressor
                print(f"Residualized {reg1} from {reg2} and dropped {reg2}.")
            else:
                print(f"Residualized {reg1} from {reg2}, keeping {reg2}.")
        else:
            print(f"Skipped pair ({reg1}, {reg2}) — one or both not in data.")
    return data


def plot_scatter(data, reg1, reg2, model, x):
    fig = plt.figure(figsize=(8, 8))
    th = 0.09
    # Generate fit line over range of reg2 values
    x_vals = np.linspace(data[reg2].min(), data[reg2].max(), 100)
    x_fit = sm.add_constant(x_vals)
    y_fit = model.predict(x_fit)

    # Plot regression line
    plt.scatter(data[reg2], data[reg1])
    plt.plot(x_vals, y_fit, color='red', label='Fitted Line')
    plt.axvline(th, color='black', linestyle='--', linewidth=2)
    plt.axhline(th, color='black', linestyle='--', linewidth=2)

    # Residual lines for each point
    fitted_vals = model.predict(x)
    for i in range(len(data)):
        plt.plot([data[reg2].iloc[i], data[reg2].iloc[i]],
                 [data[reg1].iloc[i], fitted_vals.iloc[i]],
                 color='gray', alpha=0.4)

    # Labels and legend
    plt.xlabel(reg2)
    plt.ylabel(reg1)
    plt.title(f'{reg1} vs {reg2} with Regression Line (R2={model.rsquared:.3f})')

    plt.tight_layout()
    save_dir = f'{Config.BASE_DIR}/figures/new_figures/scatter_{reg1}'
    plt.savefig(f'{save_dir}.jpg', dpi=300)
    plt.close(fig)


def compare_histograms(reg1, reg2, residual, reg_name):
    fig = plt.figure(figsize=(12, 4))
    th = 0.09
    x_min = min(min(reg1), min(reg2), min(residual))
    x_max = max(max(reg1), max(reg2), max(residual))

    # Histogram of original reg1
    plt.subplot(1, 3, 1)
    plt.hist(reg1, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(th, color='red', linestyle='--', linewidth=2)
    plt.title(f'Original: {reg_name}')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.xlim((x_min, x_max))

    # Histogram of motor regressor reg2
    plt.subplot(1, 3, 2)
    plt.hist(reg2, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(th, color='red', linestyle='--', linewidth=2)
    plt.title(f'Motor')
    plt.xlabel('Score')
    plt.xlim((x_min, x_max))

    # Histogram of residuals (motor-free reg1)
    plt.subplot(1, 3, 3)
    plt.hist(residual, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    plt.axvline(th, color='red', linestyle='--', linewidth=2)
    plt.title(f'Residual (Motor Removed)')
    plt.xlabel('Residual Score')
    plt.xlim((x_min, x_max))

    plt.tight_layout()
    save_dir = f'{Config.BASE_DIR}/figures/new_figures/hist_{reg_name}'
    plt.savefig(f'{save_dir}.jpg', dpi=300)
    plt.close(fig)
    # plt.show()


def z_score(df, axis=0):
    return (df - df.mean(axis=axis)) / df.std(axis=axis)


def plot_cluster_map(data, sw_colors, min_max=(-1, 5)):
    styles = PlotStyle()
    g = sns.clustermap(
        data,
        figsize=(15, 8),
        z_score=1,
        cmap=styles.cmap_default,
        method='average',
        metric='euclidean',
        vmin=min_max[0],
        vmax=min_max[1],
        row_cluster=False,
        col_colors=sw_colors,
        dendrogram_ratio=(.1, .1),
        # cbar_pos=(.005, .2, .03, .4),
        # cbar_kws={'label': 'Score [SD]'},
        cbar_pos=None
    )

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=10, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=14)

    g.fig.subplots_adjust(right=0.8, bottom=0.2)  # tweak as needed
    g.ax_heatmap.set_xticklabels([])
    # g.ax_heatmap.set_xlabel("Neurons")
    g.ax_heatmap.tick_params(axis='x', bottom=False, labelbottom=False)

    # # Suppose neuron_names is a list of your custom x-axis labels, in the same order as data.columns
    # g.ax_heatmap.set_xticklabels(roi_names, rotation=90, fontsize=8)
    # g.ax_heatmap.set_xlabel("Neurons")  # Optional axis label


def main():
    styles = PlotStyle()
    # scoring_data = pd.read_csv(Config.linear_scoring_file)
    scoring_data = pd.read_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv')

    # Remove multiple entries: Motor Spontaneous and Moving Target 01 and 02
    # There are mean values for these already present
    idx = scoring_data['reg'].isin(['motor_spontaneous', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor', 'moving_target_02_motor'])
    scoring_data_filtered = scoring_data[~idx].reset_index(drop=True)

    # Pivot data frame to get a long table format for the Scores
    values = 'score'
    df = scoring_data_filtered.pivot(index=['roi', 'sw'], columns='reg', values=values).reset_index()

    # Rename the columns to remove the _MEAN name
    df = df.rename(columns={
        'moving_target_MEAN': 'moving_target',
        'motor_spontaneous_MEAN': 'motor_spontaneous',
        'moving_target_motor_MEAN': 'moving_target_motor'
    })

    # Change index to roi names
    # df.index = df['roi']
    roi_names = df['roi']

    scores_org = df.drop(columns=['roi', 'sw']).reset_index(drop=True)
    scores_org.columns.name = None

    # Generate color code for sweep numbers
    sweeps = df['sw'].rename('sweeps')
    unique_sweeps = sweeps.unique()

    # Generate a color palette with as many colors as there are species
    palette = sns.color_palette("hsv", len(unique_sweeps))  # or "tab20", "Set2", etc.

    # Create the lookup table
    lut = dict(zip(unique_sweeps, palette))

    # Map species to their corresponding colors
    sw_colors = sweeps.map(lut)

    # # Plot as a horizontal color bar
    # colors = sw_colors.tolist()  # Or: np.array(list(row_colors))
    # fig, ax = plt.subplots(figsize=(8, 1))
    # ax.imshow([colors], aspect='auto')
    # ax.set_axis_off()
    # plt.title("Species Color Strip", fontsize=10)
    # plt.show()

    # Keep only rows where at least one value is greater than the threshold
    # scores = scores_org[~scores_org.gt(0.2).any(axis=1)]
    scores = scores_org.copy()

    # ==================================================================================================================
    # Combine highly correlated regressors (e.g. grating_0 and grating_0_motor)
    # plot_corr_matrix(scores, 'all_regs', save_fig=True)
    scores_combined = scores.copy()
    combine_stimuli = ['grating_0', 'dark_loom', 'dark_flash', 'grating_disappears']
    for s_type in combine_stimuli:
        scores_combined[f'{s_type}_combined'] = scores_combined[[s_type, f'{s_type}_motor']].mean(axis=1)
        scores_combined.drop([s_type, f'{s_type}_motor'], axis=1, inplace=True)

    # Plot Regressor Correlation Matrix
    # plot_corr_matrix(scores_combined, 'combined_regs', save_fig=True)

    # ==================================================================================================================
    # Get Residuals for motor-unbiased visual scores
    scores_corrected = get_residuals(scores.copy(), drop_motor=False)
    scores_corrected_no_motor = get_residuals(scores.copy(), drop_motor=True)
    scores_corrected_z = (scores_corrected - scores_corrected.mean()) / scores_corrected.std()

    # ==================================================================================================================
    # Separate High and Low Motor Spontaneous and remove highly correlated (visual - motor) regressors
    scores_filtered = scores.drop(columns=[
        'grating_0', 'grating_0_motor', 'dark_loom', 'dark_loom_motor', 'grating_disappears',
        'grating_disappears_motor', 'dark_flash', 'dark_flash_motor'
    ])
    idx = scores_filtered['motor_spontaneous'] >= 0.1
    scores_high_spontaneous = scores_filtered[idx].drop(columns=['motor_spontaneous'])
    scores_low_spontaneous = scores_filtered[~idx].drop(columns=['motor_spontaneous'])

    embed()
    exit()
    # ==================================================================================================================
    # Principal Component Analysis

    # PCA REGRESSOR LOADINGS
    loadings, pca_embedding, explained_var = pca_loadings(z_score(scores))
    loadings, pca_embedding, explained_var = pca_loadings(z_score(scores_combined.drop(columns=['motor_spontaneous'])))
    loadings, pca_embedding, explained_var = pca_loadings(z_score(scores_combined))

    loadings, pca_embedding, explained_var = pca_loadings(z_score(scores_filtered))
    loadings, pca_embedding, explained_var = pca_loadings(z_score(scores_corrected_no_motor.drop(columns=['motor_spontaneous'])))

    plot_pca_loadings_vectors(loadings, pca_embedding, explained_var, fig_name='pca_combined_no_labels', save_fig=True, show_labels=False, show_vectors=False)
    # plot_pca_loadings_vectors(loadings, pca_embedding, explained_var, fig_name='pca_all', save_fig=True, show_labels=True)
    plot_pca_loadings(loadings, pca_embedding, explained_var, show_rois=False, save_fig=True)
    # plot_umap(scores_high_spontaneous, threshold=0.4)

    # =======================================q===========================================================================
    # Agglomerative Clustering of Scores

    # All Scores
    # normalize rows: 0 (columns: 1)
    # Rows: Regressors, Columns: Neurons
    # sns.clustermap(scores.transpose(), z_score=0, cmap="vlag", method='average', metric='euclidean')
    plot_cluster_map(scores.transpose(), sw_colors=sw_colors)
    plt.show()

    # Combined Regressors (highly correlated)
    plot_cluster_map(scores_combined.transpose(), sw_colors=sw_colors)
    plt.show()

    # Motor (Residual) Corrected Scores
    plot_cluster_map(scores_corrected.transpose(), sw_colors=sw_colors)
    plt.show()

    # All expect highly correlated Regressors
    plot_cluster_map(scores_filtered.transpose(), sw_colors=sw_colors)
    plt.show()

    plot_cluster_map(scores_filtered.drop(columns=['motor_spontaneous']).transpose(), sw_colors=sw_colors)
    plt.show()

    # High Spontaneous Motor Scores
    plot_cluster_map(scores_high_spontaneous.transpose(), sw_colors=sw_colors)
    plt.show()

    # Low Spontaneous Motor Scores
    plot_cluster_map(scores_low_spontaneous.transpose(), sw_colors=sw_colors)
    plt.show()

    # Residual corrected scores
    plot_cluster_map(scores_corrected_no_motor.transpose(), sw_colors=sw_colors)
    plt.show()

    plot_cluster_map(scores_corrected_no_motor.drop(columns=['motor_spontaneous']).transpose(), sw_colors=sw_colors)
    plt.show()

    # Residual corrected scores and removing highly correlated regressors (motor)
    scores_corrected_filtered = scores_corrected_no_motor.drop(columns=['motor_spontaneous', 'grating_0', 'dark_flash', 'dark_loom', 'grating_disappears'])
    plot_cluster_map(scores_corrected_filtered.transpose(), sw_colors=sw_colors, min_max=(-1, 3))
    plt.show()


if __name__ == "__main__":
    main()
