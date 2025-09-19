import os

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
import PyComplexHeatmap as pch
import scanpy as sc
import scipy.ndimage as ndi
import seaborn as sns

from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from venn import generate_petal_labels, draw_venn, generate_colors

def wrap_label(label, width=40):
    parts = label.split(' ')
    half = len(parts) // 2
    return ' '.join(parts[:half]) + '\n' + ' '.join(parts[half:]) if len(parts) > 3 else label

def print_cell_count(adata, group_by = "Sample"):
    # Get the counts for each sample
    sample_counts = adata.obs[group_by].value_counts()
    
    # Get the total number of cells
    total_cells = adata.n_obs
    
    # Print the results in a formatted way
    print("--- Cell Counts ---")
    print("Cells per sample:")
    print(sample_counts)
    print("\n" + "-"*20)
    print(f"Total cells: {total_cells}")
    print("-------------------")

def rerun_PCA(adata):
    try:
        adata = adata.raw.to_adata().copy()
    except AttributeError:
        print("no raw")
    sc.pp.highly_variable_genes(adata, flavor='seurat') # Identify highly variable genes using Seurat's method
    sc.pl.highly_variable_genes(adata) # # Visualize variability to see selected genes
    adata.raw = adata.copy()
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10) # Scale genes to zero mean and unit variance (clip extreme values)
    sc.pp.pca(adata, n_comps=75)  # Recalculate PCA for the first 50 components
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=75)

    return adata.copy()

def plot_composition_barplot(
    adata,
    x_axis_obs: str,
    hue_obs: str,
    normalize_by: str = 'x_axis',
    x_axis_colors: dict = None,
    palette: dict = None,
    x_axis_order: list = None,
    hue_order: list = None,
    figsize: tuple = (10, 5),
    title: str = None,
    ylabel: str = None,
    plot_name: str = None,
    plot_formats: list[str] = ["png"],
    dpi: int = 300
):
    """
    Generates a stacked bar plot showing the percentage composition of one observational
    category within another.

    Args:
        adata (AnnData): The annotated data matrix.
        x_axis_obs (str): The column name in adata.obs to be plotted on the x-axis (e.g., 'Condition').
        hue_obs (str): The column name in adata.obs used for color grouping (e.g., 'leiden').
        normalize_by (str, optional): Determines the normalization axis. Must be 'x_axis' or 'hue'.
            - 'x_axis' (default): For each x-axis category, its bar segments sum to 100%.
              Answers: "What is the cellular composition of each {x_axis_obs}?"
            - 'hue': For each hue category, its values across all x-axis bars sum to 100%.
              Answers: "How is each {hue_obs} distributed across the {x_axis_obs} categories?"
        x_axis_colors (dict, optional): A dictionary mapping x-axis categories to colors. Defaults to None.
        palette (dict, optional): A dictionary mapping hue categories to colors. 
                                     If None, a default seaborn palette is used. Defaults to None.
        x_axis_order (list, optional): A list specifying the order of categories on the x-axis. 
                                       If None, a default order is used. Defaults to None.
        hue_order (list, optional): A list specifying the order of hue categories.
                                    If None, a default sorted order is used. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
        title (str, optional): The title of the plot. Defaults to a generated title.
        ylabel (str, optional): The label for the y-axis. Defaults to a generated label.
        save_path (str, optional): The file path to save the figure. If None, the figure is not saved.
                                   Defaults to None.
        dpi (int, optional): The resolution for the saved figure. Defaults to 300.

    Returns:
        matplotlib.axes.Axes: The Axes object of the plot for further customization.
    """
    # --- 1. Input Validation and Setup ---
    if x_axis_obs not in adata.obs.columns or hue_obs not in adata.obs.columns:
        raise ValueError(f"'{x_axis_obs}' or '{hue_obs}' not found in adata.obs")
    if normalize_by not in ['x_axis', 'hue']:
        raise ValueError("normalize_by must be either 'x_axis' or 'hue'")

    if hue_obs == "leiden":
        adata.obs[hue_obs] = adata.obs[hue_obs].astype(int)

    df = adata.obs[[x_axis_obs, hue_obs]].copy()
    df[hue_obs] = df[hue_obs].astype('category')
    df[x_axis_obs] = df[x_axis_obs].astype('category')

    # Determine order of categories if not provided
    if x_axis_order is None:
        x_axis_order = df[x_axis_obs].cat.categories.tolist()
    if palette is not None:
        hue_order = list(df[hue_obs].cat.categories)
    if hue_order is None:
        try:
            hue_order = sorted(df[hue_obs].cat.categories.astype(float))
        except ValueError:
            hue_order = sorted(df[hue_obs].cat.categories)

    
    # --- 2. Data Processing ---
    # Calculate the number of cells for each combination
    composition_df = df.groupby([x_axis_obs, hue_obs]).size().unstack(fill_value=0)

    # Calculate the percentage based on the 'normalize_by' flag
    if normalize_by == 'x_axis':
        # Each x-axis bar sums to 100%
        composition_percent = composition_df.div(composition_df.sum(axis=1), axis='index') * 100
    else: # normalize_by == 'hue'
        # Each hue category (color) sums to 100% across all x-axis bars
        composition_percent = composition_df.div(composition_df.sum(axis=0), axis='columns') * 100

    # Ensure the DataFrame is ordered correctly
    composition_percent = composition_percent.loc[x_axis_order, hue_order]
        

    # Melt the DataFrame for seaborn
    melted_df = composition_percent.reset_index().melt(
        id_vars=x_axis_obs,
        var_name=hue_obs,
        value_name='Percent'
    )
    melted_df[hue_obs] = melted_df[hue_obs].astype('category')

    # --- 3. Plotting ---
    if palette is None:
        palette = sns.color_palette(n_colors=len(hue_order))
    else:
        palette = [palette[cat] for cat in hue_order]

    plt.figure(figsize=figsize)
    
    ax = sns.barplot(
        data=melted_df,
        x=x_axis_obs,
        y='Percent',
        hue=hue_obs,
        palette=palette,
        order=x_axis_order,
        hue_order=hue_order,
        estimator=sum 
    )

    # --- 4. Final Touches ---
    if title is None:
        title = f'Composition of {x_axis_obs} by {hue_obs}'
    if ylabel is None:
        if normalize_by == 'x_axis':
            ylabel = f'% of Cells in {x_axis_obs}'
        else: # normalize_by == 'hue'
            ylabel = f'% of Cells in {hue_obs}'

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(x_axis_obs, fontsize=12)
    
    plt.legend(title=hue_obs, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure if a path is provided
    
    if plot_name is not None:
        save_dir = "figures/"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f'{save_dir}compositionbarplot_{plot_name}_goi.{plot_format}', dpi = dpi, bbox_inches='tight')

    plt.show()
    
    return ax

def plot_nebulosa(
    adata, 
    gois: list[str], 
    plot_name: str = None, 
    plot_formats: list[str] = ["png"],
    show_plot = False,
    size = 4
):
    for goi in gois:
        try:
            goi_fig = sc.pl.umap(
                adata, 
                color=goi, 
                size = 12, 
                return_fig = True, 
                add_outline = True, 
                outline_width=(1,1), 
                outline_color = ('black', 'white')
            )
        except KeyError:
            print(goi)
            continue
        
        ax = goi_fig.gca()
        plt.close()
        
        scatter = ax.collections[2] 
        
        # Get positions (x, y) and expression values
        xy_positions = scatter.get_offsets()  # Nx2 array of (x, y) coordinates
        expression_values = scatter.get_array()  # Expression values (color mapping)
        
        # Convert to numpy arrays
        x_coords = xy_positions[:, 0]*100
        y_coords = xy_positions[:, 1]*100
        
        image_size = (2000, 2000)
        binary_image = np.zeros(image_size)
        
        # Mark the positions of x_coords and y_coords in the 2D binary image
        # First, shift coordinates to match the image size
        x_scaled = np.clip(x_coords.astype(int) + 800, 0, image_size[1] - 1)
        y_scaled = np.clip(y_coords.astype(int) + 800, 0, image_size[0] - 1)
        
        # Set the corresponding pixels to 1 (binary) for each coordinate
        binary_image[y_scaled, x_scaled] = 1
        
        # Step 2: Dilate the image by 10 pixels using a square kernel
        dilated_10 = np.ones(image_size)
        dilated_crop = ndi.binary_dilation(binary_image, structure=np.ones((20, 20))).astype(int)
        
        # Step 3: Dilate the image by 3 pixels using a smaller square kernel
        dilated_3 = ndi.binary_dilation(binary_image, structure=np.ones((19, 19))).astype(int)
        
        # Step 4: Subtract the dilated image by 3 pixels from the dilated image by 10 pixels to get the outline
        outline = dilated_10 - dilated_3
        outline = ndi.binary_dilation(outline, structure=np.ones((3, 3))).astype(int)
        
        non_zero_indices = np.where(dilated_crop > 0)
        y_min_crop = non_zero_indices[0].min()
        y_max_crop = non_zero_indices[0].max()
        x_min_crop = non_zero_indices[1].min()
        x_max_crop = non_zero_indices[1].max()
        
        # Define grid size (adjust as needed)
        grid_size = 400 
        
        # Get min/max of UMAP coordinates
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Create bins
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # Digitize coordinates into bins
        x_bin_indices = np.digitize(x_coords, x_bins) - 1
        y_bin_indices = np.digitize(y_coords, y_bins) - 1
        
        # Create an empty grid with background = -1
        heatmap = np.full((grid_size, grid_size), -0.1)
        
        expression_values = expression_values-np.mean(expression_values)
        
        # Accumulate expression values into bins
        for x_idx, y_idx, value in zip(x_bin_indices, y_bin_indices, expression_values):
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:  # Ensure within range
                heatmap[y_idx, x_idx] += value  # Sum values in each bin
        
        # Normalize (optional)
        heatmap1 = np.clip(heatmap, a_min=0, a_max=None)  # Ensure no negative values
        
        sigma = 5  # Smoothing factor
        smoothed_heatmap = gaussian_filter(heatmap1, sigma=sigma)
        
        y_coords = y_coords + np.min(y_coords)*-1
        x_coords = x_coords + np.min(x_coords)*-1
        
        y_coords = (y_coords / np.max(y_coords) * 399).astype(int)
        x_coords = (x_coords / np.max(x_coords) * 399).astype(int)
        
        values = []
        for y_coord, x_coord in zip(y_coords, x_coords):
            values.append(smoothed_heatmap[y_coord,x_coord])
        
        # Extract values at these coordinates
        #values = smoothed_heatmap[x_coords, y_coords]  # NumPy uses row (y) first, then column (x)
        
        # Scatter plot
        plt.figure(figsize=(7, 5))
        plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=size)
        
        # Add colorbar
        plt.colorbar(label="Smoothed Density")
        
        # Labels and title
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"{goi} Expression Density")
        plt.tick_params(left = False, right = False, 
                        labelbottom = False, bottom = False, labelleft = False) 
        
        if plot_name is not None:
            save_dir = "figures/nebulosa/"
            os.makedirs(save_dir, exist_ok=True)
            for plot_format in plot_formats:
                plt.savefig(f'{save_dir}density_{plot_name}_{goi}.{plot_format}', dpi = 300)
                
        if show_plot:
            plt.show()
        else:
            plt.close()
    
        heatmap+=abs(np.min(heatmap))
        
        sigma = 5  # Smoothing factor
        smoothed_heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        y_coords = y_coords + np.min(y_coords)*-1
        x_coords = x_coords + np.min(x_coords)*-1
        
        y_coords = (y_coords / np.max(y_coords) * 399).astype(int)
        x_coords = (x_coords / np.max(x_coords) * 399).astype(int)
        
        values = []
        for y_coord, x_coord in zip(y_coords, x_coords):
            values.append(smoothed_heatmap[y_coord,x_coord])
            
        # Scatter plot
        plt.figure(figsize=(7, 5))
        plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=size)
        
        # Add colorbar
        plt.colorbar(label="Smoothed Expression")
        
        # Labels and title
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"{goi} Expression")
        plt.tick_params(left = False, right = False, 
                        labelbottom = False, bottom = False, labelleft = False)

        if plot_name is not None:
            save_dir = "figures/nebulosa/"
            os.makedirs(save_dir, exist_ok=True)
            for plot_format in plot_formats:
                plt.savefig(f'{save_dir}expression_{plot_name}_{goi}.{plot_format}', dpi = 300)
                
        if show_plot:
            plt.show()
        else:
            plt.close()

def plot_gsea_dotplot(
    adata, 
    ref_cluster: str = "Untreated",
    cluster_group_by: str = "Condition", 
    plot_name: str = None, 
    adj_pval_cutoff: float = 10e-5, 
    log2fc_cutoff: float = 1, 
    use_pathways: int = 10, 
    figure_size: tuple = (5, 4), 
    plot_formats: list[str] = ["png"],
    min_size: int = 5, 
    leg_off: bool = False
):

    clusters = list(set(adata.obs[cluster_group_by]))
    marker_dict = {}
    subset = adata.copy()
    
    print("Computing marker genes...")
    
    if ref_cluster is not None:
        clusters = [cluster for cluster in clusters if cluster != ref_cluster]
        sc.tl.rank_genes_groups(subset, groupby=cluster_group_by, method="t-test", reference=ref_cluster, use_raw=True, key_added="temp")
    else:
        sc.tl.rank_genes_groups(subset, groupby=cluster_group_by, method="t-test", use_raw=True, key_added="temp")
    
    for cluster in clusters:
        print(f"Processing cluster: {cluster}")
        marker = sc.get.rank_genes_groups_df(subset, cluster, key='temp')
        marker['pvals_adj'] = marker['pvals_adj'].fillna(10e-300)
        marker.loc[(marker['logfoldchanges'].isna()) & (marker['scores'] > 0), 'log2fc'] = 10
        marker.loc[(marker['logfoldchanges'].isna()) & (marker['scores'] < 0), 'log2fc'] = -10
        marker = marker[(marker["pvals_adj"] < adj_pval_cutoff) & (abs(marker["logfoldchanges"]) > log2fc_cutoff)]
            
        marker_dict[cluster] = marker
    
    # Run GSEA
    outs = {}
    used_terms = []
    
    print("Running GSEA...")
        
    for cluster in list(marker_dict.keys()):
        res = marker_dict[cluster]
        res['Symbol'] = res.names
        ranking = res[['Symbol', 'scores']].dropna().sort_values('scores', ascending=False, key = abs).drop_duplicates('Symbol')
    
        pre_res = gp.prerank(
            rnk=ranking,
            gene_sets='GO_Biological_Process_2025',
            threads=10,
            min_size=min_size,
            max_size=1000,
            permutation_num=1000,
            outdir=None,
            seed=6,
            verbose=True,
        )
    
        out = [
            [term,
             pre_res.results[term]['fdr'],
             pre_res.results[term]['es'],
             pre_res.results[term]['nes']]
            for term in pre_res.results
        ]

        out_df = pd.DataFrame(out, columns=['Term', 'fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop=True)
        outs[cluster] = out_df
        terms = out_df.head(use_pathways).Term
        used_terms.append(terms)

    # Create combined FDR/NES matrices
    top_terms = sorted(set(term for sublist in used_terms for term in sublist))
    fdr_matrix = pd.DataFrame(index=top_terms, columns=marker_dict.keys())
    nes_matrix = pd.DataFrame(index=top_terms, columns=marker_dict.keys())
    
    for cluster, df in outs.items():
        for _, row in df.iterrows():
            term = row['Term']
            if term in top_terms:
                fdr_matrix.loc[term, cluster] = row['fdr']
                nes_matrix.loc[term, cluster] = row['nes']
    
    # Convert to float
    fdr_matrix = fdr_matrix.astype(float)
    nes_matrix = nes_matrix.astype(float)
    
    # Prepare data for plotting
    fdr_long = fdr_matrix.reset_index().melt(id_vars='index', var_name=cluster_group_by, value_name='FDR')
    nes_long = nes_matrix.reset_index().melt(id_vars='index', var_name=cluster_group_by, value_name='NES')
    plot_df = pd.merge(fdr_long, nes_long, on=['index', cluster_group_by])
    plot_df.rename(columns={'index': 'Term'}, inplace=True)
    
    # Transform and cap values
    plot_df['FDR10'] = -np.log10(plot_df['FDR'])
    # plot_df['FDR10'] = plot_df['FDR10'].replace([np.inf, -np.inf], np.nan).fillna(1).clip(upper=10)
    plot_df['FDR10'] = plot_df['FDR10'].clip(upper=2)
    plot_df['NES'] = plot_df['NES'].clip(-2, 2)
    plot_df['Significant'] = plot_df['FDR'] < 0.01
    
    # Sort clusters by number of enriched terms
    cluster_order = fdr_matrix.notna().sum(axis=0).sort_values(ascending=False).index
    plot_df[cluster_group_by] = pd.Categorical(plot_df[cluster_group_by], categories=cluster_order, ordered=True)
    
    # plot_df['Term'] = plot_df['Term'].apply(wrap_label)
    
    plt.figure(figsize=figure_size)
    plt.grid(True, linestyle='--', linewidth=0.5, zorder = 0)

    dotplot = sns.scatterplot(
        data=plot_df,
        x=cluster_group_by,
        y='Term',
        size='FDR10',
        hue='NES',
        palette='viridis',
        sizes=(20, 200),
        edgecolor='black',
        zorder=5
    )
    
    # Get the legend object created by seaborn
    legend = dotplot.get_legend()
    
    # Iterate through the legend's text elements to format them
    for label in legend.get_texts():
        # Only format labels that can be converted to a number
        try:
            numeric_value = float(label.get_text())
            label.set_text(f"{numeric_value:.2f}")
        except ValueError:
            pass  # Ignore non-numeric labels (like the title)
    
    # Set the title and position of the modified legend
    legend.set_title('NES / -log10(FDR)')
    plt.setp(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    if leg_off:
        plt.legend("")
    
    plt.title(f'Top Enriched GO Terms by {cluster_group_by}')
    plt.suptitle(f"pval_cutoff: {adj_pval_cutoff}, log2fc_cutoff: {log2fc_cutoff}, no pathways: {use_pathways}", fontsize=4)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5)
    plt.tight_layout()
    
    if plot_name is not None:
        save_dir = "figures/"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"{save_dir}GSEAdotplot_{plot_name}_{reference_cluster}_{cluster_group_by}.{plot_format}", dpi = 300)
            
    plt.show()

def plot_volcano(
    adata,
    contrast: list,
    highlight_genes: dict,
    palette: dict,
    groupby: str = "Condition",
    plot_name: str = None,
    p_value_cutoff: float = 0.01,
    log2fc_cutoff: float = 1.0,
    p_value_col: str = "pvals_adj",
    figure_size: tuple = (8, 8),
    x_range: tuple = None,
    label_top_genes: int = 20,
    hard_stop: bool = True,
    font_size: int = 9,
    plot_formats: list[str] = ["png"]
):
    """
    Generates a volcano plot from a DESeq2 results DataFrame.

    Args:
        deseq_results (pd.DataFrame): The DataFrame of results from a DESeq2 analysis.
        contrast (list): A list of the two conditions that were compared, with the
            reference condition first (e.g., ["Untreated", "Treated"]).
        highlight_genes (dict): A dictionary to color specific gene labels.
            Keys are colors, values are lists of gene symbols.
        palette (dict): A dictionary mapping condition names to colors for the plot points.
        plot_name (str): A base name for the output plot files.
        p_value_cutoff (float): The significance threshold for the p-value. Defaults to 0.01.
        log2fc_cutoff (float): The minimum absolute log2 fold change for significance. Defaults to 1.0.
        p_value_col (str): The p-value column to use for filtering. Defaults to "padj".
        figure_size (tuple): The size (width, height) of the output figure. Defaults to (8, 8).
        x_range (tuple): The x-axis (logfoldchanges) limits for the plot. Defaults to None.
        label_top_genes (int): The number of top genes to label. Defaults to 50.
        hard_stop (bool): If True, strictly labels the top N genes. If False, uses
            the custom_slicer logic. Defaults to False.
        font_size (int): The font size for gene labels. Defaults to 9.
        plot_formats (List[str]): A list of file formats to save the plots in. Defaults to ["png"].
    """
    # --- Step 1: Prepare DataFrame for plotting ---
    sc.tl.rank_genes_groups(adata, groupby=groupby, reference=contrast[0], key_added="temp_rank")
    plot_df = sc.get.rank_genes_groups_df(adata, group=contrast[1], key = "temp_rank")

    plot_df[f'-log10({p_value_col})'] = -np.log10(plot_df[p_value_col])

    up_color = palette[contrast[1]]
    down_color = palette[contrast[0]]
    not_sig_color = 'lightgray'

    plot_df['color'] = np.select(
        [
            (plot_df[p_value_col] < p_value_cutoff) & (plot_df['logfoldchanges'] >= log2fc_cutoff),
            (plot_df[p_value_col] < p_value_cutoff) & (plot_df['logfoldchanges'] <= -log2fc_cutoff)
        ],
        [up_color, down_color],
        default=not_sig_color
    )

    plot_df.replace([np.inf, -np.inf], 324, inplace=True)
    plot_df.dropna(subset=['logfoldchanges', f'-log10({p_value_col})'], inplace=True)

    # --- Step 2: Create the plot ---
    plt.figure(figsize=figure_size)
    if x_range:
        plt.xlim(x_range)
    
    sorted_df = plot_df.sort_values(by='color', ascending=False)
    
    sns.scatterplot(
        data=sorted_df,
        x='logfoldchanges',
        y=f'-log10({p_value_col})',
        c=sorted_df['color'],
        edgecolor=None,
        alpha=0.7,
    )

    # --- Step 3: Label significant genes ---
    texts = []
    bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)

    text_df = plot_df[
        (abs(plot_df['logfoldchanges']) >= log2fc_cutoff) & (plot_df[p_value_col] < p_value_cutoff)
    ].copy()
    text_df["plot_stat"] = abs(text_df["scores"])*abs(text_df["logfoldchanges"])
    text_df = text_df.sort_values("plot_stat", key=abs, ascending=False)

    if hard_stop:
        text_df_pos = text_df[text_df["logfoldchanges"] > 0].head(label_top_genes)
        text_df_neg = text_df[text_df["logfoldchanges"] < 0].head(label_top_genes)
    else:
        text_df_pos = custom_slicer(
            text_df[text_df["logfoldchanges"] > 0], 
            initial_rows=label_top_genes, 
            block_size=label_top_genes, 
            step=2)
        text_df_neg = custom_slicer(
            text_df[text_df["logfoldchanges"] < 0], 
            initial_rows=label_top_genes, 
            block_size=label_top_genes, 
            step=2)

    flattened_values = [
        value for key, lst in highlight_genes.items() if key != "black" for value in lst
    ]
    flattened_values = [gene for gene in flattened_values if gene in text_df.index]
    text_df_other = text_df.loc[flattened_values]
    
    text_df = pd.concat([text_df_pos, text_df_neg, text_df_other], ignore_index=True)
    text_df = text_df.drop_duplicates(subset=['names'], keep='first')

    for _, row in text_df.iterrows():
        gene = row['names']
        label_color = "black"
        for color, genes in highlight_genes.items():
            if gene in genes:
                label_color = color
        
        texts.append(plt.text(
            x=row['logfoldchanges'],
            y=row[f'-log10({p_value_col})'],
            s=gene,
            bbox=bbox_props,
            fontsize=font_size,
            weight='bold',
            color=label_color, ha='center', va='center'
        ))
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='#1f1f1f'), force_explode=(1, 4), force_static=(2, 4), time_lim=5)

    # --- Step 4: Finalize and show plot ---
    plt.title(f'Volcano Plot: {contrast[1]} vs {contrast[0]}', fontsize=16)
    plt.suptitle(f"left: {contrast[0]}, right: {contrast[1]}, {p_value_col}: {p_value_cutoff}, genes displayed: {label_top_genes*2}/total: {len(texts)}", fontsize=9)
    plt.xlabel('log2 Fold Change', fontsize=13)
    plt.ylabel(f'-log10({p_value_col})', fontsize=13)
    if int(np.max(sorted_df[f'-log10({p_value_col})'])) == int(324):
        plt.axhline(324, color='black', linestyle='--', label='python limit')
        plt.gca().set_ylim(top=350)
        plt.yticks([0, 50, 100, 150, 200, 250, 300, 324], labels=['0', '50', '100', '150', '200', '250', '300', '> 324'])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"{save_dir}/volcano_{plot_name}_{contrast[0]}vs{contrast[1]}.{plot_format}", dpi=300)
    plt.show()

def plot_expression_heatmap(
    adata,
    groupby: str = 'Condition',
    display_top_genes: int = None,
    label_top_genes: int = 7,
    ref_condition: str = "rest",
    skip_cells: int = 75,
    p_value_cutoff: float = 0.01,
    log2fc_cutoff: float = 1.0,
    p_value_col: str = "pvals_adj",
    highlight_genes: dict = {},
    left_ha_dict: dict = None,
    palette: dict = None,
    plot_name: str = None,
    plot_formats: list[str] = ["png"],
    figure_size: tuple = (12, 10),
    normalize: bool = True,
    plot_not_significant: bool = True,
    abs_gene: bool = True,
    v_range: tuple = (-2,2)
):
    left_ha = {}
    key_name = f"{groupby}_rank"
    
    adata = adata[adata.obs[groupby].notna(), :]
    sc.tl.rank_genes_groups(
        adata, 
        groupby=groupby,
        use_raw=True, 
        method="t-test", 
        key_added=key_name, 
        reference = ref_condition
    )
    
    used_genes = set()
    markers = []
    label_genes = []
    display_genes = []
    grps = np.unique(adata.obs[groupby])
    
    if palette is None:
        set3_colors = sns.color_palette("Set3", n_colors=len(grps))
        set3_colors = dict(zip(grps, set3_colors[:len(grps)]))
    else:
        # If colors were provided, use them
        set3_colors = palette

    if display_top_genes is None:
        display_top_genes = 100000000000

    if (len(grps) == 2 and plot_not_significant):
        plot_not_significant = True
        p_value_cutoff = p_value_cutoff*2
    else:
        plot_not_significant = False
    
    for grp in grps[grps!=ref_condition]:
        # Get ranked genes for the group
        marker = sc.get.rank_genes_groups_df(adata, group=grp, key=key_name)
        
        # Sort by score descending
        if abs_gene:
            marker = marker.sort_values(by='scores', ascending=False, key = abs)
        else:
            marker = marker.sort_values(by='scores', ascending=False)
        
        # Filter out genes already used
        marker = marker[~marker['names'].isin(used_genes)]
        marker = marker[abs(marker["logfoldchanges"])>=log2fc_cutoff]
        
        marker = marker[marker[p_value_col]<p_value_cutoff]

        if plot_not_significant and used_genes == set():
            if left_ha_dict is None:
                left_ha_dict = {}
            left_ha_pval_series = pd.Series(list(marker[p_value_col]), index=list(marker["names"]))
            left_ha_pval_series = left_ha_pval_series.replace([-np.inf], 0)
            
            sig_cmap = LinearSegmentedColormap.from_list(
                    "sig_coolwarm",
                    [(0, "white"), (p_value_cutoff, "yellow"), (p_value_cutoff, "red"), (1.0, "red")]
            )
            
            left_ha_dict["Significance"] = pch.anno_simple(
                left_ha_pval_series,
                cmap = sig_cmap,
                legend=True, 
                add_text=False, 
                text_kws={'fontsize':9,'color':'black'}, 
                height = 10
            )
        
        # Keep top genes
        label_genes.append(marker.head(label_top_genes)["names"].tolist())
        display_genes.append(marker.head(display_top_genes)["names"].tolist())
        
        # Track which genes we've used
        used_genes.update(set(display_genes[0]))
    
    # Combine results
    
    label_genes = [gene for sublist in label_genes for gene in sublist]
    display_genes = [gene for sublist in display_genes for gene in sublist]

    if normalize:
        # Find the size of the smallest group
        min_group_size = adata.obs[groupby].value_counts().min()
        sampled_obs = adata.obs.groupby(groupby).sample(n=min_group_size, random_state=42)
        adata = adata[sampled_obs.index, :].copy()
        
    grapher = pd.DataFrame(sc.get.obs_df(adata.raw.to_adata(), keys=display_genes).T)
    subset = grapher.loc[grapher.index.isin(display_genes)]
    subset = subset.iloc[:,::skip_cells]
    
    # Flatten top_markers (list of lists) into a flat list
    
    expr = subset.copy()
    
    row_colors_dict = {}

    for gene in label_genes:
        row_colors_dict[gene] = 'black'
        
    for color, goi_list in highlight_genes.items():
        for gene in goi_list:
            row_colors_dict[gene]=color

    selected_rows = [gene for sublist in highlight_genes for gene in sublist]
    selected_rows = list(set(label_genes + selected_rows))
    label_rows = expr.apply(lambda x:x.name if x.name in selected_rows else None,axis=1)
    expr = expr[adata.obs[groupby][::skip_cells].sort_values(ascending = True).index]

    if left_ha_dict is not None:
        left_ha = pch.HeatmapAnnotation(
            **left_ha_dict,
            axis=0, 
            plot_legend=True, 
            legend_side='right'
        )
    
    row_ha = pch.HeatmapAnnotation(
        selected=pch.anno_label(
            label_rows,
            colors=row_colors_dict, 
            relpos=(0,0.4), 
            extend = True, 
            merge = True, 
            height = 10
        ),
        axis=0, 
        fontsize=1,
        verbose=0,
        orientation='right'
    )

    set3_colors = {k: v for k,v in set3_colors.items() if k in grps}
    col_ha = pch.HeatmapAnnotation(
        label=pch.anno_label(
            adata.obs[groupby][::skip_cells].sort_values(ascending = True), 
            merge=True, 
            rotation=45, 
            colors=set3_colors
        ),
        Condition=pch.anno_simple(
            adata.obs[groupby][::skip_cells].sort_values(ascending = True), 
            legend=True, 
            colors=set3_colors
        ), 
        label_side='right', 
        axis=1
    )
    
    plt.figure(figsize=figure_size)
    if left_ha != {}:
        cm = pch.ClusterMapPlotter(
            data=expr, 
            z_score=0,
            right_annotation=row_ha, 
            top_annotation=col_ha, 
            left_annotation=left_ha,
            # col_split=adata_multi.obs[groupby][::50].sort_values(ascending = True),
            # col_split_gap=0.25,
            col_cluster=False,
            row_cluster=False,
            label='Expression',
            row_dendrogram=False,
            col_dendrogram=False,
            show_rownames=False,
            show_colnames=False,
            cmap='viridis',
            tree_kws={'row_cmap': 'Dark2'},
            xticklabels_kws={'labelrotation':-45,'labelcolor':'blue'},
            yticklabels_kws = {'labelsize':8}, 
            vmin=v_range[0],
            vmax=v_range[1]
        )
    else:
        cm = pch.ClusterMapPlotter(
            data=expr, 
            z_score=0,
            right_annotation=row_ha, 
            top_annotation=col_ha, 
            # col_split=adata_multi.obs[groupby][::50].sort_values(ascending = True),
            # col_split_gap=0.25,
            col_cluster=False,
            row_cluster=False,
            label='Expression',
            row_dendrogram=False,
            col_dendrogram=False,
            show_rownames=False,
            show_colnames=False,
            cmap='viridis',
            tree_kws={'row_cmap': 'Dark2'},
            xticklabels_kws={'labelrotation':-45,'labelcolor':'blue'},
            yticklabels_kws = {'labelsize':8}, 
            vmin=v_range[0],
            vmax=v_range[1]
        )
    if plot_name is not None:
        save_dir = "figures/"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"figures/{plot_name}_cm_{key_name}_{ref_condition}.{plot_format}", bbox_inches='tight', dpi = 300)

import ExpressLane.bulk as el_bulk

def get_pseudobulk(
    adata,
    available_metadata_columns,
    ref_condition: str = "Untreated",
    contrast_design: str = None,
    design_formula: str = None,
):
    if design_formula is None:
        design_formula = f"~{available_metadata_columns[0]}"
    if contrast_design is None:
        contrast_design = available_metadata_columns[0]
    
    if len(available_metadata_columns) < 2:
        available_metadata_columns = [available_metadata_columns[0], "Replicate"]
        adata.obs['Replicate'] = np.random.choice(["1", "2", "3"], size=len(adata.obs))
        
    adata = adata.raw.to_adata().copy()
    counts_df = pd.DataFrame(index = adata.var_names)
    metadata = pd.DataFrame(index = available_metadata_columns)
    
    conditions = set(adata.obs[available_metadata_columns[0]])
    for cond in conditions:
        condition_adata = adata[adata.obs[available_metadata_columns[0]] == cond]
            
        replicates = set(condition_adata.obs[available_metadata_columns[1]])
        for rep in replicates:
            replicate_adata = condition_adata[condition_adata.obs[available_metadata_columns[1]] == rep]
            expression_sum_per_gene = replicate_adata.X.sum(axis=0)
            expression_sum_per_gene = np.asarray(expression_sum_per_gene).reshape(-1)
            expression_sum_per_gene = [int(x) for x in expression_sum_per_gene]
            counts_df[f"{cond}_{rep}"] = expression_sum_per_gene
            metadata[f"{cond}_{rep}"] = [cond, rep]

    dds, res_dict, contrast_design = el_bulk._run_deseq_analysis(
        counts_df = counts_df.T,
        metadata = metadata.T,
        design_formula = design_formula,
        contrast_design = contrast_design,
        ref_condition = ref_condition
    )

    return metadata, dds, res_dict

def get_venn_data(
    adata,
    ref_condition = 'Untreated',
    adj_pval_cutoff = 10e-5,
    logfc_cutoff = (-1,1),
    group_by = "Condition"
):    
    conditions = set(adata.obs[group_by])
    conditions = [c for c in conditions if c != ref_condition]
    
    marker_venn_up = {}
    marker_venn_down = {}
    marker_dict = {}
    
    top_markers = []
    
    for condition in conditions:
        subset = adata[adata.obs[group_by].isin([condition, ref_condition])].copy()
        sc.tl.rank_genes_groups(subset, groupby=group_by, method="t-test", use_raw = True)

        marker = sc.get.rank_genes_groups_df(subset, condition)
        marker = marker[marker["pvals_adj"] < adj_pval_cutoff]
        
        marker_down = marker[marker["logfoldchanges"] < logfc_cutoff[0]]
        marker_down = marker_down.sort_values(by='scores', ascending=True)
        marker_venn_down[condition] = set(marker_down['names'])
        print(f"Found {len(marker_down)} DEGs for {condition} at pval: {adj_pval_cutoff} and log2fc: {logfc_cutoff[0]}")
        
        marker_up = marker[marker["logfoldchanges"] > logfc_cutoff[1]]
        marker_up = marker_up.sort_values(by='scores', ascending=False)
        marker_venn_up[condition] = set(marker_up['names'])
        print(f"Found {len(marker_up)} DEGs for {condition} at pval: {adj_pval_cutoff} and log2fc: {logfc_cutoff[1]}")
    
    # with open(f'figures/{prefix}_DEGs_per_condition_logfc{logfc_cutoff}.txt', 'w') as file:
    #      file.write(str(marker_venn))
    return (marker_venn_up, marker_venn_down), (ref_condition, adj_pval_cutoff, logfc_cutoff)

def plot_venn(
    venns_to_draw, 
    condition_colors, 
    cutoff_parameters, 
    plot_name, 
    leg_off = False
):
    

    for i, venn_to_draw in enumerate(venns_to_draw):
    
        petal_labels = generate_petal_labels(venn_to_draw.values(), fmt="{size}")
        condition_colors_ordered = {key: condition_colors[key] for key in venn_to_draw if key in condition_colors}
        hex_colors = list(condition_colors_ordered.values())
        
        # Convert to RGBA (with full opacity)
        rgba_colors = [(r, g, b, 0.5) for r, g, b, _ in [mcolors.to_rgba(color) for color in hex_colors]]
        draw_venn(
            petal_labels=petal_labels, dataset_labels=venn_to_draw.keys(),
            hint_hidden=False, colors=rgba_colors,
            figsize=(8, 8), fontsize=25, legend_loc="upper right", ax=None
        )
        
        plt.title("Venn Diagram of DEGs")
        plt.suptitle(f"{list(venn_to_draw.keys())}: adj_pval {cutoff_parameters[1]}, log2fc {cutoff_parameters[2][i]}", size = 7.5)
        plt.savefig(f'figures/venn_{plot_name}_{cutoff_parameters[2][i]}.png')
        if leg_off:
            plt.legend("")
        plt.show()