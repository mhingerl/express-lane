import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PyComplexHeatmap as pch
import gseapy as gp
import scanpy as sc
import os

# Import libraries for analysis
from adjustText import adjust_text
from gprofiler import GProfiler
from itertools import chain, combinations
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pytximport import tximport
from pytximport.utils import create_transcript_gene_map
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import random
from sanbomics.tools import id_map
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from scipy.stats import sem
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from venn import generate_petal_labels, draw_venn, generate_colors

def custom_slicer(df, initial_rows=50, block_size=50, step = 3):
  """
  Selects rows using a dynamic stepping pattern.

  Args:
    df (pd.DataFrame): The input DataFrame.
    initial_rows (int): The number of rows to select from the start.
    block_size (int): The size of each subsequent block for stepped slicing.

  Returns:
    pd.DataFrame: A new DataFrame with the selected rows.
  """
  # Ensure DataFrame is large enough for initial selection
  if len(df) == 0:
    return pd.DataFrame()
    
  # 1. Get the list of initial row indices
  indices = list(range(min(initial_rows, len(df))))

  # 2. Loop to get indices for the following blocks with an increasing step
  start_index = initial_rows
  while start_index < len(df) and step < step * 2:
    end_index = start_index + block_size
    block_indices = range(start_index, min(end_index, len(df)), step)
    indices.extend(block_indices)
    
    # Prepare for the next block
    start_index = end_index
    step += step
  
  # 3. Select all desired rows at once using the final index list
  return df.iloc[indices]

def wrap_text_by_words(text, words_per_line=2):
        """Inserts a newline character into a string after a set number of words."""
        words = text.split()
        if len(words) > words_per_line:
            words_per_line = int(len(words)/2)
            # Join the first part, add a newline, then join the rest
            return ' '.join(words[:words_per_line]) + '\n' + ' '.join(words[words_per_line:])
        return text

def _get_color_palette(keys: List[str]) -> Dict[str, str]:
    """Generates a dictionary of random hex colors for a given list of keys.

    This function is useful for creating color mappings for plotting when a
    pre-defined palette is not available.

    Args:
        keys (List[str]): A list of strings that will be used as keys
                          in the output dictionary (e.g., condition names).

    Returns:
        Dict[str, str]: A dictionary where each key from the input list
                        maps to a randomly generated hex color code (e.g., '#a4c2f4').
    """
    palette = {}
    for key in keys:
        color = f'#{random.randint(0, 0xFFFFFF):06x}'
        
        while color in palette.values():
            color = f'#{random.randint(0, 0xFFFFFF):06x}'
            
        palette[key] = color
        
    return palette

def _run_deseq_analysis(counts_df, metadata, design_formula, contrast_design, ref_condition):
    """Sets up and runs the full DESeq2 analysis pipeline."""
    inference = DefaultInference(n_cpus=20)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design=design_formula,
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()

    conditions = list(set(dds.obs[contrast_design]))
    if ref_condition not in conditions:
        print(f"Please define valid ref_condition. Available: conditions")
        
    conditions = [c for c in conditions if c != ref_condition]

    res_dict = {}
    for condition in conditions:
        contrast = [contrast_design, condition, ref_condition]
        stat_model = DeseqStats(dds, contrast=contrast)
        stat_model.summary()
        res = stat_model.results_df
        res = res[res.baseMean >= 10]
        res['Symbol'] = res.index
        res_dict[condition] = res
        
    return dds, res_dict, contrast_design


def _get_significant_genes(res_dict, 
                           p_value_cutoff, 
                           p_value_col, 
                           log2fc_cutoff, 
                           contrast_design, 
                           primary_column_order, 
                           display_top_genes):
    """Aggregates and filters significant genes from all contrasts."""
    all_sig_genes = []
    for condition, res_df in res_dict.items():
        # Get top up-regulated genes
        sigs_up = res_df[res_df[p_value_col] < p_value_cutoff].copy()
        sigs_up = sigs_up[sigs_up["log2FoldChange"]>=log2fc_cutoff]
        sigs_up = sigs_up.sort_values('stat', ascending=False).head(display_top_genes)
        sigs_up[contrast_design] = condition
        all_sig_genes.append(sigs_up)

        # Get top down-regulated genes
        sigs_down = res_df[res_df[p_value_col] < p_value_cutoff].copy()
        sigs_down = sigs_down[sigs_down["log2FoldChange"]<=-log2fc_cutoff]
        sigs_down = sigs_down.sort_values('stat', ascending=True).head(display_top_genes)
        sigs_down[contrast_design] = condition
        all_sig_genes.append(sigs_down)

    if not all_sig_genes:
        return pd.DataFrame()

    # Combine, sort, and remove duplicates to get the final gene list
    combined_df = pd.concat(all_sig_genes, ignore_index=True)
    combined_df = combined_df.sort_values('stat', ascending=False)
    combined_df = combined_df.drop_duplicates(subset='Symbol', keep='first')
    combined_df[contrast_design] = pd.Categorical(
        combined_df[contrast_design], categories=primary_column_order, ordered=True
    )
    combined_df = combined_df.sort_values(contrast_design).reset_index(drop=True)
    
    return combined_df

def generate_volcano_data(
    metadata: pd.DataFrame,
    conditions: list,
    design_formula: str = "~Condition",
    contrast_design = "Condition",
    data_base_path: str = "..",
    transcriptome_species = "human"
) -> pd.DataFrame:
    """Filters metadata for a specific contrast, runs DESeq2, and returns the results.

    This function is designed to prepare data for a volcano plot by performing
    a differential expression analysis on a specific two-condition comparison
    within a single experiment.

    Args:
        metadata (pd.DataFrame): The full metadata DataFrame. Must include 'sample',
            'batch', 'Experiment', and 'Condition' columns.
        conditions (list): A list of exactly two conditions to compare, where
            the first element is treated as the reference (e.g., ["Untreated", "Treated"]).
        experiment (str): The name of the specific experiment to filter by from
            the 'Experiment' column in the metadata.
        design_formula (str): The formula for the DESeq2 design matrix.
            Defaults to "~Condition".
        data_base_path (str): The base directory path where the data folders
            (e.g., 'batch_1/') are located. Defaults to "..".

    Returns:
        pd.DataFrame: A DataFrame containing the DESeq2 results for the
            specified contrast.
    """
    # --- Step 1: Filter metadata and set up file paths ---
    metadata.index = metadata["sample"]
    metadata["file_path"] = [f"{data_base_path}/{row['batch']}/1_salmon-processing/3_salmon-output/{row['sample'].split(' ')[-1]}/quant.sf" for i, row in metadata.iterrows()]

    # --- Step 2: Run tximport to get gene counts ---
    transcript_gene_map = create_transcript_gene_map(species=transcriptome_species)
    results = tximport(
        metadata["file_path"],
        data_type="salmon",
        transcript_gene_map=transcript_gene_map,
    )
    counts_df = pd.DataFrame(results.X, index=metadata.index, columns=results.var_names).T
    counts_df.index.name = 'Geneid'
    counts_df = counts_df.loc[counts_df.sum(axis=1) > 0].T.astype(int)

    # --- Step 3: Map Ensembl IDs to Gene Symbols ---
    mapper = id_map(species=transcriptome_species)
    counts_df.columns = counts_df.columns.map(mapper.mapper)
    counts_df = counts_df.groupby(counts_df.columns, axis=1).sum()

    # --- Step 4: Run DESeq2 analysis ---
    inference = DefaultInference(n_cpus=20)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design=design_formula,
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()

    # --- Step 5: Define contrasts and get results ---
    ref_condition = conditions[0]
    comp_conditions = [c for c in conditions if c != ref_condition]

    res_dict = {}
    for condition in comp_conditions:
        contrast = [contrast_design, condition, ref_condition]
        ds = DeseqStats(dds, contrast=contrast)
        ds.run_wald_test()
        ds.summary()
        res = ds.results_df
        res = res[res.baseMean >= 10]
        res['Symbol'] = res.index
        res_dict[condition] = res

    return res_dict[conditions[1]]

def plot_volcano(
    deseq_results: pd.DataFrame,
    contrast: list,
    highlight_genes: dict,
    palette: dict,
    plot_name: str = None,
    p_value_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    p_value_col: str = "padj",
    figure_size: tuple = (8, 8),
    x_range: tuple = None,
    label_top_genes: int = 20,
    hard_stop: bool = True,
    font_size: int = 9,
    plot_formats: List[str] = ["png"]
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
        p_value_cutoff (float): The significance threshold for the p-value. Defaults to 0.05.
        log2fc_cutoff (float): The minimum absolute log2 fold change for significance. Defaults to 1.0.
        p_value_col (str): The p-value column to use for filtering. Defaults to "padj".
        figure_size (tuple): The size (width, height) of the output figure. Defaults to (8, 8).
        x_range (tuple): The x-axis (log2FoldChange) limits for the plot. Defaults to None.
        label_top_genes (int): The number of top genes to label. Defaults to 50.
        hard_stop (bool): If True, strictly labels the top N genes. If False, uses
            the custom_slicer logic. Defaults to False.
        font_size (int): The font size for gene labels. Defaults to 9.
        plot_formats (List[str]): A list of file formats to save the plots in. Defaults to ["png"].
    """
    # --- Step 1: Prepare DataFrame for plotting ---
    plot_df = deseq_results.copy()
    plot_df[f'-log10({p_value_col})'] = -np.log10(plot_df[p_value_col])

    up_color = palette[contrast[1]]
    down_color = palette[contrast[0]]
    not_sig_color = 'lightgray'

    plot_df['color'] = np.select(
        [
            (plot_df[p_value_col] < p_value_cutoff) & (plot_df['log2FoldChange'] >= log2fc_cutoff),
            (plot_df[p_value_col] < p_value_cutoff) & (plot_df['log2FoldChange'] <= -log2fc_cutoff)
        ],
        [up_color, down_color],
        default=not_sig_color
    )

    plot_df.replace([np.inf, -np.inf], 324, inplace=True)
    plot_df.dropna(subset=['log2FoldChange', f'-log10({p_value_col})'], inplace=True)

    # --- Step 2: Create the plot ---
    plt.figure(figsize=figure_size)
    if x_range:
        plt.xlim(x_range)
    
    sorted_df = plot_df.sort_values(by='color', ascending=False)
    
    sns.scatterplot(
        data=sorted_df,
        x='log2FoldChange',
        y=f'-log10({p_value_col})',
        c=sorted_df['color'],
        edgecolor=None,
        alpha=0.7,
    )

    # --- Step 3: Label significant genes ---
    texts = []
    bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)

    text_df = plot_df[
        (abs(plot_df['log2FoldChange']) >= log2fc_cutoff) & (plot_df[p_value_col] < p_value_cutoff)
    ].copy()
    text_df["plot_stat"] = abs(text_df["stat"])*(text_df["log2FoldChange"]**2)
    text_df = text_df.sort_values("plot_stat", key=abs, ascending=False)

    if hard_stop:
        text_df_pos = text_df[text_df["log2FoldChange"] > 0].head(label_top_genes)
        text_df_neg = text_df[text_df["log2FoldChange"] < 0].head(label_top_genes)
    else:
        text_df_pos = custom_slicer(text_df[text_df["log2FoldChange"] > 0], initial_rows=label_top_genes, block_size=label_top_genes, step=2)
        text_df_neg = custom_slicer(text_df[text_df["log2FoldChange"] < 0], initial_rows=label_top_genes, block_size=label_top_genes, step=2)

    flattened_values = [
        value for key, lst in highlight_genes.items() if key != "black" for value in lst
    ]
    flattened_values = [gene for gene in flattened_values if gene in text_df.index]
    text_df_other = text_df.loc[flattened_values]
    
    text_df = pd.concat([text_df_pos, text_df_neg, text_df_other], ignore_index=True)
    text_df = text_df.drop_duplicates(subset=['Symbol'], keep='first')

    for _, row in text_df.iterrows():
        gene = row['Symbol']
        label_color = "black"
        for color, genes in highlight_genes.items():
            if gene in genes:
                label_color = color
        
        texts.append(plt.text(
            x=row['log2FoldChange'],
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
    metadata: pd.DataFrame,
    highlight_genes, 
    palette: Dict[str, str],
    dds_dict: tuple = None,
    include: Optional[Dict[str, List[Any]]] = None,
    exclude: Optional[Dict[str, List[Any]]] = None,
    plot_name: str = None,
    design_formula: str = "~Condition",
    contrast_design = "Condition", 
    ref_condition = "Untreated", 
    p_value_cutoff: float = 0.05,
    p_value_col: str = "padj",
    log2fc_cutoff: float = 1.0,
    label_top_genes=5, 
    log_transform=False,
    normalize_to_ref = False,
    v_range=(-3, 3), 
    n_row_clusters=None,
    display_top_genes = 1000000, 
    add_left_ha_dict = None, 
    figure_size=(8, 14), 
    column_annotations: Optional[List[str]] = None,
    plot_formats: List[str] = ["png"],
    available_metadata_columns: List[str] = ["Condition", "Experiment"],
    data_base_path: str = "..",
    gois_only = False,
    transcriptome_species = "human"
):
    """Performs a complete bulk RNA-seq analysis and generates a clustered heatmap.
    
    This high-level workflow function automates the entire process from raw
    data loading to final visualization. It handles metadata filtering, imports
    Salmon quantification files using tximport, runs a DESeq2 differential
    expression analysis, identifies significant genes, and plots the results
    as a publication-quality clustermap and a PCA plot.
    
    Args:
        metadata (pd.DataFrame): DataFrame containing sample information. Must
            include 'sample' and 'batch' columns.
        plot_name (str): A base name for all output files (e.g., "Myeloid_Analysis").
        highlight_genes (dict): A dictionary to highlight specific genes on the
            heatmap's row annotations. Keys are colors (e.g., 'red'), and
            values are lists of gene symbols.
        palette (Dict[str, str]): A dictionary mapping metadata values to colors
            for all plot annotations (e.g., {'Untreated': 'blue'}).
        include (Optional[Dict[str, List[Any]]]): Dictionary to filter metadata
            for rows to include. E.g., {'Condition': ['Treated']}. Defaults to None.
        exclude (Optional[Dict[str, List[Any]]]): Dictionary to filter metadata
            for rows to exclude. E.g., {'Experiment': ['Exp3']}. Defaults to None.
        design_formula (str): The design formula for DESeq2, specifying the
            experimental design. Defaults to "~Condition".
        contrast_design (str): The column in metadata to use for DESeq2
            contrasts. Defaults to "Condition".
        ref_condition (str): The reference level for DESeq2 comparisons
            (e.g., the control group). Defaults to "Untreated".
        p_value_cutoff (float): The significance threshold for the p-value.
            Defaults to 0.05.
        p_value_col (str): The specific p-value column from the DESeq2 results
            to use for filtering. Defaults to "padj".
        log2fc_cutoff (float): The minimum absolute log2 fold change required for
            a gene to be considered significant. Defaults to 1.0.
        label_top_genes (int): The number of top up- and down-regulated genes
            from each contrast to label on the heatmap. Defaults to 5.
        log_transform (bool): If True, applies a log1p transformation to the
            normalized counts before plotting. Defaults to False.
        v_range (Tuple[int, int]): The minimum and maximum values for the heatmap's
            color scale (z-score). Defaults to (-3, 3).
        n_row_clusters (int): The number of clusters to partition genes into for
            the heatmap's left annotation. Defaults to 4.
        display_top_genes (int): The maximum number of up- and down-regulated
            genes to retrieve from each contrast for the heatmap. Defaults to 1,000,000 (effectively all).
        add_left_ha_dict (Optional[Tuple]): A tuple containing a DataFrame and a
            color dictionary for adding extra row annotations. Defaults to None.
        figure_size (Tuple[int, int]): The size (width, height) of the output
            heatmap figure in inches. Defaults to (8, 14).
        column_annotations (Optional[List[str]]): A list of metadata columns to use
            for the heatmap's top annotation. Defaults to ["Experiment", "Condition"].
        plot_formats (List[str]): A list of file formats to save the plots in
            (e.g., ["png"]). Defaults to ["png"].
        available_metadata_columns (List[str]): Metadata columns to be used for
            PCA coloring and other internal operations. Defaults to ["Condition", "Experiment"].
        data_base_path (str): The base directory path where the data folders
            (e.g., 'batch_1/') are located. Defaults to "..".
    
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
            - A DataFrame of the normalized expression data used for the heatmap,
              with genes as rows and samples as columns. Returns None if not enough
              significant genes were found.
            - A pandas Series where the index contains the gene symbols and the
              values are their assigned cluster IDs. Returns None if clustering
              was not performed.
    """
    
    if column_annotations is None:
        column_annotations = [available_metadata_columns[-1], available_metadata_columns[0]]
    if highlight_genes is None:
        highlight_genes = {}
        
    if dds_dict is None:
        print("You are not using a dds file as input. While this is possible, I highly recommend running\n dds, results = get_dds_result()\n before and adding\n dds_dict = (dds, results)\n to this function.")
        # --- 1. Data Loading and Preparation ---
        metadata["xp_cond"] = [f"{i[available_metadata_columns[0]]}_{i[available_metadata_columns[-1]]}" for _, i in metadata.iterrows()]
        if include is not None:
            for k, v in include.items():
                metadata = metadata[metadata[k].isin(v)]
        if exclude is not None:
            for k, v in exclude.items():
                metadata = metadata[~metadata[k].isin(v)]
    
        metadata.index = metadata["sample"]
        metadata["file_path"] = [f"{data_base_path}/{row['batch']}/1_salmon-processing/3_salmon-output/{row['sample'].split(' ')[-1]}/quant.sf" for _, row in metadata.iterrows()]
        
        transcript_gene_map = create_transcript_gene_map(species=transcriptome_species)
        tx_results = tximport(metadata["file_path"], data_type="salmon", transcript_gene_map=transcript_gene_map)
        
        counts_df = pd.DataFrame(tx_results.X, index=metadata.index, columns=tx_results.var_names).T
        counts_df = counts_df[counts_df.sum(axis=1) > 0].T.astype(int)
        
        mapper = id_map(species=transcriptome_species)
        counts_df.columns = counts_df.columns.map(mapper.mapper)
        counts_df = counts_df.loc[:, ~counts_df.columns.duplicated(keep='first')]
        counts_df = counts_df.groupby(counts_df.columns, axis=1).sum()
    
        # --- 2. DESeq2 Analysis ---
        dds, res_dict, contrast_design = _run_deseq_analysis(
            counts_df, metadata, design_formula, contrast_design, ref_condition
        )
    else:
        dds, res_dict = dds_dict
        metadata = dds.obs

    sc.tl.pca(dds)

    if plot_name is not None:
        for plot_format in plot_formats:
            sc.pl.pca(dds, color = available_metadata_columns, size = 200, palette = palette, save = f'PCA_{plot_name}.{plot_format}')
    else:
        sc.pl.pca(dds, color = available_metadata_columns, size = 200, palette = palette)

    # --- 4. Gene Selection for Heatmap ---
    # Assuming stimulation_colors and experiment_colors are globally available
    primary_column_order = [cond for cond in palette if cond in set(metadata[available_metadata_columns[0]])]
    
    significant_genes_df = _get_significant_genes(
        res_dict, p_value_cutoff, p_value_col, log2fc_cutoff, contrast_design, primary_column_order, display_top_genes
    )
    unique_symbols = significant_genes_df["Symbol"].dropna().unique()
    dds_sigs = dds[:, unique_symbols].copy()

    # --- 5. Heatmap Generation ---
    if log_transform:
        dds_sigs.layers['use_counts'] = np.log1p(dds_sigs.layers['normed_counts'])
    else:
        dds_sigs.layers['use_counts'] = dds_sigs.layers['normed_counts']

    if gois_only:
        expression_df = pd.DataFrame(
            dds_sigs.layers['use_counts'].T,
            index=dds_sigs.var_names,
            columns=dds_sigs.obs_names
        )
        subset_gois = [entry for sublist in highlight_genes.values() for entry in sublist]
        subset_gois = [gene for gene in subset_gois if gene in dds_sigs.var_names]
        expression_df = expression_df.loc[subset_gois]
    else:
        expression_df = pd.DataFrame(
            dds_sigs.layers['use_counts'].T,
            index=dds_sigs.var_names,
            columns=dds_sigs.obs_names
        )

    # --- ROBUSTNESS CHECKS ---
    if expression_df.shape[0] < 2:
        print(f"\nWarning: Not enough significant genes ({expression_df.shape[0]}) found to create a cluster heatmap.")
        return None, None
    if expression_df.shape[1] < 2:
        print(f"\nWarning: Not enough samples ({expression_df.shape[1]}) found to perform clustering.")
        return None, None

    for available_metadata_column in available_metadata_columns:
        metadata_column_order = [col for col in palette if col in set(metadata[available_metadata_column])]
        metadata[available_metadata_column] = pd.Categorical(metadata[available_metadata_column], categories=metadata_column_order, ordered=True)

    metadata = metadata.sort_values(column_annotations)
    expression_df = expression_df[metadata.index]

    left_ha_dict = {}

    if n_row_clusters is not None:
        # --- 5a. Hierarchical Clustering and Row Reordering (Z-Score Corrected) ---
        scaled_expression_df = expression_df.T.apply(zscore).T.fillna(0)
    
        row_linkage = linkage(pdist(scaled_expression_df, metric='euclidean'), method='ward')
        row_clusters = cut_tree(row_linkage, n_clusters=n_row_clusters).flatten()
        
        ordered_row_indices = leaves_list(row_linkage)
        ordered_gene_names = expression_df.index[ordered_row_indices]
        
        expression_df = expression_df.loc[ordered_gene_names]
        row_clusters = pd.Series(row_clusters[ordered_row_indices], index=ordered_gene_names)
    
        # --- 5b. Define Annotations ---
        cluster_palette = sns.color_palette("Set2", n_row_clusters)
        row_clusters = pd.DataFrame(row_clusters, columns = ["Cluster"]).sort_values(by="Cluster")
        left_ha_dict["Gene Cluster"] = pch.anno_simple(row_clusters["Cluster"], 
                                                        colors=cluster_palette,
                                                        legend=True, add_text=True, text_kws={'fontsize':9,'color':'black'})
        
        row_clusters["Cluster"] = row_clusters["Cluster"].astype(str)
    
    if add_left_ha_dict is not None:
        # new
        for col in add_left_ha_dict[0].columns:
            left_ha_df = []
            for gene in expression_df.index:
                if gene in list(add_left_ha_dict[0].index):
                    left_ha_df.append(add_left_ha_dict[0].loc[gene][col])
                else:
                    left_ha_df.append("other")

            left_ha_dict_color = {k:v for k,v in add_left_ha_dict[1].items() if k in set(add_left_ha_dict[0][col])}
            left_ha_dict_color["other"] = add_left_ha_dict[1]["other"]
            left_ha_dict[col] = pch.anno_simple(pd.Series(left_ha_df, index=expression_df.index), 
                                                        colors= left_ha_dict_color,
                                                        legend=True, add_text=False, text_kws={'fontsize':9,'color':'black'}, height = 10)

    if left_ha_dict != {}:
        left_ha = pch.HeatmapAnnotation(
            **left_ha_dict,
            axis=0, plot_legend=True, legend_side='right'
        )

    top_markers = []
    for res in res_dict.values():
        top_markers.extend(res.sort_values(by='stat', ascending=False).head(label_top_genes)["Symbol"].tolist())
        top_markers.extend(res.sort_values(by='stat', ascending=False).tail(label_top_genes)["Symbol"].tolist())

    if normalize_to_ref:
        empty_df = pd.DataFrame()
        metadata["lane"] = metadata["Experiment"]
        for time in np.unique(metadata["lane"]):
            samples = metadata[(metadata["lane"] == time) & (metadata[contrast_design] == ref_condition)]["sample"]
            divide = expression_df[samples].mean(axis=1)
            samples = metadata[(metadata["lane"] == time) & (metadata[contrast_design] != ref_condition)]["sample"]
            # new_expression_df = expression_df[samples].div(subtract, axis=0) 
            new_expression_df = expression_df.div(divide, axis=0)
            for column in new_expression_df:
                empty_df[column] = new_expression_df[column]
        expression_df = empty_df.copy()
        expression_df= expression_df.replace([np.inf, -np.inf], np.nan)
        expression_df = expression_df.dropna()
        row_means = expression_df.mean(axis=1)
        if not gois_only:
            tf_condition = (row_means < 0.95) | (row_means > 1.05)
            expression_df = expression_df[tf_condition]
        row_stds = expression_df.std(axis=1)
        sorted_stds = row_stds.sort_values(ascending=False)
        top_markers1 = sorted_stds.head(label_top_genes*len(list(res_dict.keys()))).index.to_list()
        v_range = (0, round(np.max(expression_df)))

        custom_cmap = LinearSegmentedColormap.from_list(
                "my_coolwarm",
                [(0, "blue"), (1/round(np.max(expression_df)), "white"), (1.0, "red")]
        )

    row_colors_dict = {gene: 'black' for gene in top_markers}
    for color, genes in highlight_genes.items():
        for gene in genes:
            row_colors_dict[gene] = color
            
    selected_rows = list(set(top_markers + [g for sublist in highlight_genes.values() for g in sublist]))
    label_rows = expression_df.index.to_series().apply(lambda x: x if x in selected_rows else None)
    
    row_ha = pch.HeatmapAnnotation(
        selected=pch.anno_label(label_rows, colors=row_colors_dict, relpos=(0,0.4), extend = True, merge = True, height = 10),
        axis=0, fontsize=10, verbose=0, orientation='right'
    )
    
    col_anno_dict = {}
    for ca in column_annotations:
        available_colex = {k: v for k, v in palette.items() if k in set(metadata[ca])}
        col_anno_dict[ca] = pch.anno_simple(metadata[ca], legend=True, colors=available_colex)

    available_colex = {k: v for k, v in palette.items() if k in set(metadata[column_annotations[0]])}
    col_ha = pch.HeatmapAnnotation(label=pch.anno_label(metadata[column_annotations[0]], merge=True, rotation=45, colors=available_colex),
                                   **col_anno_dict, 
                                   label_side='right', axis=1)

    # --- 5c. Plot the Heatmap ---
    plt.figure(figsize=figure_size)
    if normalize_to_ref:
        cm = pch.ClusterMapPlotter(data=expression_df, #z_score=0,
                                right_annotation=row_ha, top_annotation=col_ha,
                                col_cluster=False,row_cluster=False,
                                label='Expression',row_dendrogram=False, col_dendrogram=False,
                                show_rownames=False,show_colnames=False,
                                cmap=custom_cmap,
                                tree_kws={'row_cmap': 'Dark2'},
                                xticklabels_kws={'labelrotation':-45,'labelcolor':'blue'},
                                yticklabels_kws = {'labelsize':6}, vmin=v_range[0],vmax=v_range[1]
                               )
    else:
        if left_ha_dict != {}:
            cm = pch.ClusterMapPlotter(
                data=expression_df, z_score=0, 
                left_annotation=left_ha,
                right_annotation=row_ha,
                top_annotation=col_ha,
                row_cluster=[False if n_row_clusters is not None else True], 
                row_dendrogram=False,
                col_cluster=False,
                col_dendrogram=False,
                label='Z-score',
                show_rownames=False,
                show_colnames=False,
                cmap='viridis',
                vmin=v_range[0], vmax=v_range[1]
            )
        else:
            cm = pch.ClusterMapPlotter(
                data=expression_df, z_score=0, 
                right_annotation=row_ha,
                top_annotation=col_ha,
                row_cluster=[False if n_row_clusters is not None else True], 
                row_dendrogram=False,
                col_cluster=False,
                col_dendrogram=False,
                label='Z-score',
                show_rownames=False,
                show_colnames=False,
                cmap='viridis',
                vmin=v_range[0], vmax=v_range[1]
            )
    
    plt.suptitle(f"exp_design: {design_formula}, contrast_design: {contrast_design}, pval: {p_value_cutoff}, #genes per contrast: {label_top_genes}, used_pvalue: {p_value_col}")

    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"{save_dir}/cm_{plot_name}_{design_formula}_{contrast_design}.{plot_format}", bbox_inches='tight', dpi=300)
    plt.show()

    if n_row_clusters is not None:
        return expression_df, row_clusters
    else:
        return expression_df, None

def plot_go(
    genes, 
    plot_name: str = None,
    go_organism="hsapiens",
    go_sources=['GO:BP'], 
    figure_size=(6, 5),
    plot_formats: List[str] = ["png"]
):
    """
    Performs GO enrichment analysis for gene clusters and generates a bubble plot
    with fixed, discrete legends for p-value and gene ratio.
    """
    # --- Helper functions to categorize continuous data ---
    def categorize_p_value(p_val):
        if p_val <= 1e-4: return '<= 1e-4'
        if p_val <= 1e-3: return '1e-3'
        if p_val <= 1e-2: return '1e-2'
        if p_val <= 1e-1: return '1e-1'
        return '> 1e-1'

    def categorize_gene_ratio(ratio):
        if ratio >= 0.8: return '>= 0.8'
        if ratio >= 0.6: return '0.6'
        if ratio >= 0.4: return '0.4'
        if ratio >= 0.2: return '0.2'
        return '< 0.2'
    # ----------------------------------------------------
    if type(genes) == pd.DataFrame:
        cluster_df = genes
    if type(genes) == list:
        cluster_df = pd.DataFrame(index = genes)
        cluster_df["Cluster"] = "Gene List"
        
    for cluster_no in np.unique(cluster_df["Cluster"]):
        gene_list = list(cluster_df[cluster_df["Cluster"] == cluster_no].index)
        
        # Run enrichment analysis
        gp = GProfiler(return_dataframe=True)
        go_results = gp.profile(
            organism=go_organism,
            query=gene_list,
            sources=go_sources
        )
        
        # Prepare data for plotting
        go_df = go_results[go_results['p_value'] < 0.05].copy()
        if go_df.empty:
            print(f"No significant GO terms found for Cluster {cluster_no}. Skipping plot.")
            continue
            
        go_df['gene_ratio'] = go_df['intersection_size'] / go_df['query_size']
        go_df['-log10_p_value'] = -np.log10(go_df['p_value'])
        top_10_go = go_df.sort_values('-log10_p_value', ascending=False).head(10)

        # Apply the text wrapping function to the 'name' column
        top_10_go['name'] = top_10_go['name'].apply(wrap_text_by_words)

        # 1. Create categorical columns based on your desired fixed values
        top_10_go['p_value_cat'] = top_10_go['p_value'].apply(categorize_p_value)
        top_10_go['gene_ratio_cat'] = top_10_go['gene_ratio'].apply(categorize_gene_ratio)
        
        # 2. Define the exact colors and sizes for each category
        # P-value color map
        p_val_order = ['<= 1e-4', '1e-3', '1e-2', '1e-1']
        p_val_colors = sns.color_palette("viridis_r", n_colors=len(p_val_order))
        p_value_palette = dict(zip(p_val_order, p_val_colors))

        # Gene ratio size map
        ratio_order = ['>= 0.8', '0.6', '0.4', '0.2', '< 0.2'] # <-- Added '< 0.2'
        ratio_sizes_list = [500, 350, 200, 50, 25] # <-- Added a size for the smallest category
        gene_ratio_sizes = dict(zip(ratio_order, ratio_sizes_list))

        # 3. Create the plot without the automatic legend
        
        save_dir = "DEGs"
        os.makedirs(save_dir, exist_ok=True)
        
        go_df.to_csv(f"{save_dir}/goterm_{plot_name}_{cluster_no}.csv")
        fig, ax = plt.subplots(figsize=figure_size)
        plot = sns.scatterplot(
            data=top_10_go,
            x='-log10_p_value',
            y='name',
            hue='p_value_cat',
            size='gene_ratio_cat',
            hue_order=p_val_order,
            size_order=ratio_order,    # Use the updated order
            palette=p_value_palette,
            sizes=gene_ratio_sizes,    # Use the updated size map
            size_norm=mcolors.NoNorm(),# <-- Add this for robustness
            edgecolor='black',
            linewidth=1.5,
            ax=ax,
            legend=False
        )

        # 4. Manually create the legends with fixed values
        # P-value legend
        p_val_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=p_value_palette[label], markersize=10, label=label)
                         for label in p_val_order]
        legend1 = ax.legend(handles=p_val_handles, title='p-value',
                            bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        ax.add_artist(legend1)

        # Gene Ratio legend
        ratio_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='gray', markersize=np.sqrt(gene_ratio_sizes[label]), label=label)
                          for label in ratio_order]
        ax.legend(handles=ratio_handles, title='Gene Ratio',
                  bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=False)
        
        # 5. Customize the plot
        ax.set_xlabel('-log10(Adjusted p-value)', fontsize=14)
        ax.set_ylabel('')
        ax.set_title(f'Top 10 Enriched GO:BP Terms, Cluster {cluster_no}', fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=12)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if plot_name is not None:
            save_dir = "figures"
            os.makedirs(save_dir, exist_ok=True)
            for plot_format in plot_formats:
                plt.savefig(f"{save_dir}/goterm_{plot_name}_{cluster_no}.{plot_format}", dpi=300, bbox_inches='tight')
        plt.show()

def plot_gene_curves(
    metadata: pd.DataFrame, 
    expression_data: pd.DataFrame, 
    genes_to_plot: list,
    plot_name: str = None,
    category_column: str = 'Experiment',
    sample_col: str = 'sample',
    condition_col: str = 'Condition',
    condition: str = None,
    plot_order: list[str] = None,
    plot_points: bool = False, 
    figure_size: tuple = (6, 4.5),
    normalize_to_one: bool = True,
    plot_formats: List[str] = ["png"]
):
    """
    Plots the expression of specified genes over time, with options for normalization and showing individual data points.

    Args:
        metadata (pd.DataFrame): DataFrame with sample metadata. Must contain columns for time,
                                 sample ID, and optionally, experimental condition.
        expression_data (pd.DataFrame): DataFrame with expression values. Index should be gene names
                                        and columns should be sample IDs.
        genes_to_plot (list): A list of gene names (strings) to plot.
        plot_name (str): A descriptive name for the output plot file.
        category_column (str): The name of the column in `metadata` that contains the numeric time values (e.g., hours).
        sample_col (str): The name of the column in `metadata` that contains the unique sample IDs.
        condition_col (str): The name of the column in `metadata` that contains the condition labels.
        condition (str, optional): If specified, filters the data to only this condition. Defaults to None.
        plot_points (bool): If True, plots individual data points for each replicate. Defaults to False.
        figure_size (tuple): The size of the matplotlib figure.
        normalize_to_one (bool): If True, normalizes each gene's curve so the mean at the first timepoint is 1.
    """
    # --- 1. Data Filtering and Preparation ---
    
    # Make copies to avoid modifying the original DataFrames
    meta_df = metadata.copy()
    meta_df = metadata[metadata["sample"].isin(list(expression_data.columns))]
    
    # If a specific condition is provided, filter the metadata
    if condition is not None:
        if condition_col not in meta_df.columns:
            print(f"Warning: Condition column '{condition_col}' not found in metadata. Plotting all data.")
        else:
            meta_df = meta_df[meta_df[condition_col] == condition]

    # Filter the expression data to include only the samples present in the (potentially filtered) metadata
    samples_to_plot = meta_df[sample_col].tolist()
    expr_df = expression_data[samples_to_plot]

    # Get the unique, sorted timepoints that will form the x-axis
    if plot_order is None:
        try:
            timepoints = meta_df[category_column].unique()
        except KeyError:
            print(f"Error: Column '{category_column}' not found in metadata. Cannot proceed.")
            return
    else:
        timepoints = plot_order

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=figure_size)

    # Loop through each gene to calculate its trend and plot it
    for gene in genes_to_plot:
        if gene not in expr_df.index:
            print(f"Warning: Gene '{gene}' not found in expression data. Skipping.")
            continue

        y_means = []
        y_sems = []
        all_raw_values = []

        # For each timepoint, find the corresponding samples and get their expression values
        for t in timepoints:
            # Find sample IDs for the current timepoint
            samples_at_t = meta_df[meta_df[category_column] == t][sample_col].tolist()
            
            # Get the expression values for those samples for the current gene
            values_at_t = expr_df.loc[gene, samples_at_t]
            
            # Calculate mean and SEM and store them
            y_means.append(values_at_t.mean())
            y_sems.append(sem(values_at_t))
            
            # Store raw values if we need to plot them later
            if plot_points:
                all_raw_values.append(values_at_t)

        # Convert lists to numpy arrays for easier calculations
        y_means = np.array(y_means)
        y_sems = np.array(y_sems)
        
        # --- 3. Normalization (Optional) ---
        
        y_means_final = y_means
        norm_offset = 0

        # Normalize the curve so that the first timepoint's mean is 1
        if normalize_to_one and len(y_means) > 0:
            # This is an additive shift. The variance (and thus SEM) does not change.
            norm_offset = y_means[0]
            y_means_final = y_means / norm_offset
            y_sems_final = y_sems / norm_offset

        # --- 4. Drawing on the Plot ---
        
        # Plot the mean expression line with an error band for SEM
        ax.errorbar(timepoints, y_means_final, yerr=y_sems_final, fmt='-o', capsize=4, elinewidth=1.5, markeredgewidth=1.5, label=gene)

        # Plot the individual data points (replicates)
        if plot_points:
            for i, t in enumerate(timepoints):
                # Apply the same normalization offset to the raw values
                raw_values_norm = all_raw_values[i] - norm_offset
                # Add a small amount of jitter to the x-axis to prevent points from overlapping perfectly
                x_jitter = np.random.normal(0, np.mean(np.diff(timepoints))*0.02, size=len(raw_values_norm))
                ax.scatter(t + x_jitter, raw_values_norm, alpha=0.5, s=40, zorder=10)

    # --- 5. Final Plot Formatting ---
    
    ax.set_xlabel(f"Time")
    ylabel = "Expression"
    if normalize_to_one:
        ylabel += " (Normalized to T0=1)"
    ax.set_ylabel(ylabel)
    
    title = "Gene Expression Over Time"
    if condition:
        title += f" ({condition})"
    ax.set_title(title, fontweight='bold')
    
    # Place legend outside the plot area
    ax.legend(title="Genes", bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_xticklabels(timepoints, rotation=40, ha="right")
    
    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make space on the right for the legend
    
    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"{save_dir}/curve_{plot_name}.{plot_format}", dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_cluster_curve(
    metadata: pd.DataFrame,
    expression_data: pd.DataFrame,
    genes_to_plot: list,
    cluster: any,
    plot_name: str = None,
    category_column: str = "Experiment",
    condition_col: str = 'Condition',
    condition: str = None,
    plot_order: list[str] = None,
    plot_points: bool = False,
    figure_size: tuple = (7, 5),
    plot_formats: List[str]=["png"],
):
    """
    Plots gene expression over time, handling variable numbers of replicates per timepoint.

    Args:
        expression_data (pd.DataFrame): DataFrame with expression values (genes x samples).
        metadata (pd.DataFrame): DataFrame with sample metadata. Must contain 'sample'
                                 and 'Experiment' columns.
        genes_to_plot (list): A list of gene names to plot.
        plot_name (str): A descriptive name for the output plot file.
        cluster (any): A cluster identifier, used for the plot title.
        plot_points (bool): If True, plots individual data points for each replicate.
        figure_size (tuple): The size of the matplotlib figure.
        save_plot (bool): If True, saves the plot to a file.
    """
    # --- 1. Determine X-axis values from metadata ---
    metadata = metadata[metadata["sample"].isin(list(expression_data.columns))]
    if condition is not None:
        metadata = metadata[metadata[condition_col]==condition]
    
    # Get the unique, sorted experiment names (e.g., ['D48', 'D58', ...])
    if plot_order is None:
        experiments = metadata[category_column].unique()
    else:
        experiments = plot_order
    
    # Convert experiment names to numeric x-values (e.g., 'D48' -> 48)
    # This assumes a 'D' prefix, adjust if your naming is different.
    try:
        x_values = [int(exp[1:]) for exp in experiments]
    except (ValueError, IndexError):
        print(f"Warning: Could not parse numeric values from {category_column} column. Using simple integer sequence for x-axis.")
        x_values = experiments

    # --- 2. Create Plot ---
    plt.figure(figsize=figure_size)
    
    # --- 3. Loop through genes, calculate stats, and plot ---
    
    for gene in genes_to_plot:
        if gene not in expression_data.index:
            print(f"Warning: Gene '{gene}' not found in expression_data. Skipping.")
            continue

        gene_row = expression_data.loc[gene]
        y_means, y_sems, all_raw_values = [], [], []

        # Group samples by experiment to handle variable replicate numbers
        for exp in experiments:
            # Find all sample names for the current experiment
            replicate_samples = metadata[metadata[category_column] == exp]['sample'].tolist()
            
            # Get the expression values for these replicates
            values = gene_row[replicate_samples]
            
            # Calculate mean and SEM for this timepoint
            y_means.append(values.mean())
            y_sems.append(sem(values))
            all_raw_values.append(values)

        y_means = np.array(y_means)
        y_sems = np.array(y_sems)

        # Normalize to start at 1 (optional)
        norm_offset = y_means[0] # - 1.0 #if len(y_means) > 0 else 0
        y_means_norm = y_means / norm_offset 
        y_sems_norm = y_sems / norm_offset
        
        # Plot mean ± SEM line (using your original style)
        # A single color is used per gene now for clarity
        line, = plt.plot(x_values, y_means_norm, label=gene, alpha=0.2, lw=2, color = "red")
        plt.fill_between(x_values, y_means_norm - y_sems_norm, y_means_norm + y_sems_norm, color="yellow", alpha=0.2)

        # Plot all raw points (replicates)
        if plot_points:
            for i, x in enumerate(x_values):
                # Apply the same normalization to the raw values for this timepoint
                values_norm = all_raw_values[i] - norm_offset
                # Scatter the points with a little horizontal jitter to prevent overlap
                x_jitter = np.random.normal(0, 0.5, size=len(values_norm))
                plt.scatter(x + x_jitter, values_norm, alpha=0.6, s=30, color=line.get_color())

    # --- 4. Final Plot Formatting ---
    
    plt.xlabel("Time") # Changed from hours to match 'D48' etc.
    plt.ylabel("Expression (Normalized, Mean ± SEM)")
    plt.title(f"Gene Expression Over Time in Cluster {cluster}")
    plt.xticks(experiments, rotation=45, ha='right')
    plt.tight_layout()
    
    if plot_name is not None:
        save_dir = "figures/cluster_curves"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            filename = os.path.join(save_dir, f"curve_{plot_name}_{genes_to_plot[0]}_{len(genes_to_plot)}.{plot_format}")
            plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        
    plt.show()

def get_dds_results(
    metadata,
    include: Optional[Dict[str, List[Any]]] = None,
    exclude: Optional[Dict[str, List[Any]]] = None, 
    design_variable="~Condition",
    contrast_design = "Condition", 
    ref_condition = "Untreated",
    data_base_path = "..",
    transcriptome_species="human"
):
    # --- 1. Data Loading and Preparation ---
    metadata["xp_cond"] = [f"{i['Experiment']}_{i['Condition']}" for _, i in metadata.iterrows()]
    if include is not None:
        for k, v in include.items():
            metadata = metadata[metadata[k].isin(v)]
    if exclude is not None:
        for k, v in exclude.items():
            metadata = metadata[~metadata[k].isin(v)]

    metadata.index = metadata["sample"]
    metadata["file_path"] = [f"{data_base_path}/{row['batch']}/1_salmon-processing/3_salmon-output/{row['sample'].split(' ')[-1]}/quant.sf" for _, row in metadata.iterrows()]

    transcript_gene_map = create_transcript_gene_map(species=transcriptome_species)
    tx_results = tximport(metadata["file_path"], data_type="salmon", transcript_gene_map=transcript_gene_map)
    
    counts_df = pd.DataFrame(tx_results.X, index=metadata.index, columns=tx_results.var_names).T
    counts_df = counts_df[counts_df.sum(axis=1) > 0].T.astype(int)
    
    mapper = id_map(species=transcriptome_species)
    counts_df.columns = counts_df.columns.map(mapper.mapper)
    counts_df = counts_df.loc[:, ~counts_df.columns.duplicated(keep='first')]
    counts_df = counts_df.groupby(counts_df.columns, axis=1).sum()

    # --- 2. DESeq2 Analysis ---
    dds, res_dict, contrast_design = _run_deseq_analysis(
        counts_df, metadata, design_variable, contrast_design, ref_condition
    )
    
    return dds, res_dict

def plot_venns(
    plot_dict,
    palette,
    compare_response,
    plot_name: str = None,
    p_value_cutoff = 0.05,
    p_value_col = "padj",
    log2fc_cutoff = (-1,1),
    plot_formats = ["png"]):

    significant_genes_down = {}
    for key, df in plot_dict.items():
        filtered_df = df[(df[p_value_col] < p_value_cutoff) & (df['log2FoldChange'] < log2fc_cutoff[0])]
        significant_genes_down[key] = set(filtered_df.index)
    
    petal_labels = generate_petal_labels(significant_genes_down.values(), fmt="{size}")
    condition_colors_ordered = {key: palette[key] for key in significant_genes_down if key in palette}
    hex_colors = list(condition_colors_ordered.values())
    
    # Convert to RGBA (with full opacity)
    rgba_colors = [(r, g, b, 0.5) for r, g, b, _ in [mcolors.to_rgba(color) for color in hex_colors]]
    
    draw_venn(
        petal_labels=petal_labels, dataset_labels=significant_genes_down.keys(),
        hint_hidden=False, colors=rgba_colors,
        figsize=(8, 8), fontsize=30, legend_loc="upper right", ax=None
    )
    
    plt.title(f"Venn Diagram of Downregulated Genes in {compare_response[1]} Compared to {compare_response[0]}")
    plt.suptitle(f"{list(significant_genes_down.keys())}: {p_value_col}: {p_value_cutoff}, log2fc {log2fc_cutoff[-1]}, downregulations", size = 7.5)

    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f'figures/venn_{plot_name}_downregulations.{plot_format}', dpi = 300)
    plt.show()

    significant_genes_up = {}
    for key, df in plot_dict.items():
        filtered_df = df[(df[p_value_col] < p_value_cutoff) & (df['log2FoldChange'] > log2fc_cutoff[1])]
        significant_genes_up[key] = set(filtered_df.index)
    
    petal_labels = generate_petal_labels(significant_genes_up.values(), fmt="{size}")
    condition_colors_ordered = {key: palette[key] for key in significant_genes_up if key in palette}
    hex_colors = list(condition_colors_ordered.values())
    
    # Convert to RGBA (with full opacity)
    rgba_colors = [(r, g, b, 0.5) for r, g, b, _ in [mcolors.to_rgba(color) for color in hex_colors]]
    
    draw_venn(
        petal_labels=petal_labels, dataset_labels=significant_genes_up.keys(),
        hint_hidden=False, colors=rgba_colors,
        figsize=(8, 8), fontsize=30, legend_loc="upper right", ax=None
    )
    
    plt.title(f"Venn Diagram of Upregulated Genes in {compare_response[1]} Compared to {compare_response[0]}")
    plt.suptitle(f"{list(significant_genes_up.keys())}: {p_value_col}: {p_value_cutoff}, log2fc {log2fc_cutoff[1]}, upregulations", size = 7.5)

    if plot_name is not None:
        for plot_format in plot_formats:
            plt.savefig(f'figures/venn_{plot_name}_upregulations.{plot_format}', dpi = 300)
    plt.show()

    return significant_genes_down, significant_genes_up

def plot_tpm_curve(
    dds,
    palette,
    highlight_genes,
    plot_name: str = None,
    conditions = None,
    condition_col = "Condition",
    plot_formats = ["png"],
    figure_size = (10,5)
):
    if conditions is None:
        conditions = list(set(dds.obs[condition_col]))
        
    # Create a figure with 1 row, 2 columns of subplots.
    # sharey=True links the y-axes for direct comparison.
    fig, axes = plt.subplots(1, len(conditions), figsize=(figure_size[0], figure_size[1]), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    
    # --- 2. Loop through conditions and plot on each axis ---
    for ax, condition in zip(axes, conditions):
        # --- Data Preparation for the current condition ---
        dds_bam = dds[dds.obs[condition_col] == condition]
        tpm_df = pd.DataFrame(dds_bam.X, index = dds_bam.obs[condition_col], columns=dds_bam.var_names)
        x = tpm_df.mean(axis=0).sort_values(ascending=True)
        x = x[x > 10]
    
        # --- Plotting on the current axis 'ax' ---
        ax.plot(list(range(0, len(x))), x, linestyle='-', markersize=4, label='Data Series', color=palette[condition])
        ax.set_yscale("log")
    
        # --- Highlight points and add labels ---
        points_to_highlight = []
        for k,vs in highlight_genes.items():
            for v in vs:
                if v in x.index:
                    points_to_highlight.append((v, k))
                
        texts = []
    
        for label, color in points_to_highlight:
            x_pos = x.index.get_loc(label)
            y_val = x.loc[label]
            ax.scatter(x_pos, y_val, color='red', s=0, zorder=5) # s=0 makes marker invisible
            texts.append(ax.text(x_pos, y_val, label, color=color,
                                   ha='center', zorder=10,
                                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')))
    
        # --- Adjust text specifically for the current axis 'ax' ---
        adjust_text(texts, ax=ax, # Pass the current axis to the function
                    expand_points=(1.5, 1.5),
                    force_static=(1.5,1.5),
                    force_explode=(2,2),
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    
        # --- Final styling for the current axis 'ax' ---
        ax.set_title(f'Normalized Expression Values for "{condition}"')
        ax.set_xlabel("Gene Rank")
    
    # --- 3. Final global figure styling ---
    axes[0].set_ylabel("Normalized Expression (log scale)") # Set Y-label only for the left plot
    fig.suptitle(f"Comparison of Normalized Expression Values by {condition_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"figures/TPMcurve_{plot_name}.{plot_format}", dpi = 300)
        
    plt.show()

def plot_tpm_curve_comparison(
    dds,
    conditions,
    highlight_genes,
    palette,
    plot_name = None,
    condition_col="Condition",
    plot_formats=["png"],
    figure_size = (10, 5)
):
    # --- 1. Prepare data for BOTH conditions ---
    dds_1 = dds[dds.obs[condition_col] == conditions[0]]
    tpm_df_1 = pd.DataFrame(dds_1.X, index=dds_1.obs[condition_col], columns=dds_1.var_names)
    x_1 = tpm_df_1.mean(axis=0).sort_values(ascending=True)
    x_1 = x_1[x_1 > 10] # Base series for plotting
    
    # --- 2. Plotting ---
    plt.figure(figsize=figure_size)
    
    # Plot the original "- TGFB1" data series
    plt.plot(list(range(0, len(x_1))), x_1, linestyle='-', markersize=4, label=f'Normalized Expression ({conditions[0]})', color=palette[conditions[0]])
    plt.yscale("log")
    
    texts = []
    ts = []
    
    # --- 3. Highlight points and add vertical lines ---
    for condition in conditions[1:]:
        dds_2 = dds[dds.obs[condition_col] == condition]
        tpm_df_2 = pd.DataFrame(dds_2.X, index=dds_2.obs[condition_col], columns=dds_2.var_names)
        x_2 = tpm_df_2.mean(axis=0)
        x_2 = x_2[x_2 > 10]
        
        points_to_highlight = []
        for k,vs in highlight_genes.items():
            for v in vs:
                if v in x_2.index:
                    points_to_highlight.append((v, k))
            
        
        for label, col in points_to_highlight:
            if label in x_1 and label in x_2:
                # Get coordinates for both conditions
                x_pos = x_1.index.get_loc(label)
                y_val_minus = x_1.loc[label]
                y_val_plus = x_2.loc[label]
        
                # Add the marker for the original point (-TGFB1)
                plt.scatter(x_pos, y_val_minus, color=palette[conditions[0]], s=1, zorder=5)
        
                # Draw the vertical line connecting the two conditions
                plt.plot([x_pos, x_pos], [y_val_minus, y_val_plus], color='gray', linestyle='--', linewidth=1.5, zorder=4)
    
                color = palette[condition]
                plt.scatter(x_pos, y_val_plus, color=color, s=50, zorder=5)
                
                if label not in ts:
                    ts.append(label)
                    texts.append(plt.text(x_pos, y_val_plus, label, fontsize=10, ha='center', zorder=10, color = col,
                                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')))

    # Call adjust_text to automatically position labels without overlap
    adjust_text(texts, 
                expand_points=(2, 2),
                expand_text=(1.6, 1.6),
                force_explode=(2,2),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    # --- 4. Final plot styling ---
    plt.suptitle(f"{len(points_to_highlight)} genes labeled that have abs(log2fc) > 2.5 and are similar regulated in human and mouse")
    plt.title(f"Normalized Expression Change {conditions[0]} and {conditions[1]}")
    plt.xlabel("Gene Rank")
    plt.ylabel("Normalized Expression (log scale)")
    
    # Update the legend to match the new colors
    legend_elements = [Line2D([0], [0], color=palette[conditions[0]], lw=2, label=f'Normalized Expression ({conditions[0]})')]
    for condition in conditions[1:]:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[condition], markersize=8, label=f'Up-/Downregulated in {condition}'))
        
    plt.legend(handles=legend_elements, loc='best')

    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"figures/TPMcurvecomparison_{plot_name}.{plot_format}", dpi = 300)
            
    plt.show()

def plot_tpm_bar(
    dds,
    conditions,
    genes,
    palette,
    plot_points: bool = False,
    condition_col: str = "Condition",
    normalize_to_first: bool = True,
    plot_name = None,
    plot_formats: list[str]=["png"]
):
    # --- Data Preparation ---
    expression_df = pd.DataFrame(dds.X.T, columns=dds.obs["sample"], index=dds.var_names)

    genes = [gene for gene in genes if gene in dds.var_names]
    
    df_data = {}
    for i in conditions:
        samples = dds.obs[dds.obs[condition_col]==i]["sample"]
        df_data[f"{i}"] = expression_df.loc[genes, samples].mean(axis=1)
        df_data[f"{i}_sem"] = expression_df.loc[genes, samples].sem(axis=1)
        
    df = pd.DataFrame(df_data)

    # --- Normalization (with bug fix for SEM) ---
    if normalize_to_first:
        # Store normalization factors (mean of the first condition)
        norm_factors = df[conditions[0]].copy()
        
        # Normalize all columns (both means and SEMs) by these factors
        for col in df.columns:
            df[col] = df[col].div(norm_factors, axis=0)

    # --- Plotting Setup ---
    x = np.arange(len(df.index))  # The label locations
    num_conditions = len(conditions)
    total_width = 0.8  # The total width for a group of bars
    bar_width = total_width / num_conditions # The width of a single bar
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # --- Plotting Loop ---
    for i, condition in enumerate(conditions):
        # Calculate the offset to center the bars
        offset = (i - (num_conditions - 1) / 2) * bar_width
        
        # Plot the bars
        ax.bar(
            x + offset, 
            df[condition], 
            bar_width, 
            label=condition,
            color=palette[condition], 
            yerr=df[f"{condition}_sem"], 
            capsize=4
        )
        
        # --- NEW: Plot individual data points ---
        if plot_points:
            # 1. Get sample names for the current condition
            samples = dds.obs[dds.obs[condition_col] == condition]["sample"]
            
            # 2. Get the raw expression data for these samples
            points_data = expression_df.loc[genes, samples]
            
            # 3. Normalize the points if required
            if normalize_to_first:
                plot_points_data = points_data.div(norm_factors, axis=0)
            else:
                plot_points_data = points_data
            
            # 4. Add jitter for better visualization
            # Create an x-position for each data point, matching the bar's position
            x_points = np.repeat(x + offset, len(samples))
            # Add a small amount of random noise to the x-positions
            jitter = np.random.normal(0, bar_width * 0.1, size=len(x_points))
            
            # 5. Plot the points
            ax.scatter(
                x_points + jitter, 
                plot_points_data.values.ravel(), 
                s=15,           # size of the points
                color='0.3',    # dark gray color
                zorder=2,       # ensure points are on top of bars
                alpha=0.7       # slight transparency
            )

    # --- Final Plot Adjustments ---
    if normalize_to_first:
        ax.set_ylabel(f'TPM normalized to {conditions[0]}')
        ax.set_title('Normalized TPM')
    else:
        ax.set_ylabel('TPM')
        ax.set_title('TPM')
        
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()

    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"figures/TPM_{plot_name}.{plot_format}", bbox_inches='tight', dpi=300)
    plt.show()

def plot_go_enrichment(
    deseq_results: pd.DataFrame,
    contrast: list,
    palette: dict,
    p_value_cutoff = 0.05,
    p_value_col = "padj",
    log2fc_cutoff: tuple = (-1,1),
    plot_name: str = None,
    n_top_terms: int = 10,
    lfc_range: tuple = (-1.5, 1.5),
    figure_size: tuple = (10, 8),
    plot_formats: list = [],
    go_organism = "hsapiens",
    go_sources=['GO:BP']
    
):
    """
    Generates a stacked bar plot of top enriched GO terms with a custom continuous colormap.

    Args:
        go_results (pd.DataFrame): DataFrame of results from gProfiler.
            Must be run with no_evidences=False to include the 'intersection' column.
        deseq_results (pd.DataFrame): DataFrame of DESeq2 results containing
            'log2FoldChange' values, indexed by gene symbols.
        plot_name (str): A base name for the output plot files.
        contrast (list): A list of two strings defining the contrast,
            e.g., ["Untreated", "Treated"]. The first is used for the negative
            color, the second for the positive.
        palette (dict): A dictionary mapping condition names to colors.
        n_top_terms (int): The number of top GO terms to display. Defaults to 10.
        lfc_range (tuple): The range of log2 fold change values for the colormap.
            Defaults to (-1.5, 1.5).
        figure_size (tuple): The size (width, height) of the output figure.
        plot_formats (List[str]): List of file formats to save the plots in.
    """
    significant_genes = deseq_results[deseq_results[p_value_col] <= p_value_cutoff]
    significant_genes = significant_genes[
    (significant_genes["log2FoldChange"] >= log2fc_cutoff[1]) |
    (significant_genes["log2FoldChange"] <= log2fc_cutoff[0])
    ]
    gp = GProfiler(return_dataframe=True)
    go_results = gp.profile(
        organism=go_organism,
        query=list(significant_genes["Symbol"]),
        sources=go_sources,
        no_evidences=False
    )
    # --- 1. Prepare the data for plotting ---
    top_terms = go_results.head(n_top_terms).copy()
    top_terms['log_p_val'] = -np.log10(top_terms['p_value'])
    top_terms = top_terms.sort_values(by='log_p_val', ascending=True)

    # --- 2. Set up the custom colormap and normalization ---
    norm = mcolors.Normalize(vmin=lfc_range[0], vmax=lfc_range[1])
    
    # THE FIX 1: Dynamically get colors from the palette based on the contrast
    color_neg = palette[contrast[0]] # e.g., "Untreated" color
    color_pos = palette[contrast[1]] # e.g., "Treated" color
    
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_diverging_map", [color_neg, "white", color_pos]
    )

    # --- 3. Create the plot ---
    fig, ax = plt.subplots(figsize=figure_size)
    y_labels = []
    y_positions = np.arange(len(top_terms))

    for i, row in enumerate(top_terms.itertuples()):
        y_labels.append(row.name)
        bar_length = row.log_p_val
        
        # RESTORED FIX: Split the intersection string to get the list of genes
        intersecting_genes = row.intersections
        valid_genes = [gene for gene in intersecting_genes if gene in deseq_results.index]
        if not valid_genes:
            continue
            
        gene_lfc = deseq_results.loc[valid_genes]['log2FoldChange'].sort_values()
        
        if len(gene_lfc) == 0: continue

        segment_width = bar_length / len(gene_lfc)
        current_pos = 0
        
        for lfc in gene_lfc:
            color = custom_cmap(norm(lfc))
            ax.barh(
                y=i, width=segment_width, left=current_pos,
                color=color, edgecolor='white', linewidth=0
            )
            current_pos += segment_width

    # --- 4. Finalize and style the plot ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel('-log10(p-value)', fontsize=14)
    ax.set_title('Top Enriched GO Pathways', fontsize=16)
    
    # THE FIX 2: Use the 'custom_cmap' for the colorbar, not an undefined variable
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('log2 Fold Change', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

    if plot_name is not None:
        for plot_format in plot_formats:
            plt.savefig(f"figures/go_enrichment_{plot_name}.{plot_format}", dpi=300)
    plt.show()

def plot_logfc_correlation(
    deg_dfs: list,
    conditions: list,
    palette: dict,
    p_value_cutoff = (0.05, 0.05),
    p_value_col = "padj",
    log2fc_cutoff: tuple = ((-1,1),(-1,1)),
    label_top_genes: int = 20,
    plot_name: str = None,
    plot_format: list[str] = [],
    figure_size: tuple = (7, 5)
):
    for i, deg_df in enumerate(deg_dfs):
        deg_df = deg_df.rename(
            columns={
                "log2FoldChange": f"log2fc_{i}",
                "padj": f"padj_{i}"
    })[['Symbol', f'log2fc_{i}', f'padj_{i}']]
        deg_dfs[i] = deg_df
        
    merged_df = pd.merge(deg_dfs[0], deg_dfs[1], on='Symbol', how='outer')

    up_0 = set(
        merged_df.loc[
        (merged_df['padj_0'] <= p_value_cutoff[0]) & 
        (merged_df['log2fc_0'] >= log2fc_cutoff[0][1]), 
        'Symbol'])
    up_1 = set(
        merged_df.loc[
        (merged_df['padj_1'] <= p_value_cutoff[1]) & 
        (merged_df['log2fc_1'] >= log2fc_cutoff[1][1]), 
        'Symbol'])
    
    # Downregulated genes (log2fc < -1 and significant)

    down_0 = set(
        merged_df.loc[
        (merged_df['padj_0'] <= p_value_cutoff[0]) & 
        (merged_df['log2fc_0'] <= log2fc_cutoff[0][0]), 
        'Symbol'])
    down_1 = set(
        merged_df.loc[
        (merged_df['padj_1'] <= p_value_cutoff[1]) & 
        (merged_df['log2fc_1'] <= log2fc_cutoff[1][0]), 
        'Symbol'])

    upregulated_dict = {conditions[0]: up_0, conditions[1]: up_1}
    downregulated_dict = {conditions[0]: down_0, conditions[1]: down_1}

    
    condition_colors_ordered = {key: palette[key] for key in conditions if key in palette}
    hex_colors = list(condition_colors_ordered.values())

    rgba_colors = [(r, g, b, 0.5) for r, g, b, _ in [mcolors.to_rgba(color) for color in hex_colors]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Upregulated Venn Diagram ---
    petal_labels_up = generate_petal_labels(upregulated_dict.values(), fmt="{size}")
    draw_venn(
        petal_labels=petal_labels_up,
        dataset_labels=upregulated_dict.keys(),
        hint_hidden=False,
        colors=rgba_colors,  # Use the custom colors
        figsize=(3, 3),
        fontsize=40,
        legend_loc="best",
        ax=ax1
    )
    ax1.set_title("Commonly Upregulated Genes", fontsize=16)
    
    # --- Downregulated Venn Diagram ---
    petal_labels_down = generate_petal_labels(downregulated_dict.values(), fmt="{size}")
    draw_venn(
        petal_labels=petal_labels_down,
        dataset_labels=downregulated_dict.keys(),
        hint_hidden=False,
        colors=rgba_colors,  # Use the custom colors
        figsize=(3, 3),
        fontsize=40,
        legend_loc="best",
        ax=ax2
    )
    ax2.set_title("Commonly Downregulated Genes", fontsize=16)
    plt.tight_layout()
    
    if plot_name is not None:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        for plot_format in plot_formats:
            plt.savefig(f"{save_dir}/venncorrelation_{plot_name}_{conditions[0]}vs{conditions[1]}.{plot_format}", dpi=300)
        os.makedirs("DEGs", exist_ok=True)
        merged_df.to_csv(f"DEGs/venn_{plot_name}.csv")
    plt.show()

    ## correlation plot
    # merged_df = merged_df[(merged_df["padj_0"] <= p_value_cutoff[0]) & (merged_df["padj_1"] <= p_value_cutoff[1])]
    criteria = [
        (merged_df['padj_0'] <= p_value_cutoff[0]) & (merged_df['padj_1'] <= p_value_cutoff[1]),
        (merged_df['padj_0'] <= p_value_cutoff[0]),
        (merged_df['padj_1'] <= p_value_cutoff[1])
    ]
    choices = ['Significant in Both', f'Significant in {criteria[0]}', f'Significant in {criteria[1]}']
    merged_df['Significance'] = np.select(criteria, choices, default='Not Significant')
    
    merged_df['combined_fcs'] = abs(merged_df["log2fc_0"])+abs(merged_df["log2fc_1"])
    merged_df = merged_df.dropna()
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df["log2fc_0"], merged_df["log2fc_1"])
    r_squared = r_value**2
    
    top_genes = merged_df.sort_values("combined_fcs", key=abs, ascending=False).head(label_top_genes)
    
    # Create the plot
    plt.figure(figsize=figure_size)
    sns.scatterplot(
        data=merged_df,
        x='log2fc_0',
        y='log2fc_1',
        hue='Significance',
        hue_order=['Significant in Both', 
                   f'Significant in {conditions[0]}', 
                   f'Significant in {conditions[1]}', 
                   'Not Significant'],
        palette={'Significant in Both': '#1f1f1f', 
                 f'Significant in {conditions[0]}': palette[conditions[0]], 
                 f'Significant in {conditions[1]}': palette[conditions[1]], 
                 'Not Significant': 'lightgray'},
        alpha=0.7,
        s=50,
        edgecolor='black',
        linewidth=0.5,
        zorder = 5, legend=True
    )
    
    texts = []
    for i, row in top_genes.iterrows():
        texts.append(plt.text(row['log2fc_0'], row['log2fc_1'], row['Symbol'], fontsize=12, ha='center', zorder=10, 
                              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')))
    
    # Use adjust_text to prevent labels from overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    # Add lines for reference
    plt.plot(merged_df["log2fc_0"], intercept + slope*merged_df["log2fc_0"], 'r', label='Fitted line')
    plt.axhline(0, color='gray', linestyle='--', zorder=0)
    plt.axvline(0, color='gray', linestyle='--', zorder=1)
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$\n y = {intercept:.2f} + x * {slope:.2f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top')

    plt.xlabel(f'{conditions[0]} (log2FC)', fontsize=12)
    plt.ylabel(f'{conditions[1]} (log2FC)', fontsize=12)
    # plt.savefig(f"figures/correlation_{plot_name}.png")
    # plt.savefig(f"figures/correlation_{plot_name}.svg")
    plt.show()
    return merged_df, (upregulated_dict, downregulated_dict)

def get_venn_regions(data_dict: dict) -> dict:
    """
    Calculates the exclusive items for every possible intersection of keys in a dictionary.

    Args:
        data_dict: A dictionary where keys are identifiers and values are lists of items.

    Returns:
        A dictionary where keys are strings representing the combination of input keys
        (e.g., "A & B") and values are the sets of items found only in that
        specific combination.
    """
    # Convert lists to sets for efficient operations
    sets = {key: set(value) for key, value in data_dict.items()}
    all_keys = list(sets.keys())
    results = {}

    # Generate the power set of keys (all combinations from size 1 up to all keys)
    key_combinations = chain.from_iterable(
        combinations(all_keys, r) for r in range(1, len(all_keys) + 1)
    )

    for combo in key_combinations:
        # Find the intersection of all sets in the current combination
        in_group = [sets[k] for k in combo]
        intersection = set.intersection(*in_group)

        # Find the union of all sets NOT in the current combination
        out_group_keys = [k for k in all_keys if k not in combo]
        out_group = [sets[k] for k in out_group_keys]
        # Use an empty set as the start value for union
        union_of_others = set.union(set(), *out_group)

        # The exclusive items are those in the intersection minus those in any other set
        exclusive_items = intersection - union_of_others

        # Only add non-empty sets to the results
        if exclusive_items:
            result_key = ' & '.join(sorted(combo))
            results[result_key] = list(exclusive_items)

    return results

def show_off(
    metadata: pd.DataFrame,
    highlight_genes: dict,
    palette: dict = None,
    conditions: list = None,
    available_metadata_columns: list = ["Condition"],
    plot_name:str = None,
    plot_formats: list = []
    
):
    if conditions is None:
        try:
            conditions = list(metadata[available_metadata_columns[0]].cat.categories)
        except AttributError:
            conditions = list(set(metadata[available_metadata_columns[0]]))
    if palette is None:
        palette = _get_color_palette(conditions)
        if len(available_metadata_columns) > 1:
            for exp in set(metadata[available_metadata_columns[1]]):
                palette[exp] = list(_get_color_palette(exp).values())[0]
    
    dds, results = get_dds_results(
        metadata = metadata,
        ref_condition=conditions[0]
    )

    plot_tpm_bar(
        dds = dds,
        conditions = conditions,
        genes = list(highlight_genes.values())[0],
        palette = palette,
        plot_points = True,
        plot_name = plot_name,
        plot_formats = plot_formats
    )
    
    expression_df, row_clusters = plot_expression_heatmap(
        metadata = metadata,
        dds_dict = (dds, results),
        highlight_genes=highlight_genes,
        palette = palette,
        available_metadata_columns = available_metadata_columns,
        ref_condition=conditions[0],
        n_row_clusters=2,
        plot_name = plot_name,
        plot_formats = plot_formats
    )
    
    plot_go(
        row_clusters,
        plot_name = plot_name,
        plot_formats = plot_formats
    )

    plot_gene_curves(
        metadata = metadata,
        expression_data = expression_df,
        genes_to_plot = list(highlight_genes.values())[0],
        category_column = available_metadata_columns[0],
        plot_order = metadata[available_metadata_columns[0]].cat.categories,
        plot_name = plot_name,
        plot_formats = plot_formats
    )

    for cluster in set(row_clusters["Cluster"]):
        cluster_genes = list(row_clusters[row_clusters["Cluster"]==cluster].index)
        if len(cluster_genes) > 0:
            print(len(cluster_genes))
            plot_cluster_curve(
                metadata=metadata, 
                expression_data = expression_df, 
                genes_to_plot = cluster_genes, 
                cluster = cluster,
                category_column = available_metadata_columns[0],
                plot_order = metadata[available_metadata_columns[0]].cat.categories,
                plot_name = plot_name,
                plot_formats = plot_formats
            )

    plot_tpm_curve(
        dds = dds,
        palette = palette,
        highlight_genes = highlight_genes,
        conditions = conditions,
        plot_name = plot_name,
        plot_formats = plot_formats
    )
    
    plot_tpm_curve_comparison(
        dds = dds,
        conditions = conditions,
        highlight_genes = highlight_genes,
        palette = palette,
        plot_name = plot_name,
        plot_formats = plot_formats
    )

    for cond in conditions[1:]:
        plot_volcano(
            deseq_results = results[cond],
            contrast = [conditions[0], cond],
            highlight_genes=highlight_genes,
            palette=palette,
            plot_name = plot_name,
            plot_formats = plot_formats
        )
    
        plot_go_enrichment(
            deseq_results = results[cond],
            contrast = [conditions[0], cond],
            palette = palette,
            plot_name = plot_name,
            plot_formats = plot_formats
        )

    return dds, results, palette