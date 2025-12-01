'''
This module provides a comprehensive suite of visualization and analysis tools
for single-cell RNA sequencing (scRNA-seq) data, leveraging the adata structure
for organizing and managing the data. It includes functions to visualize quality control (QC),
clustering results, and differentially expressed genes (DEGs) across clusters and conditions.

Key functionalities include:

1. **Visualization of DEGs and Clustering**:
   - ``plot_per_group_DEG_dotplot_n_gene_dendrogram``: Generates dot plots and dendrograms for DEGs per cluster.
   - ``plot_per_group_stacked_violins``: Produces stacked violin plots for visualizing gene expression.
   - ``plot_per_group_DEG_umaps``: Creates UMAP visualizations for DEGs per cluster.
   - ``get_DEGs_per_group_for_plotting``: Retrieves DEGs for plotting based on various filtering criteria.
   - ``create_gene_dendrogram``: Generates and saves a dendrogram for specified genes.

2. **Quality Control and Summary Visualizations**:
   - ``plot_hist_of_each_layer``: Displays histograms of each layer in an adata object.
   - ``plot_qc_violins``: Generates violin plots for quality control metrics.
   - ``plot_qc_scatter``: Creates scatter plots for QC metrics.
   - ``plot_qc_scatter_combinations``: Generates scatter plots for combinations of QC metrics.
   - ``plot_filtering``: Visualizes QC steps with violins and scatterplots.
   - ``show_num_cells_and_count_per_gene``: Logs the number of cells and counts per gene.

3. **Specialized Plotting**:
   - ``plot_ridgeline``: Creates a ridgeline plot for visualizing overlapping data distributions.
   - ``plot_density_scatter``: Generates density scatter plots for cell counts vs. genes.
   - ``plot_ref_dotplots``: Creates dot plots for marker genes per cluster.
   - ``plot_ref_stacked_violins``: Generates stacked violin plots for marker genes.
   - ``plot_mediods_heatmap``: Creates a heatmap for gene expression across clusters.

4. **UMAP Plot Helpers**:
   - ``discrete_umap_helper``: Creates a discretized UMAP plot for specific markers.
   - ``continuos_umap_helper``: Generates continuous UMAP plots for visualizing gradients of marker genes.
   - ``plot_umap``: Wrapper function for creating UMAP plots with custom configurations.

Additional Details:
    - The module is designed for use with the adata object from the ``scanpy`` library,
      and many functions rely on precomputed data within this object.
    - Funcations often provide options for customizing the output, such as specifying clusters,
      adjusting plot aesthetics, and saving figures to specified directories.
    - Dependencies include ``scanpy``, ``matplotlib``, and other common Python libraries for data
      manipulation and visualization.

This module is intended for use in exploratory data analysis and publication-quality visualizations
of single-cell RNA-seq data.
'''


from scToolkit import (  # noqa: F401
    # Standard Library Imports
    os_path, remove, combinations, Any, time,  # noqa: F401
    warnings, mpl, Sequence,
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed,  # noqa: F401
    # Pandas and NumPy for data manipulation and analysis and sklearn
    np, pd, MinMaxScaler, iqr, dendrogram, linkage, set_link_color_palette,   # noqa: F401
    leaves_list,
    # Matplotlib for plotting and visualization
    plt, LinearSegmentedColormap, clustermap, Wedge, Normalize, mpl_rectangle,
    mpl_path, TwoSlopeNorm,
    ListedColormap, cm, colormaps, mcolors, MarkerStyle, Colormap,  # noqa: F401
    BoundaryNorm, to_rgba, is_color_like, rc_context,
    # Marsilea for plotting and visualization
    Heatmap, SizedMesh, SizedHeatmap, WhiteBoard, Labels, Title,
    # Legendkit for plotting and visualization
    vstack, colorart,  # lk_legend
    # Scanpy and Anndata for single-cell data analysis
    sc, ad,
    # Scipy for scientific and technical computing
    hierarchy, jaccard_score,
    # Sklearn
    KMedoids, FFTKDE, gaussian_kde,
    # ##########################
    # scToolkit specifics
    # Logging Functions
    get_logger,
    # Custom Analysis Functions
    get_all_markers_from_ref_dict,
    get_DEGs_per_group, get_save_path, get_group_hierarchy,
    subset_degs, get_msigdb_df, map_geneset_to_degs,
    count_genesets_per_group, get_adata_sub_keys, replace_small_counts,
    ref_dict_sort_values_hvg_like, ref_dict_long_value_split,
    create_acronym, ensure_unique_acronyms, calc_group_density,
    get_rows_cols_figsize, get_deg_df, clip_dataframe,
    min_max_scaler, min_max_scale_axis, get_colors_wrapped,
    deduplicate_ref_dict, get_categorical_columns, get_n_unique,
    validate_groupby_column, _validate_thresholds, get_thresholds,
    # ###########################
    # unsorted
    distinctipy, violinplot, stripplot, copy, heatmap)
from .sc_volcano import plot_volcano


logger = get_logger(name="sc_plots")


# ###################################################################################################
# Plotting helpers
def show_num_cells_and_count_per_gene(
            adata: ad.AnnData,
            genelist: list[str]
        ) -> None:
    """Logs the number of cells and total counts per gene for a given list of genes.

    This function retrieves and logs metadata for each gene in the provided list
    using the ``adata.var`` annotations. It reports the number of cells in which
    each gene is expressed and the total counts observed across all cells. This
    is useful for diagnostics and exploratory data analysis, particularly when
    inspecting the sparsity and expression level of selected genes.

    NOTE:
        This function assumes that ``adata.var`` contains the keys ``n_cells``
        and ``n_counts`` for each gene, which may need to be precomputed using
        appropriate preprocessing tools.

    Args:
        adata (anndata.AnnData): Adata object containing gene metadata in
            ``adata.var``.
        genelist (list[str]): List of gene names to retrieve the cell and count
            information for.

    Returns:
        None:
            This function does not return any value. It logs information about
            each gene.

    Raises:
        KeyError: If a gene in ``genelist`` is not found in ``adata.var`` or the
            expected metadata keys ``n_cells`` or ``n_counts`` are missing.

    TODO:
        Add error handling for missing genes or missing metadata fields in
        ``adata.var``.

    Tags:
        qc, summary, var
    """
    # #########################################################
    # Iterate over each gene and log number of expressing cells and total count
    for g in genelist:
        # ##########################################
        # Log metadata for the given gene
        logger.info(
            f"Gene: {g}, n_cells: {adata.var.n_cells[g]}, n_counts: {adata.var.n_counts[g]}")


def get_DEGs_per_group_for_plotting(
            adata: ad.AnnData,
            top_n_genes: int = 5,
            groups: list[str] | None = None,
            overlap_fill_to_top_n: bool = False,
            perc: float = 0.15,
            p_val_cutoff: float = 0.01,
            lfc: float = 0.01,
            direction: str = "force_up_n_down",
            deg_key: str = "rank_genes_groups",
        ) -> tuple[dict[str, list[str]], list[str]]:
    """Retrieves differentially expressed genes (DEGs) per group/cluster for plotting.

    This function identifies DEGs across specified groups from the provided
    adata. It supports directional filtering, p-value and log fold
    change thresholds, and optional overlap handling between groups. The top N
    genes per group are returned for visualization or further processing.

    Args:
        adata (anndata.AnnData): Adata object containing DEGs.
        top_n_genes (int, optional): Number of top genes to select per group.
            Defaults to 5.
        groups (list[str] | None, optional): Specific groups to analyze.
            Defaults to None.
        overlap_fill_to_top_n (bool, optional): If True, includes new genes to
            maintain top N genes even if some overlap with previously selected
            genes. Defaults to False.
        perc (float, optional): Percentile cutoff used when computing DEGs.
            Defaults to 0.15.
        p_val_cutoff (float, optional): Adjusted p-value cutoff for selecting
            significant DEGs. Defaults to 0.01.
        lfc (float, optional): Log fold change threshold for filtering DEGs.
            Defaults to 0.01.
        direction (str, optional): Direction of DEGs to include
            ("force_up_n_down", "up", "down"). Defaults to "force_up_n_down".
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".

    Returns:
        tuple[dict[str, list[str]], list[str]]:

            - variables_per_group (dict[str, list[str]]): Dictionary with group
              names as keys and lists of selected genes as values.
            - all_variables_per_group (list[str]): Combined list of all selected
              genes across groups, without duplicates.

    Calls:
        get_DEGs_per_group

    Called By:
        plot_per_group_DEG_dotplot_n_gene_dendrogram,
        plot_per_group_stacked_violins

    TODO:
        Add support for additional filtering strategies and integrate with
        plotting functions.
    """
    # ########################################################################################
    # define containers
    variables_per_group = {}
    all_variables_per_group = []
    # ########################################################################################
    # Handle the "force_up_n_down" direction, which considers both upregulated and downregulated genes
    if direction == "force_up_n_down":
        variables_up = get_DEGs_per_group(
            adata, groups=groups, n_genes=1e200, perc=perc, deg_key=deg_key,
            p_val_cutoff=p_val_cutoff, lfc=lfc, direction="up")
        variables_down = get_DEGs_per_group(
            adata, groups=groups, n_genes=1e200, perc=perc, deg_key=deg_key,
            p_val_cutoff=p_val_cutoff, lfc=lfc, direction="down")
        # #####################################################
        # For each group, select top_n_genes from both up and downregulated genes
        for c in variables_up.keys():
            if overlap_fill_to_top_n:
                genes_to_add_up = [
                    x for x in variables_up[c] if x not in all_variables_per_group]
                genes_to_add_down = [
                    x for x in variables_down[c] if x not in all_variables_per_group]
            else:
                genes_to_add_up = [
                    x for x in variables_up[c][:top_n_genes]
                    if x not in all_variables_per_group]
                genes_to_add_down = [
                    x for x in variables_down[c][:top_n_genes]
                    if x not in all_variables_per_group]

            genes_to_add = (genes_to_add_up[:top_n_genes // 2]
                            + genes_to_add_down[:top_n_genes // 2])

            # Check if group genes are empty before adding
            if len(genes_to_add) > 0:
                all_variables_per_group.extend(genes_to_add)
                variables_per_group[c] = genes_to_add

    else:
        # #####################################################
        # Handle cases where a specific direction other than "force_up_n_down" is provided
        variables = get_DEGs_per_group(
            adata, groups=groups, n_genes=None, perc=perc, deg_key=deg_key,
            p_val_cutoff=p_val_cutoff, lfc=lfc, direction=direction)

        for c in variables.keys():
            if overlap_fill_to_top_n:
                genes_to_add = [
                    x for x in variables[c] if x not in all_variables_per_group]
            else:
                genes_to_add = [
                    x for x in variables[c][:top_n_genes]
                    if x not in all_variables_per_group]

            genes_to_add = genes_to_add[:top_n_genes]

            # Check if group genes are empty before adding
            if len(genes_to_add) > 0:
                all_variables_per_group.extend(genes_to_add)
                variables_per_group[c] = genes_to_add
    # ########################################################################################
    return variables_per_group, all_variables_per_group


def get_data(
            adata: ad.AnnData,
            keys: list[str],
            obsm_keys: list[str] | None = None,
            groupby: str | None = None
        ) -> pd.DataFrame:
    """Extracts specified keys from an adata object into a DataFrame.

    NOTE:
        DEPRECATED! Use get_adata_sub_keys.

    Args:
        adata (anndata.AnnData): Adata object.
        keys (list[str]): List of keys to extract.
        obsm_keys (list[str] | None, optional): Optional list of keys in the
            obsm attribute to search. Defaults to None.
        groupby (str | None, optional): Optional key to group data by, must be a
            column in adata.obs. Defaults to None.

    Returns:
        pandas.DataFrame:
            Containing the extracted data.

    Calls:
        validate_groupby_column

    Tags:
        extraction, groupby, obs, utils, var
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    # #########################################################
    data = pd.DataFrame()

    for key in keys:
        key_found = False  # Track if key is found in any location
        # #########################################################
        # Search in obsm if obsm_keys are provided
        if obsm_keys is not None:
            for obsm_key in obsm_keys:
                if obsm_key not in adata.obsm or not isinstance(adata.obsm[obsm_key], pd.DataFrame):
                    continue

                if key in adata.obsm[obsm_key].columns:
                    data[key] = adata.obsm[obsm_key][key]
                    key_found = True
                    break  # Exit the loop once the key is found
        # #########################################################
        # If not found in obsm, search in adata.var or adata.obs
        if not key_found:
            if key in adata.var_names:
                # Key refers to a gene
                gene_data = adata[:, key].X
                data[key] = gene_data.toarray().flatten() if hasattr(gene_data, "toarray") else gene_data.flatten()
                key_found = True
            # #########################################################
            elif key in adata.obs.columns:
                # Key refers to an obs-level key
                data[key] = adata.obs[key].values
                key_found = True

        if not key_found:
            raise ValueError(f"Key {key} not found in adata.obsm[{obsm_keys}], adata.var_names, or adata.obs.columns.")

    if groupby is not None:
        data[groupby] = adata.obs[groupby].values

    return data


def get_bandwidth(
            data: np.ndarray,
            bw: int | float | str,
            categories: np.ndarray
        ) -> float:
    """Calculate the bandwidth for kernel density estimation.

    Args:
        data (numpy.ndarray): The input data array.
        bw (int | float | str): Bandwidth value or method. Accepts
            numeric values or one of 'sheather_jones', 'scott', 'silverman'.
        categories (numpy.ndarray): Array of category labels.

    Returns:
        float:
            Computed bandwidth.

    Raises:
        ValueError: If an invalid bandwidth method is provided.

    Called By:
        plot_embedding_density

    Tags:
        calculation, trajectory, utils
    """
    # ############################################################
    # calculate bandwith
    if isinstance(bw, (int, float)):
        bw_ = bw
    elif isinstance(bw, str):
        n = len(categories)
        d = 2
        std_dev = np.std(data, axis=0).mean()
        # print(np.std(data, axis=0))
        if bw == "sheather_jones":
            # https://homepage.stat.uiowa.edu/~luke/classes/STAT7400-2023/slides/smooth.html#26
            data_iqr = iqr(data)
            bw_ = (0.9 * min(std_dev, data_iqr / 1.34)) * (n ** (-1 / 5))
        elif bw == "scott":
            bw_ = np.power(n, -1 / (d + 4)) * std_dev
        elif bw == "silverman":
            bw_ = np.power(n * (d + 2) / 4.0, -1 / (d + 4)) * std_dev
        else:
            raise ValueError(
                "bw param: For Strings, please provide one of "
                "'sheather_jones', 'scott', 'silverman'")
    else:
        raise ValueError(
            "bw param: Please provide a bw method as string or numeric.")
    return bw_


def split_title_text(
            text: str,
            max_chars: int = 45,
            force_word_split: bool = True
        ) -> str:
    """Split text into lines of at most ``max_chars`` characters for plotting.

    Args:
        text (str): The input string to split.
        max_chars (int, optional): Maximum characters per line. Defaults to 45.
        force_word_split (bool, optional): Whether to split long words
            arbitrarily if needed. Defaults to True.

    Returns:
        str:
            A string with line breaks.

    Called By:
        plot_embedding_density, plot_umap, plot_violin

    Tags:
        plotting, utils
    """
    lines, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if len(word) > max_chars and force_word_split:
                if current:
                    lines.append(current)
                    current = ""
                while len(word) > max_chars:
                    lines.append(word[:max_chars])
                    word = word[max_chars:]
                current = word
            else:
                if current:
                    lines.append(current)
                current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def create_gene_dendrogram(
            adata: ad.AnnData,
            geneset: list[str],
            fig_name: str,
            top_n_genes: str = "",
            figsize: Sequence[float | int] = (10, 5),
            part: str = "/downstream/"
        ) -> None:
    """Create and save a dendrogram plot for a given set of genes.

    Generates a dendrogram using hierarchical clustering for a specified gene
    set in the adata object and saves the resulting plot as a PDF. The
    dendrogram is saved with the provided figure name and optional suffix.

    Args:
        adata (anndata.AnnData): Adata object.
        geneset (list[str]): List of gene names to include in the dendrogram.
        fig_name (str): Base name of the output figure.
        top_n_genes (str, optional): Optional suffix for the filename to
            indicate top N genes. Defaults to "".
        figsize (Sequence[float | int], optional): Width and height of the output figure in
            inches.
        part (str, optional): Path to the directory where the figure will be
            saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified output directory does not exist.

    Called By:
        plot_per_group_DEG_dotplot_n_gene_dendrogram

    TODO:
        Validate that all genes in ``geneset`` exist in ``adata.var_names`` before
        plotting.

    Tags:
        DEG, clustering, io, var, visualization
    """
    # #########################################################
    # Prepare the gene expression data for the specified gene set
    ytdist = adata[:, np.in1d(adata.var_names.values.astype(str), geneset)].X.toarray().T
    # Perform hierarchical clustering on the gene expression data
    linkage_ = linkage(ytdist, 'ward')
    # Set up the figure and plot the dendrogram
    plt.figure(figsize=figsize)
    _ = dendrogram(  # noqa: F841
            linkage_, labels=list(adata.var_names[np.in1d(adata.var_names.values.astype(str), geneset)]))
    # Reset the link color palette to default after use
    set_link_color_palette(None)
    # #########################################################
    # Determine the output file name based on whether top_n_genes is provided
    if top_n_genes:
        string = f"{fig_name}_top{top_n_genes}.pdf"
    else:
        string = f"{fig_name}.pdf"
    # Remove the file if it already exists to avoid overwriting issues
    if os_path.exists(string):
        remove(string)
    # #########################################################
    # Save the generated dendrogram as a PDF file
    plt.savefig(string)


def calc_dotplot_figsize(
            shape: tuple[int, int],
            xlabels: list[str],
            ylabels: list[str],
            font_size: float = 10,        # Default font size in points (used for all text elements)
            dpi: int = 100,               # Standard figure resolution (dots per inch)
            dot_spacing: float = .85,     # Grid spacing between dot centers, in "dot units"
            dot_unit: float = 0.25,       # Physical size of one "dot unit" in inches (e.g. 0.25in ~ 6.35mm)
            legend_stack: int = 2,        # Number of rows to split the legend into (to avoid long single-row legends)
            legend_box_pad: float = 1.2,  # Vertical spacing multiplier between legend entries (for readability)
            title_height: float = 0.4     # Reserved vertical space (in inches) for a potential title
        ) -> tuple[float, float]:
    """
    Compute a content-aware figure size (in inches) for a Marsilea split-dotplot.

    All inputs are interpreted as physical constraints and rendered elements.
    The function ensures that the largest dot fits cleanly into the spacing and
    that labels and legends are properly padded.

    This considers:
      - Dot matrix size based on dot count and spacing.
      - Max circle radius for visual margin.
      - Label size from longest x and y labels.
      - Legend box size from estimated text width/height.
      - Top padding for an optional title.

    Args:
        shape: (rows, cols) of the dotplot grid.
        xlabels: List of bottom axis labels.
        ylabels: List of left axis labels.
        font_size: Font size in points used for labels and legends.
        dpi: Resolution in dots per inch. Controls pt to inch conversion.
        dot_spacing: Distance (in "dot units") between dot centers.
        dot_unit: Physical size (in inches) of one spacing unit.
        legend_stack: Number of legend rows.
        legend_box_pad: Vertical padding multiplier for legends.
        title_height: Vertical padding (in inches) to reserve for title.

    Returns:
        tuple[float, float]:
              Tuple of (width, height) in inches to pass to matplotlib or
              Marsilea board.

    Called By:
        plot_dotplot, plot_proximap, plot_split_dotplot
    """
    rows, cols = shape

    # Use the largest radius across both dot matrices
    # max_radius = max(size1.max(), size2.max())  # ∈ [0,1]
    # dot_diam = 2 * max_radius * dot_unit        # largest possible circle diameter (in inches)

    # Effective spacing between dot centers must at least fit the circle
    spacing = dot_spacing * dot_unit  # max(dot_spacing * dot_unit, dot_diam)

    # Compute grid size (dot matrix dimensions only)
    matrix_width = cols * spacing
    matrix_height = rows * spacing

    # Estimate label widths:
    # - Each character is ~0.6 pt wide in sans-serif/monospace style
    # - Convert pt to px to inches
    CHAR_WIDTH_PT = 0.6  # Empirical factor: 1 char ≈ 0.6 pt
    x_label_len = max(len(str(lbl)) for lbl in xlabels)
    y_label_len = max(len(str(lbl)) for lbl in ylabels)
    x_label_space = (x_label_len * font_size * CHAR_WIDTH_PT) / dpi
    y_label_space = (y_label_len * font_size * CHAR_WIDTH_PT) / dpi

    # Estimate legend width:
    # - Each entry assumes 6pt width including color box and padding
    LEGEND_CHAR_WIDTH_PT = 6.0
    legend_cols = np.ceil(cols / legend_stack)
    legend_width = (LEGEND_CHAR_WIDTH_PT * legend_cols) / dpi

    # Estimate legend height:
    # - font_size per row, scaled by padding multiplier
    legend_height = (font_size * legend_stack * legend_box_pad) / dpi

    # Final figure dimensions
    width = matrix_width + y_label_space + legend_width
    height = matrix_height + x_label_space + legend_height + title_height

    # # Debug print of all contributing components
    # print(f"{'--- Dotplot Figure Size Debug ---'}")
    # print(f"cols               = {cols}")
    # print(f"rows               = {rows}")
    # print(f"font_size          = {font_size} pt")
    # print(f"dpi                = {dpi}")
    # print(f"dot_spacing        = {dot_spacing} units")
    # print(f"dot_unit           = {dot_unit:.2f} in")
    # # print(f"max_radius         = {max_radius:.2f}")
    # # print(f"dot_diam           = {dot_diam:.2f} in")
    # print(f"effective_spacing  = {spacing:.2f} in")
    # print()
    # print(f"matrix_width       = {matrix_width:.2f} in")
    # print(f"y_label_space      = {y_label_space:.2f} in")
    # print(f"legend_width       = {legend_width:.2f} in")
    # print(f"TOTAL width        = {width:.2f} in")
    # print()
    # print(f"matrix_height      = {matrix_height:.2f} in")
    # print(f"x_label_space      = {x_label_space:.2f} in")
    # print(f"legend_height      = {legend_height:.2f} in")
    # print(f"title_height        = {title_height:.2f} in")
    # print(f"TOTAL height       = {height:.2f} in")
    # print(f"{'-'*40}")
    return round(width, 2), round(height, 2)


def get_colormap(
            cmap: str = "turbo",
            fade_alpha: bool = False,
            alphas: np.ndarray | None = None,
            set_alpha: float | None = None,
            zero_alpha_only: bool = False
        ) -> ListedColormap:
    """Returns a custom or matplotlib colormap with optional alpha fading.

    Args:
        cmap (str, optional): Name of the colormap or "R" for custom diverging
            map. Defaults to "turbo".
        fade_alpha (bool, optional): Whether to fade the alpha channel.
            Defaults to False.
        alphas (numpy.ndarray | None, optional): Optional custom alpha values.
            Defaults to None.
        set_alpha (float | None, optional): If given (0–1), sets early fraction
            of alpha to 0. Defaults to None.
        zero_alpha_only (bool, optional): If True, sets only the first alpha
            value to 0, overrides set_alpha. Defaults to False.

    Returns:
        ListedColormap:
            A colormap with appropriate alpha
            settings.

    Raises:
        ValueError: If ``cmap`` is invalid and not "R".

    Calls:
        get_colormap.create_alphas

    Called By:
        continuos_umap_helper, discrete_umap_helper, plot_mediods_heatmap,
        plot_per_group_DEG_dotplot_n_gene_dendrogram, plot_ref_dotplots,
        plot_ref_stacked_violins, run_downstream

    TODO:
        Add check to prevent ``alphas[0] = 0`` when ``alphas`` is None and
        ``zero_alpha_only=True``, to avoid TypeError.

    Tags:
        utils, visualization
    """
    def create_alphas(n: int) -> np.ndarray:
        """
        Helper for alphachannel creation.

        Args:
            n (int): Number of alpha values to generate.

        Returns:
            np.ndarray:
                Array of alpha values.
        """
        a = 1 / (1 + np.exp(-np.linspace(-1, 6, n)))
        return (a - a.min()) / (a.max() - a.min())

    if cmap == "R":
        n_colors = 1000
        if fade_alpha:
            if alphas is None:
                alphas = create_alphas(n_colors)
            else:
                n_colors = len(alphas)
        elif zero_alpha_only:
            alphas[0] = 0
        else:
            alphas = np.ones(n_colors)

        if set_alpha is not None:
            alphas[:int(len(alphas) * set_alpha)] = 0

        base_colors = [
            (2/256, 63/256, 165/256),
            (161/256, 166/256, 200/256),
            (226/256, 226/256, 226/256),
            (202/256, 156/256, 164/256),
            (142/256, 6/256, 59/256)]
        cmap = LinearSegmentedColormap.from_list("red_yellow_green", base_colors, N=n_colors)
        colors = cmap(np.linspace(0, 1, n_colors))
        colors[:, -1] = alphas
        return ListedColormap(colors)

    elif cmap not in colormaps():
        logger.error(f'Colormap {cmap} not found. Falling back to "RdGy_r".')
        cmap = "RdGy_r"

    discretized_cmap = getattr(cm, cmap)
    n_colors = discretized_cmap.N

    if fade_alpha:
        if alphas is None:
            alphas = create_alphas(n_colors)
        else:
            n_colors = len(alphas)
    else:
        alphas = np.ones(n_colors)

    if zero_alpha_only:
        alphas[0] = 0

    elif set_alpha is not None:
        alphas[:int(len(alphas) * set_alpha)] = 0

    colors = np.array(discretized_cmap(np.linspace(0, 1, n_colors)))[:, :3]
    cmap_with_alpha = np.hstack([colors, np.array(alphas).reshape(-1, 1)])
    return ListedColormap(cmap_with_alpha)


# ###################################################################################################
# Default plots
def plot_heatmap(
            adata: ad.AnnData,
            keys: str | list[str],
            obsm_keys: list[str] | None = None,
            groupby: str | None = None,
            condition_obs_key: str | None = None,
            conditions: list[str] | None = None,
            var_keys_only: bool = False,
            save_path: str | None = None,
            base_figsize: Sequence[float | int] = (6, 4),
            cmap: str = "turbo",
            figsize_scale: float = 1.0,
            vmin: float | None = None,
            vmax: float | None = None,
            clip_intervals: tuple[int, int] | None = None,
            ignore_zeros: bool = True,
            xlabel_rotation: int = 90,
            ylabel_rotation: int = 0,
            xtick_fontsize: int = 8,
            ytick_fontsize: int = 8,
            heatmap_kwargs: dict | None = None,
            sort_genes_by_hvg: bool = False,
            zscore_axis: int | None = None,
            highly_variable_rank_key: str = "highly_variable_rank",
            **extract_kwargs: dict
        ) -> None:
    """
    Create a customizable heatmap for adata with group and condition support.

    Heatmaps are fucking useless, don't use them....use the median heatmap, this
    is too slow

    NOTE: This function is unfinished, please use the sc.pl.heatmap()!

    Args:
        adata (anndata.AnnData): Adata object.
        keys (str | list[str]): Gene/obs/obsm keys to plot.
        obsm_keys (list[str] | None, optional): Keys in .obsm to query from.
            Defaults to None.
        groupby (str | None, optional): Column in .obs to group cells along
            x-axis. Defaults to None.
        condition_obs_key (str | None, optional): Column to split conditions by.
            Defaults to None.
        conditions (list[str] | None, optional): List of conditions to keep (and
            order). Defaults to None.
        var_keys_only (bool, optional): Limit keys to .var_names only.
            Defaults to False.
        save_path (str | None, optional): Path to save the figure.
            Defaults to None.
        base_figsize (Sequence[float | int], optional): Base figure size (w, h).
            Defaults to (6, 4)
        cmap (str): Matplotlib colormap. Defaults to "tubro".
        figsize_scale (float): Scale multiplier for figure.
        vmin (float | None): Minimum color value.
        vmax (float | None): Maximum color value.
        clip_intervals (tuple[int, int] | None): Percentiles to clip data.
        ignore_zeros (bool): Ignore zeros in percentile clipping.
        xlabel_rotation (int): Rotation of x-axis labels.
        ylabel_rotation (int): Rotation of y-axis labels.
        xtick_fontsize (int): Font size for x-axis.
        ytick_fontsize (int): Font size for y-axis.
        heatmap_kwargs (dict | None): Additional arguments to sns.heatmap.
        sort_genes_by_hvg (bool): Sort genes by .var['highly_variable'] if
            available.
        zscore_axis (int | None): Apply z-score normalization (0=per gene, 1=per
            cell).
        **extract_kwargs: Additional keyword arguments forwarded to
            ``sc_utils.get_adata_sub_keys``.

    Returns:
        None

    Calls:
        get_adata_sub_keys, get_colors_wrapped, validate_groupby_column

    Tags:
        groupby, heatmap, obs, var, visualization
    """
    # ##########################
    # Setup
    start = time.perf_counter()

    if isinstance(keys, str):
        keys = [keys]
    if isinstance(obsm_keys, str):
        obsm_keys = [obsm_keys]
    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    logger.info(f"[{time.perf_counter() - start:.2f}s] Setup done")
    # ##########################
    # Optional HVG sorting
    start = time.perf_counter()

    if sort_genes_by_hvg and highly_variable_rank_key in adata.var.columns:
        hvg_bool = adata.var.loc[keys, highly_variable_rank_key].fillna(False).astype(bool)
        keys = pd.Series(keys)[hvg_bool.values].tolist() + pd.Series(keys)[~hvg_bool.values].tolist()

    logger.info(f"[{time.perf_counter() - start:.2f}s] HVG sorting done")
    # ##########################
    # Extract data (genes, obs, or obsm)
    start = time.perf_counter()

    df = get_adata_sub_keys(
        adata,
        keys=keys if not condition_obs_key else list(keys) + [condition_obs_key],
        obsm_keys=obsm_keys,
        groupby=groupby,
        var_keys_only=var_keys_only,
        **extract_kwargs,)
    # ##########################
    # IO checks
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    validate_groupby_column(
        adata.obs, condition_obs_key, check_categorical=True, groups=conditions)
    # ##########################
    logger.info(f"[{time.perf_counter() - start:.2f}s] Data extraction done")
    # ##########################
    # Condition + Group filtering
    start = time.perf_counter()
    # Handle joint group/condition filtering and labeling
    obs_subset = adata.obs.loc[df.index]

    if condition_obs_key and conditions:
        cond_mask = obs_subset[condition_obs_key].isin(conditions)
        df = df.loc[cond_mask]
        obs_subset = obs_subset.loc[cond_mask]

    sort_keys = []
    if groupby:
        sort_keys.append(obs_subset[groupby].astype(str))
    if condition_obs_key:
        sort_keys.append(pd.Categorical(obs_subset[condition_obs_key], categories=conditions if conditions else None))

    if sort_keys:
        sort_df = pd.DataFrame({f"__sort_{i}__": col for i, col in enumerate(sort_keys)}, index=df.index)
        df = df.loc[sort_df.sort_values(list(sort_df.columns)).index]

    logger.info(f"[{time.perf_counter() - start:.2f}s] Group/condition filtering done")
    # ##########################
    # Clipping values by percentile
    start = time.perf_counter()

    if clip_intervals:
        for key in keys:
            col = df[key].values
            if ignore_zeros:
                col = col[col != 0]
            if len(col) > 0:
                low, high = np.percentile(col, clip_intervals)
                df[key] = np.clip(df[key], low, high)

    logger.info(f"[{time.perf_counter() - start:.2f}s] Clipping done")
    # ##########################
    # Z-score normalization
    start = time.perf_counter()

    if zscore_axis == 0:
        df[keys] = (df[keys] - df[keys].mean(axis=0)) / df[keys].std(axis=0, ddof=0)
    elif zscore_axis == 1:
        df[keys] = ((df[keys].T - df[keys].T.mean(axis=0)) / df[keys].T.std(axis=0, ddof=0)).T

    logger.info(f"[{time.perf_counter() - start:.2f}s] Z-scoring done")
    # ##########################
    # Ensure vmin and vmax are in kwargs if defined
    if vmin is not None:
        heatmap_kwargs.setdefault("vmin", vmin)
    if vmax is not None:
        heatmap_kwargs.setdefault("vmax", vmax)
    # return matrix, cmap, row_cluster, col_cluster, df.index.astype(str), keys, heatmap_kwargs
    # ##########################
    # Plotting matrix
    start = time.perf_counter()

    row_cluster = not sort_genes_by_hvg
    col_cluster = not (groupby or condition_obs_key)

    matrix = df[keys].T  # shape: (n_genes, n_cells)

    matrix.columns = df.index.astype(str)  # needed for col_colors indexing
    # ##########################
    # Column colors for clustermap (groupby/condition annotations)
    col_colors = None
    if condition_obs_key or groupby:
        col_annot = pd.DataFrame(index=pd.Index(df.index, name="cell"))

        if condition_obs_key:
            col_annot["condition"] = adata.obs.loc[df.index, condition_obs_key].astype(str)
        if groupby:
            col_annot["group"] = adata.obs.loc[df.index, groupby].astype(str)

        # Create color maps using get_colors_wrapped
        color_map = {}
        for col in col_annot.columns:
            unique_labels = col_annot[col].unique()
            color_list = get_colors_wrapped(len(unique_labels))
            color_dict = dict(zip(unique_labels, color_list))
            color_map[col] = col_annot[col].map(color_dict).tolist()

        # Combine into a final DataFrame
        col_colors = pd.DataFrame(color_map, index=col_annot.index)
    # ##########################
    return {
        "data": matrix,
        "cmap": cmap,
        "row_cluster": row_cluster,
        "col_cluster": col_cluster,
        "col_colors": col_colors,
        "xticklabels": False,
        "yticklabels": keys,
        # "heatmap_kwargs": heatmap_kwargs,
        "figsize": (
            base_figsize[0] + figsize_scale * matrix.shape[1] / 10,
            base_figsize[1] + figsize_scale * matrix.shape[0] / 5)}

    g = clustermap(
        data=matrix,
        cmap=cmap,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=keys,
        figsize=(
            base_figsize[0] + figsize_scale * matrix.shape[1] / 10,
            base_figsize[1] + figsize_scale * matrix.shape[0] / 5),
        **heatmap_kwargs,)

    g.ax_heatmap.set_xlabel("Cells")
    g.ax_heatmap.set_ylabel("Features")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=ylabel_rotation, fontsize=ytick_fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # print(f"[{time.perf_counter() - start:.2f}s] Plotting done")


'''
long_doc.
    plot_violin possible plots
    | ``groupby`` | ``condition_obs_key`` | ``one_plot_per_key`` | X-axis          | Split (``hue``)           | Interpretation                                               |  # noqa: E501
    | --------- | --------------- | ------------------ | --------------- | ----------------------- | ------------------------------------------------------------ |  # noqa: E501
    | Yes       | No              | -                  | ``groupby``       | ``groupby``               | Classic violin per group                                     |  # noqa: E501
    | Yes       | Yes             | -                  | ``groupby``       | ``condition_obs_key`` (split) | Split violin per group by condition                          |  # noqa: E501
    | No        | Yes             | -                  | constant ``"_"``  | ``condition_obs_key`` (split) | Per key one violin split by condition                        |  # noqa: E501
    | No        | No              | True               | constant ``"_"``  | None                    | Per key one full violin                                      |  # noqa: E501
    | No        | No              | False              | ``"__feature__"`` | None                    | All keys combined in one plot (one violin per key on x-axis) |  # noqa: E501
'''


def plot_violin(
            adata: ad.AnnData,
            keys: str | list[str],
            obsm_keys: list[str] | None = None,
            groupby: str | None = None,
            var_keys_only: bool = False,
            save_path: str | None = None,
            use_stripplot: bool = True,
            jitter: bool = True,
            density_norm: str = "width",
            inner: str | None = "box",
            legend: bool = False,
            legend_loc: str = "best",
            xlabel_rotation: int = 90,
            base_figsize: Sequence[float | int] = (5, 5),
            ncols: int = 4,
            colorfull: bool = True,
            color_seed: int = 42,
            violinplot_kwargs: dict | None = None,
            stripplot_kwargs: dict | None = None,
            max_category_length: int = 20,
            clip_intervals: list | None = None,
            do_remove: bool = False,
            ignore_zeros: bool = True,
            split_title_text_kwargs: dict = {},
            condition_obs_key: str | None = None,
            conditions: list[str] | tuple[str, ...] | None = None,
            one_plot_per_key: bool = True,
            show: bool = True,
            palette: str | list[str] | dict | None = None,
            plot_thresholds: dict[str, tuple[float | None, float | None]] | None = None,
            **extract_kwargs: Any
        ) -> None | plt.Axes | list[plt.Axes]:
    """
    Create customized violin plots similar to Scanpy's sc.pl.violin,
    with optional condition-based split violins.

    This function generates violin plots for specified genes (using adata.X),
    obs, or obsm data within an adata object. It allows for extensive
    customization, including the option to split violins by a secondary
    condition (e.g. treatment/control), add strip plots with black-edged dots,
    automatically group data, and clip value ranges. The plots can be organized
    into multiple panels.

    NOTE: Automatically discards empty data plots.

    Args:
        adata (anndata.AnnData): Adata object.
        keys (str | list[str]): List of gene names (in adata.var), obs keys
            (in adata.obs), or obsm keys (specified in obsm_keys) to plot. Can
            be a single string or a list of strings.
        obsm_keys (list[str] | None, optional): List of keys in adata.obsm to
            search first. If provided, the function will first search these keys
            in adata.obsm columns, and if a key is not found, it will search in
            obs or genes. Defaults to None.
        groupby (str | None, optional): The column in adata.obs to group by for
            x-axis categories. If None, a single violin is drawn per key using
            all cells (no grouping on the x-axis). Defaults to None.
        condition_obs_key (str, optional): Column in adata.obs to use for
            splitting violins (e.g., treatment). If provided, each violin will
            be split left/right based on the condition values.
        conditions (list or tuple of str, optional): Specific values in
            ``condition_obs_key`` to include and to control left/right ordering in
            the split. If None, all values in the column are used.
        var_keys_only (bool, optional): If you want to plot adata.var keys.
            Defaults to False.
        save_path (str | None, optional): If provided, the path to save the
            image. Defaults to None.
        use_stripplot (bool, optional): If True, adds a strip plot on top of the
            violins. Defaults to True.
        jitter (bool, optional): If True, applies jitter to the strip plot for
            better visibility. Defaults to True.
        density_norm (str, optional): Scaling method for the violin plots
            ('width', 'area', 'count'). Defaults to "width".
        inner (str | None, optional): Display method for the violin plots
            ('box', 'quartile', 'point', None). Defaults to "box".
        legend (bool, optional): If True, shows a legend. Defaults to False.
        legend_loc (str, optional): Location of the legend ('best', 'upper
            right', 'center left', etc.). Defaults to "best".
        xlabel_rotation (int, optional): Degree of rotation for the x-axis
            labels. Defaults to 90.
        base_figsize (Sequence[float | int], optional): Base size for each plot in inches
            (width, height). Defaults to (5, 5).
        ncols (int, optional): Maximum number of plots per row. Defaults to 4.
        colorfull (bool, optional): If True, uses distinct colors for each
            violin. Defaults to True.
        color_seed (int, optional): Seed for reproducibility of colors. Defaults
            to 42.
        violinplot_kwargs (dict, optional): Additional keyword arguments to pass
            to sns.violinplot. Defaults to None. Core parameters like ``x``, ``y``,
            ``hue``, and ``split`` are controlled internally and will be overridden
            if present.
        stripplot_kwargs (dict, optional): Additional keyword arguments to pass
            to sns.stripplot. Defaults to None. Core parameters like ``x``, ``hue``,
            and ``dodge`` are managed internally and will be overridden if
            present.
        max_category_length (int, optional): Maximum length for category names
            before replacing them with acronyms. Defaults to 10.
            Quantile or percentile range used to clip expression values before plotting.
            Should be a two-element list, e.g. [0.01, 0.99] or [5, 95].
            When provided, data are passed to `clip_dataframe()` which trims or removes
            outliers based on this range.
            If None, no clipping is performed.
            Useful for limiting the effect of extreme values on violin shape.
        clip_intervals : list[float, float] | None = None
            Lower and upper percentiles for clipping or removal. Defines the range
            applied by `ignore_zeros` and `do_remove`.
        do_remove : bool = False
            Removes values outside limits defined by `clip_intervals`. If False,
            clips them to those limits. Works after `ignore_zeros` modifies the
            percentile range.
        ignore_zeros : bool = True
            Excludes zeros when calculating quantile limits from `clip_intervals`.
            Shifts the lower bound above zero. Combined with `do_remove=True`, this
            removes zeros as out-of-range values.
        show (bool, optional): If True executes plt.show(). Defaults to True.
        palette: Color mapping as string, list, or dict (passed to seaborn).
            Overwrites the other color arguments.
        plot_thresholds (dict[str, tuple[float | None, float | None]] | None, optional):
            Mapping of each key to (lower, upper) thresholds to draw as
            horizontal dashed lines. Use None for a value to skip plotting that
            threshold for that key. Defaults to None.
        **extract_kwargs: Additional keyword arguments forwarded to
            ``sc_utils.get_adata_sub_keys``.

    Returns:
        None | matplotlib.pyplot.Axes | list[matplotlib.pyplot.Axes]:
            The function either displays or returns the generated plot.

    Raises:
        ValueError: If any specified key is not found in .obs, .var, or .obsm.

    Calls:
        clip_dataframe, create_acronym, ensure_unique_acronyms,
        get_adata_sub_keys, get_rows_cols_figsize, split_title_text,
        validate_groupby_column

    Called By:
        plot_qc_violins

    Tags:
        groupby, obs, var, violin, visualization
    """
    # #########################################################
    # Check if groupby and condition_obs_key is properly setup
    if groupby is not None:
        if var_keys_only:
            validate_groupby_column(
                adata.var, groupby, check_categorical=True, print_name="adata.var")
            if condition_obs_key is not None:
                validate_groupby_column(
                    adata.var, condition_obs_key, check_categorical=True,
                    print_name="adata.var", groups=conditions)
        else:
            validate_groupby_column(
                adata.obs, groupby, check_categorical=True)
            if condition_obs_key is not None:
                validate_groupby_column(
                    adata.obs, condition_obs_key, check_categorical=True, groups=conditions)
    # #########################################################
    # Prepare the list of keys and determine the layout of the plots
    if isinstance(keys, str):
        keys = [keys]  # Convert single string to list

    # Check if the obsm_keys is a string, then convert it to list
    if isinstance(obsm_keys, str):
        obsm_keys = [obsm_keys]
    # Check if condition is set properly
    proper_one_key = groupby is None and condition_obs_key is None and not one_plot_per_key
    if condition_obs_key is not None or conditions is not None:
        if condition_obs_key is None or conditions is None:
            raise ValueError("Both 'condition_obs_key' and 'conditions' must be provided together.")
        if not isinstance(conditions, (list, tuple)) or len(conditions) != 2:
            raise ValueError("'conditions' must be a list or tuple of exactly two values.")
    # #########################################################
    plot_data = get_adata_sub_keys(
            adata, keys, obsm_keys=obsm_keys, groupby=groupby,
            var_keys_only=var_keys_only, **extract_kwargs,)

    # Drop columns that are entirely NaN
    if isinstance(plot_data, pd.DataFrame):
        plot_data = plot_data.loc[:, ~plot_data.isna().all()]
        keys = [k for k in keys if k in plot_data.columns]
    # Remove the grouby key to not show it in the plots
    if groupby is not None:
        if groupby in keys:
            keys.remove(groupby)
    # Drop empty columns except for the groupb
    if isinstance(plot_data, pd.DataFrame):
        plot_data = plot_data.loc[
            :, (plot_data.columns == groupby) | ~plot_data.isna().all()]
        keys = [k for k in keys if k in plot_data.columns and k != groupby]
        if not keys:
            raise ValueError(
                "All provided keys have no associated data in the adata object.")
    # #########################################################
    # Validate and normalize plot_thresholds (new)
    thresholds_by_key = plot_thresholds if isinstance(plot_thresholds, dict) else None
    if thresholds_by_key is not None:
        threshold_style = {
            "linestyle": "--", "linewidth": 1.2,
            "zorder": 4, "alpha": 0.8}
        threshold_colors = {"lower": "tab:blue", "upper": "tab:red"}
    # #########################################################
    # Optionally clip the data to intervals
    if (clip_intervals is not None) or ignore_zeros:
        clip_dataframe(plot_data,
                       clip_perc=clip_intervals,
                       inplace=True, ignore_zeros=ignore_zeros,
                       do_remove=do_remove)
    # #########################################################
    # Define or use the groupby
    if groupby is not None:
        # #############################
        # Preserve group order
        plot_data[groupby] = pd.Categorical(
            plot_data[groupby].astype(str),
            categories=plot_data[groupby].cat.categories.astype(str),
            ordered=plot_data[groupby].cat.ordered)

        groupby_values = plot_data[groupby].dropna().cat.categories

        # Create acronyms for long category names
        category_map = {}
        for category in groupby_values:
            if len(category) > max_category_length:
                acronym = create_acronym(category)
            else:
                acronym = category
            category_map[category] = acronym

        # Ensure the acronyms are unique
        category_map = ensure_unique_acronyms(category_map)

        # Add groupby_acronym to the plot_data DataFrame
        plot_data[groupby] = plot_data[groupby].cat.rename_categories(category_map)

        num_categories = len(category_map)
        # width_per_category = 0.6  # Adjust width per category
    else:
        num_categories = len(keys) if condition_obs_key is None and not one_plot_per_key else 1
        # width_per_category = 1.0
    # #########################################################
    # Add condition_obs_key-based splitting
    if condition_obs_key is not None:
        plot_data[condition_obs_key] = adata.obs[condition_obs_key].astype(str)
        conditions = [str(c) for c in conditions]
        plot_data = plot_data[plot_data[condition_obs_key].isin(conditions)]

        plot_data[condition_obs_key] = pd.Categorical(plot_data[condition_obs_key], categories=conditions)
        plot_data[condition_obs_key] = plot_data[condition_obs_key].cat.remove_unused_categories()

        condition_order = list(plot_data[condition_obs_key].cat.categories.tolist())
        if groupby is not None:
            plot_data[groupby] = plot_data[groupby].cat.remove_unused_categories()
            # Maybe there are new missing categories
    else:
        condition_order = None
    # #########################################################
    # Get the number of plots and split into rows and cols
    if proper_one_key:
        num_plots = 1
        nrows = 1
        ncols = 1
        figsize = base_figsize
    else:
        num_plots = len(keys)
        ncols, nrows, figsize = get_rows_cols_figsize(
            n_categories=num_plots,
            ncols=ncols,
            base_figsize=base_figsize,)
    # #########################################################
    # Prepare the matplotlib objects for plotting
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    axes = axes.flatten() if num_plots > 1 else [axes]  # Ensure axes is iterable
    # #########################################################
    # Generate shiny distinct colors if colorfull=True or use palette if provided
    if palette is not None:
        if isinstance(palette, list):
            if not all(is_color_like(c) for c in palette):
                raise ValueError("All elements in palette must be valid colors.")
            colors = [to_rgba(c) for c in palette]
        else:
            if not is_color_like(palette):
                raise ValueError("Palette must be a valid color or list of colors.")
            colors = [to_rgba(palette)]
        if groupby is not None and len(colors) < len(groupby_values):
            raise ValueError("Number of colors must match or exceed number of groupby categories.")
    else:
        if colorfull and groupby is not None:
            # print("using colors", groupby, num_categories, color_seed)
            colors = distinctipy.get_colors(num_categories, rng=color_seed)
            # print(colors)
        if colorfull and groupby is None:
            colors = distinctipy.get_colors(1, rng=color_seed)
        else:
            colors = None
    # #########################################################
    # Prepare defaults safely for optional arguments
    if violinplot_kwargs is None:
        violinplot_kwargs = {}
    if stripplot_kwargs is None:
        stripplot_kwargs = {}
    # Optional appearance settings (can be overridden by user)
    violinplot_kwargs.setdefault("cut", 0)
    stripplot_kwargs.setdefault("edgecolor", "black")
    stripplot_kwargs.setdefault("linewidth", 0.4)
    stripplot_kwargs.setdefault("size", 2.5)
    stripplot_kwargs.setdefault("alpha", .7)
    # #########################################################
    # Plot the violins
    if proper_one_key:
        ax = axes[0]
        df_tidy = plot_data[keys].melt(var_name="keys", value_name="values")
        violinplot(
            x="keys",
            y="values",
            data=df_tidy,
            ax=ax,
            hue=None,
            hue_order=None,
            split=False,
            density_norm=density_norm,
            inner=inner,
            legend=legend,
            **violinplot_kwargs)
        if use_stripplot:
            stripplot(
                x="keys",
                y="values",
                data=df_tidy,
                ax=ax,
                hue=None,
                dodge=False,
                jitter=jitter,
                **stripplot_kwargs)
        ax.set_title("All features")
        ax.set_ylabel("expression")
        ax.tick_params(axis="x", rotation=xlabel_rotation)
        # Draw per-key thresholds on a single-axes layout (new)
        if thresholds_by_key is not None:
            # To check if the threshold actually should be plotted
            # Clip the outliers
            # plot_data_clip = clip_dataframe(
            #     df_tidy, [.01, .99], False, do_remove=True)
            # this_data = plot_data_clip["values"].to_numpy()
            this_data = df_tidy["values"].to_numpy()
            # To check if the threshold actually should be plotted
            df_max = np.nanmax(this_data)
            df_min = np.nanmin(this_data)
            tol = (df_max - df_min) * 0.05
            df_max = df_max + tol
            df_min = df_min - tol

            tick_positions = ax.get_xticks()
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            label_to_x = {lab: tick_positions[i]
                          for i, lab in enumerate(tick_labels)}
            xpad = 0.4  # half-width around each category center
            for k, (low, up) in zip(keys, thresholds_by_key):
                x = label_to_x.get(str(k))
                if x is None:
                    continue
                if low is not None:
                    if (low >= df_min and low <= df_max):
                        ax.hlines(
                            float(low), x - xpad, x + xpad,
                            colors=threshold_colors["lower"],
                            label="_threshold_lower", **threshold_style)
                if up is not None:
                    if (up >= df_min and up <= df_max):
                        ax.hlines(
                            float(up), x - xpad, x + xpad,
                            colors=threshold_colors["upper"],
                            label="_threshold_upper", **threshold_style)
    else:
        for i, key in enumerate(keys):
            ax = axes[i]
            # ##############################
            # Define x and hue logic cleanly
            if groupby is not None:
                violin_x = groupby
            else:
                violin_x = np.full(len(plot_data), "_")

            if groupby is not None and condition_obs_key is not None:
                violin_hue = condition_obs_key
                violin_split = True
                violin_hue_order = condition_order
            elif groupby is not None:
                violin_hue = groupby
                violin_split = False
                violin_hue_order = None
            elif condition_obs_key is not None:
                violin_hue = condition_obs_key
                violin_split = True
                violin_hue_order = condition_order
            else:
                violin_hue = None
                violin_split = False
                violin_hue_order = None

            if (violin_hue is not None):
                stripplot_kwargs.setdefault("palette", colors)
                violinplot_kwargs.setdefault("palette", colors)
            else:
                if colors is not None:
                    stripplot_kwargs.setdefault("color", colors[0])
                    violinplot_kwargs.setdefault("color", colors[0])
            # ##############################
            # Violin plot
            violinplot(
                x=violin_x,
                y=key,
                data=plot_data,
                ax=ax,
                hue=violin_hue,
                hue_order=violin_hue_order,
                split=violin_split,
                density_norm=density_norm,
                inner=inner,
                legend=legend,
                **violinplot_kwargs)
            # ##############################
            # Strip plot
            if use_stripplot:
                stripplot(
                    x=violin_x,
                    y=key,
                    data=plot_data,
                    ax=ax,
                    hue=violin_hue,
                    hue_order=violin_hue_order,
                    dodge=violin_split,
                    jitter=jitter,
                    **stripplot_kwargs)
            # ##############################
            ax.set_title(split_title_text(key, **split_title_text_kwargs))
            ax.set_ylabel(key)
            if groupby is not None:
                ax.tick_params(axis="x", rotation=xlabel_rotation)
            # Draw thresholds per sub-plot (new)
            if thresholds_by_key is not None:
                # To check if the threshold actually should be plotted
                # Clip the outliers
                # plot_data_clip = clip_dataframe(
                #     plot_data, [.01, .99], False, do_remove=True)
                # this_data = plot_data_clip[key].to_numpy()
                this_data = plot_data[key].to_numpy()
                # To check if the threshold actually should be plotted
                df_max = np.nanmax(this_data)
                df_min = np.nanmin(this_data)
                tol = (df_max - df_min) * 0.05
                df_max = df_max + tol
                df_min = df_min - tol

                low, up = thresholds_by_key.get(key, (None, None))
                if low is not None:
                    if (low >= df_min and low <= df_max):
                        ax.axhline(
                            float(low),
                            color=threshold_colors["lower"],
                            label="_threshold_lower", **threshold_style)
                if up is not None:
                    if (up >= df_min and up <= df_max):
                        ax.axhline(
                            float(up),
                            color=threshold_colors["upper"],
                            label="_threshold_upper", **threshold_style)
        # #########################################################
        # Remove empty subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    # #########################################################
    # Add legend if required
    if legend and legend_loc and groupby is not None:
        handles, labels = ax.get_legend_handles_labels()
        # Add the axis-specific legend to the individual plots
        ax.legend(handles, labels, loc=legend_loc if legend_loc != 'best' else 'upper right')
    # #########################################################
    # Add a "paper-style" figure legend under the whole figure
    if groupby is not None:
        if len(category_map) > 0 and any(len(cat) > max_category_length for cat in category_map.keys()):
            acronym_legend = [f"{acr}: {full}" for full, acr in category_map.items() if acr != full]
            # Adjust position for a paper-style legend under the figure
            plt.figtext(
                0.5, -(0.04 * (1 / nrows)) * len(acronym_legend),
                '\n'.join(acronym_legend), horizontalalignment='center', fontsize=10)
    # #########################################################
    # Final adjustments and display
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        return axes


def plot_umap(
            adata: ad.AnnData,
            keys: list[str],
            figsize: Sequence[float | int] = (12, 6),
            ncols: int = 2,
            padding: float = 0.4,
            show: bool = True,
            save_path: str | None = None,
            obsm_keys: list[str] | None = None,
            sub_ids: list[str] | None = None,
            ax: mpl.axes.Axes | None = None,
            split_title_text_kwargs: dict = {},
            extract_kwargs: dict = {},
            **kwargs: Any
        ) -> None | plt.Axes:
    """
    Plots multiple UMAP visualizations with efficient handling of large datasets
    and customizable layout.

    This function renders UMAP plots for features in ``keys``, optionally using
    ``.obsm``-extracted values or subsetting by cell IDs. It supports custom
    layout, figure sizing, optional saving, and conditional coloring via values
    in ``adata.uns``. Designed to minimize memory overhead and enhance flexibility
    for bulk rendering.

    NOTE:
        If ``sub_ids`` is provided, dot size (``s``) may need to be set explicitly
        in ``**kwargs`` for consistent visual appearance across plots.

    Args:
        adata (anndata.AnnData): Adata object containing UMAP coordinates
        keys (list[str]): Keys to plot (genes, scores, or any other expression
            values).
        figsize (Sequence[float | int], optional): Base figure size per subplot. Defaults to
            (12, 6).
        ncols (int, optional): Number of columns in the subplot grid. Defaults
            to 2.
        padding (float, optional): Padding between subplots. Defaults to 0.4.
        show (bool, optional): If True executes plt.show(). Defaults to True.
        save_path (str | None, optional): If provided, saves the figure to this
            path. Defaults to None.
        obsm_keys (list[str] | None, optional): List of ``.obsm`` keys to extract
            data from. Defaults to None.
        sub_ids (list[str] | None, optional): Subset of obs_names to plot.
            Defaults to None.
        ax (matplotlib.axes.Axes | None, optional): Predefined Axes object to
            plot into. Defaults to None.
        split_title_text_kwargs (dict, optional): Keyword arguments for
            splitting long title text. Defaults to {}.
        extract_kwargs (dict, optional): Additional arguments forwarded to
            ``get_adata_sub_keys``. Defaults to {}.
        **kwargs (Any): Additional keyword arguments passed to the sc.pl.umap().

    Returns:
        None | matplotlib.pyplot.Axes:
            None or plt.Axes: The function either displays or returns the
                generated plot.

    Calls:
        get_adata_sub_keys, get_rows_cols_figsize, split_title_text

    Called By:
        continuos_umap_helper, discrete_umap_helper, plot_umap_cat_splitting,
        plot_umap_sbs, run_downstream

    Tags:
        embedding, groupby, obs, var, visualization
    """
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(obsm_keys, str):
        obsm_keys = [obsm_keys]

    renamed_keys = {}  # To track original keys and their temporary names
    extracted_keys = []

    # Handle keys extraction from ``obsm`` if ``obsm_keys`` are provided
    if obsm_keys is not None:
        # TODO: Use get_minimal_adata to actually get a minimal copy and plot this
        #       To be 100%, we don't mess with the object (currently only 99.9%)

        # Extract data from ``obsm`` and handle conflicts with ``obs``
        temp_data = get_adata_sub_keys(
            adata, keys, obsm_keys=obsm_keys, **extract_kwargs,)

        # Build a DataFrame to join all at once
        temp_obs_df = pd.DataFrame(temp_data, index=adata.obs.index)

        # Rename existing conflicting columns to temporary backups
        for key in temp_data.columns:
            if key in adata.obs.columns:  # Check if key already exists in ``obs``
                new_key = f"{key}_temp_blabla"
                adata.obs.rename(columns={key: new_key}, inplace=True)
                renamed_keys[key] = new_key
            extracted_keys.append(key)

        # In-place, fragmentation-safe join
        adata.obs = adata.obs.join(temp_obs_df)

    # Validate that all keys in ``keys`` are present in ``adata.obs`` or ``adata.var_names``
    usable_subset_obs = np.intersect1d(keys, adata.obs.columns.tolist()).tolist()
    usable_subset_var_names = np.intersect1d(keys, adata.var_names.tolist()).tolist()
    usable_subset = usable_subset_obs + usable_subset_var_names
    missing_keys = np.setdiff1d(keys, usable_subset)
    if missing_keys.size > 0:
        raise ValueError(f"The keys: {', '.join(missing_keys)} are not in ``obs`` or ``var_names``.")

    # Calculate layout parameters and initialize the plotting area
    nplots = len(keys)
    ncols, nrows, full_figsize = get_rows_cols_figsize(
        n_categories=nplots,
        ncols=ncols,
        base_figsize=figsize,
        padding=(padding, padding),)

    if ax is None:
        # Create figure and axes for the subplots
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=full_figsize, constrained_layout=True)
        axes = np.array(axes).reshape(-1)  # Flatten axes array for easier iteration
    else:
        if not isinstance(ax, list):
            axes = [ax]

    # Set xlim and ylim based on min and max of ``adata.obsm["X_umap"]``
    if "X_umap" in adata.obsm:
        x_min, x_max = np.nanmin(adata.obsm["X_umap"][:, 0]), np.nanmax(adata.obsm["X_umap"][:, 0])
        y_min, y_max = np.nanmin(adata.obsm["X_umap"][:, 1]), np.nanmax(adata.obsm["X_umap"][:, 1])
        x_dist = np.ptp([x_min, x_max])
        y_dist = np.ptp([y_min, y_max])
        x_min = x_min - x_dist * .05
        y_min = y_min - y_dist * .05
        x_max = x_max + x_dist * .05
        y_max = y_max + y_dist * .05
    else:
        x_min, x_max, y_min, y_max = None, None, None, None

    # Subset adata temporarily, only for the current UMAP plot
    temp_adata = adata if sub_ids is None else adata[sub_ids]

    # Loop over color keys and plot UMAP
    for i, c in enumerate(keys):
        this_kwargs = copy(kwargs)
        # ##############################################
        # The var names need no handling
        if c in usable_subset_var_names:
            sc.pl.umap(temp_adata, color=c, ax=axes[i], show=False, **this_kwargs)
            continue
        # ##############################################
        # The obs and obsm dependant on the dtype need special handling
        if temp_adata.obs[c].dtype.name == "category":
            # ##############################################
            # Check if custom color is provided in kwargs; if not, fetch from ``uns`` or set with distinctipy
            palette_from_uns = False
            get_palette = False
            color_key = f"{c}_colors"
            if 'palette' not in this_kwargs:
                if color_key in temp_adata.uns.keys():
                    n_cats, n_colors = len(temp_adata.obs[c].cat.categories), len(temp_adata.uns[color_key])
                    if n_cats <= n_colors:
                        palette = temp_adata.uns[color_key]
                        if n_cats < n_colors:
                            # If using a subset and the key is categorical, subset the palette
                            unique_categories = temp_adata.obs[c].cat.remove_unused_categories().cat.categories
                            categories = temp_adata.obs[c].cat.categories
                            # Filter the palette to match the unique categories present in the subset
                            palette = [palette[categories.get_loc(cat)] for cat in unique_categories]
                            temp_adata.uns[color_key] = palette
                        # this_kwargs['palette'] = palette
                        palette_from_uns = True
                    else:
                        get_palette = True
            else:
                get_palette = True

            if get_palette:
                num_categories = len(
                    temp_adata.obs[c].cat.remove_unused_categories().cat.categories)
                this_kwargs['palette'] = distinctipy.get_colors(
                    num_categories, rng=temp_adata.uns["config"]["general"]["seed"])
                # Add the palette to the object
                if not palette_from_uns:
                    # temp_adata.uns["color_key"] = this_kwargs['palette']
                    temp_adata.uns[color_key] = this_kwargs['palette']
            # ##############################################
            # Suppress warnings only when subsetting and using ``uns`` palette
            if palette_from_uns and sub_ids is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    sc.pl.umap(temp_adata, color=c, ax=axes[i], show=False, **this_kwargs)
            else:
                sc.pl.umap(temp_adata, color=c, ax=axes[i], show=False, **this_kwargs)
            # ##############################################
        else:
            sc.pl.umap(temp_adata, color=c, ax=axes[i], show=False, **this_kwargs)

        axes[i].set_title(split_title_text(c, **split_title_text_kwargs), fontsize=14)
        for label in axes[i].get_xticklabels():
            label.set_rotation(90)
            label.set_ha('right')

        for label in axes[i].get_yticklabels():
            label.set_rotation(90)
        # Apply xlim and ylim if X_umap is available
        if x_min is not None and y_min is not None:
            axes[i].set_xlim([x_min, x_max])
            axes[i].set_ylim([y_min, y_max])
    # Hide unused subplots if any
    for ax in axes[nplots:]:
        ax.axis('off')
    # Save the figure if ``save_path`` is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # Cleanup: Remove any temporarily added keys from ``adata.obs``
    if extracted_keys:
        adata.obs.drop(columns=extracted_keys, inplace=True)
    # Rename any temporarily renamed keys back to their original names
    for original_key, temp_key in renamed_keys.items():
        adata.obs[original_key] = adata.obs.pop(temp_key)
    # Display the plot if ``show`` is True
    if show:
        plt.show()
    else:
        return axes


def plot_embedding_density(
            adata: ad.AnnData,
            groupby: str,
            do_plot: bool = True,
            force_recalculate: bool = False,
            basis: str = "X_umap",
            save_densities_key: str | None = None,
            grid_size: int = 1000,
            bw: int | str = 0.1,
            save_path: str | None = None,
            ncols: int | None = None,
            base_figsize: Sequence[float | int] = (6, 6),
            dpi: int = 100,
            add_background: bool = True,
            groups: np.ndarray | None = None,
            backgroud_density_treshold: float = 0.1,
            split_title_text_kwargs: dict = {},
            **contourf_kwargs: Any
        ) -> None:
    """Compute and optionally plot group-wise embedding density using 2D KDE.

    Estimates density on a grid over an embedding (e.g., UMAP) using kernel
    density estimation grouped by ``groupby``. Densities can be cached in ``.uns``,
    plotted, or saved as images.

    NOTE:
        - If ``basis`` is spatial, the y-axis is flipped to match image-like
          orientation.
        - The function was called ``embedding_density_fix`` before

    Args:
        adata (anndata.AnnData): Adata object with embedding coordinates in
            ``.obsm``.
        groupby (str): Column in ``.obs`` to groupby.
        do_plot (bool, optional): Whether to generate a contour plot.
            Defaults to True.
        force_recalculate (bool, optional): Recompute density even if cached.
            Defaults to False.
        basis (str, optional): Key in ``.obsm`` for the embedding.
            Defaults to "X_umap".
        save_densities_key (str | None, optional): Key to store computed
            densities in ``.uns``. Defaults to None.
        grid_size (int, optional): Number of grid bins per axis for KDE.
            Defaults to 1000.
        bw (int | str, optional): Bandwidth or method for KDE. Defaults to 0.1.
        save_path (str | None, optional): Path to save the generated plot.
            Defaults to None.
        ncols (int | None, optional): Number of columns in the output plot
            layout. Defaults to None.
        base_figsize (Sequence[float | int], optional): Base figure size in
            inches. Defaults to (6, 6).
        dpi (int): Plot resolution in dots per inch. Defaults to 100.
        add_background (bool): Add density from all groups as background.
            Defaults to True.
        groups (np.ndarray | None): Subset of group values to plot. Defaults to
            None.
        backgroud_density_treshold (float): Background density threshold to
            visualize. Defaults to 0.1.
        split_title_text_kwargs (dict): Keyword arguments for title formatting.
            Defaults to {}.
        **contourf_kwargs (Any): Additional keyword arguments passed to
            ``ax.contourf()``.

    Returns:
        None:
            This function modifies ``adata.uns`` and optionally shows or saves
            plots.

    Raises:
        ValueError: If ``basis`` is not found in ``adata.obsm``.

    Calls:
        get_bandwidth, get_rows_cols_figsize, split_title_text,
        validate_groupby_column

    Called By:
        plot_umap_cat_splitting, run_downstream

    Tags:
        density, groupby, obs, trajectory, visualization
    """
    # #########################################################
    # IO checks
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True, groups=groups)

    if basis not in adata.obsm.keys():
        raise ValueError(
            f"Please make sure {basis} is in the adata.obsm.keys()!")
    # ############################################################
    # Get the mesh grid
    x_min, x_max = np.min(adata.obsm[basis][:, 0]), np.max(adata.obsm[basis][:, 0])
    y_min, y_max = np.min(adata.obsm[basis][:, 1]), np.max(adata.obsm[basis][:, 1])
    x_dist = np.ptp([x_min, x_max])
    y_dist = np.ptp([y_min, y_max])
    x_min = x_min - x_dist * .05
    y_min = y_min - y_dist * .05
    x_max = x_max + x_dist * .05
    y_max = y_max + y_dist * .05

    xx, yy = np.mgrid[
            x_min:x_max:grid_size * 1j,
            y_min:y_max:grid_size * 1j]
    # ############################################################
    # Get the categories
    if groups is None:
        if adata.obs[groupby].dtype.name == 'category':
            categories = adata.obs[groupby].cat.categories
        else:
            categories = adata.obs[groupby].unique()
    else:
        categories = groups
    # ############################################################
    if save_densities_key not in adata.uns.keys() or force_recalculate:
        densities = {}
        for key_ in categories:
            # ############################################################
            # Get subsetted data
            data = adata[adata.obs[groupby] == key_].obsm[basis]
            bw_ = get_bandwidth(data, bw, categories)
            # Perform 2D KDE on the spatial data
            kde = FFTKDE(kernel='box', bw=bw_)  # Adjust bandwidth as needed
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            # grid, density = kde.fit(data).evaluate(grid_points=(grid_size, grid_size))
            density = kde.fit(data).evaluate(xy)

            # Reshape grid and density to use with RegularGridInterpolator
            density = density.reshape(grid_size, grid_size)  # Reshape density to a 2D array
            # Scale density for better common visualization
            density = (density - density.min()) / (density.max() - density.min())
            densities[key_] = density
        # ############################################################
        # Save densitiies
        if save_densities_key is not None:
            adata.uns[save_densities_key] = densities
    else:
        densities = adata.uns[save_densities_key]
        if grid_size != densities[list(densities.keys())[0]].shape[0]:
            raise AttributeError("The provided grid size must match the previous calculated one.")

    if do_plot:
        # ###########################################################################################################
        # Add defaults to the contourf_kwargs
        if "levels" not in contourf_kwargs.keys():
            contourf_kwargs["levels"] = 10
        if "cmap" not in contourf_kwargs.keys():
            contourf_kwargs["cmap"] = "turbo"
        # ###########################################################################################################
        n_groups = len(categories)
        # ###########################################################################################################
        # Set number of columns and calculate number of rows and full figure size
        ncols, nrows, figsize = get_rows_cols_figsize(
            n_categories=n_groups,
            ncols=ncols,
            base_figsize=base_figsize,)
        # Create figure and axes for the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = np.array(axes).reshape(-1)  # Flatten axes array for easier iteration
        # ############################################################
        # Calculate background density or just use zeros
        if add_background:
            data = adata.obsm[basis]
            bw_ = get_bandwidth(data, bw, categories)
            # Perform 2D KDE on the spatial data
            kde = FFTKDE(kernel='box', bw=bw_)  # Adjust bandwidth as needed
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            # grid, density = kde.fit(data).evaluate(grid_points=(grid_size, grid_size))
            density_all = kde.fit(data).evaluate(xy)

            # Reshape grid and density to use with RegularGridInterpolator
            density_all = density_all.reshape(grid_size, grid_size)
            # return density_all, xx, yy
            density_all = (density_all - density_all.min()) / (density_all.max() - density_all.min())
            density_all[density_all < backgroud_density_treshold] = 0
            density_all[density_all > backgroud_density_treshold] = .1
            # return density_all, densities, xx, yy, categories
        else:
            density_all = np.zeros_like(next(iter(densities.values())))

        for idx, key_ in enumerate(categories):
            # Scatter plot of spatial data colored by the interpolated density
            with rc_context({'figure.figsize': figsize, 'figure.dpi': dpi}):
                ax = axes[idx]
                cs2 = ax.contourf(xx, yy, densities[key_] + density_all, **contourf_kwargs)
                # _ = ax.contourf(xx, yy, density_all, levels=1, cmap="turbo", alpha=.3)

                _ = fig.colorbar(cs2, ax=ax, label="Density", shrink=0.8)
                ax.set_xlabel(f"{basis} X", fontsize=12)
                ax.set_ylabel(f"{basis} Y", fontsize=12)
                ax.set_title(
                    split_title_text(
                        f"Density Plot in Original Space for Group {key_}",
                        **split_title_text_kwargs),
                    fontsize=12)
                if basis == "spatial":
                    ax.invert_yaxis()

        # Hide any unused subplots
        for ax in axes[n_groups:]:
            ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)

        # Display the plot
        plt.show()


def plot_spatial(
            adata: ad.AnnData,
            obs_keys: list[str],
            plot_whole_tissue: bool = True,
            subset_ids: list[int] | None = None,
            boundary_scaler: float = 0.05,
            ignore_dimension_key: bool = True,
            **kwargs
        ) -> None:
    """Plot spatial view of adata with optional cropping and subsetting.

    NOTE:
        Also filters the future warning.

    Args:
        adata (anndata.AnnData): Spatial adata object with spatial information.
        obs_keys (list[str]): Column keys from ``adata.obs`` used for plotting.
        plot_whole_tissue (bool, optional): If True, expands plot to full
            tissue. Defaults to True.
        subset_ids (list[int] | None, optional): Optional list of indices to
            subset the plot. Defaults to None.
        boundary_scaler (float, optional): Margin factor around spatial
            boundary. Defaults to 0.05.
        ignore_dimension_key (bool, optional): If True, ignore existing
            'dimensions' key. Defaults to True.
        **kwargs: Additional keyword arguments passed to ``sc.pl.spatial``.

    Returns:
        None:
            The function generates a spatial plot and returns nothing.

    Called By:
        Spatial_prox.catplot, Spatial_prox.catplot_all,
        Spatial_prox.catplot_one_by_one, Spatial_prox.pairplot
    """
    # To be able to use the method, even if there is no "dimension" key in the adata.uns["spatial"]
    # Provide subsetting
    if subset_ids is not None:
        adata_view = adata[subset_ids]
    else:
        adata_view = adata

    uns_spatial = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
    if plot_whole_tissue:
        # TODO: Maybe optionally test the dimensions removal again
        # del adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["dimensions"]
        if "dimensions" in uns_spatial.keys() and not ignore_dimension_key:
            min_dim = uns_spatial["dimensions"]["image_space"]["min_dim"]
            max_dim = uns_spatial["dimensions"]["image_space"]["max_dim"]
        else:
            min_dim = adata.obsm["spatial"].min(0)
            max_dim = adata.obsm["spatial"].max(0)
        # Calc the offsets for the boundary
        diff_0 = np.ptp((max_dim[0], min_dim[0]))
        diff_1 = np.ptp((max_dim[1], min_dim[1]))
        # Define the crop cordinates
        crop_coord = [
                min_dim[0] - diff_0 * boundary_scaler,
                max_dim[0] + diff_0 * boundary_scaler,
                min_dim[1] - diff_1 * boundary_scaler,
                max_dim[1] + diff_1 * boundary_scaler]
    else:
        crop_coord = None

    if "img_key" not in kwargs.keys():
        kwargs["img_key"] = "lowres"

    # Ignore the future warning, we will never use the pointless muon.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return sc.pl.spatial(adata_view, color=obs_keys, crop_coord=crop_coord, **kwargs)


# ###################################################################################################
# Advanced Plotting
def show_category_over_umap(
            adata: ad.AnnData,
            groupby: str,
            umap_key: str = "X_umap"
        ) -> None:
    """
    Visualizes category-wise UMAP coordinate distributions using plot_ridgeline
    plots and histograms.

    For each unique category in ``adata.obs[key]``, this function creates
    ridgeline plots and histograms for both UMAP dimensions (x and y). It helps
    understand how categorical groups are distributed along the UMAP axes and
    can highlight shifts across these categories.

    NOTE:
        - This function is intended to aid exploratory analysis by visualizing
          distributional shifts across categories on UMAP coordinates.
        - You probably want to use the embedding_density()

    Args:
        adata (anndata.AnnData): Adata object containing UMAP coordinates.
        groupby (str): Column name in ``adata.obs`` used to group cells for
            plotting.
        umap_key (str, optional): Key in ``.obsm`` that contains the UMAP
            coordinates. Defaults to "X_umap".

    Returns:
        None:
            This function produces plots but does not return any value.

    Calls:
        plot_ridgeline, validate_groupby_column

    TODO:
        Finish writing. This function might require refinement or additional
        features, such as better labeling or saving plots.

    Tags:
        embedding, groupby, obs, visualization
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
            adata.obs, groupby, check_categorical=True)
    # #########################################################
    # Compute and visualize ridgeline and histogram for UMAP x-axis (dimension 0)
    # Extract UMAP x-coordinates per category
    data2 = [
        np.array(adata[adata.obs[groupby] == c].obsm[umap_key][:, 0])
        for c in adata.obs[groupby].unique()]
    # ##########################################
    # Plot ridgeline of UMAP x-coordinates
    plot_ridgeline(data2, overlap=0, fill='y')
    # Plot histogram of UMAP x-coordinates
    plt.hist(data2, bins=20)  # noqa: E703
    plt.show()
    # #########################################################
    # Compute and visualize ridgeline and histogram for UMAP y-axis (dimension 1)
    # Extract UMAP y-coordinates per category
    data2 = [
        np.array(adata[adata.obs[groupby] == c].obsm[umap_key][:, 1])
        for c in adata.obs[groupby].unique()]
    # ##########################################
    # Plot ridgeline of UMAP y-coordinates
    plot_ridgeline(data2, overlap=0.0, fill='y')
    # Plot histogram of UMAP y-coordinates
    plt.hist(data2, bins=20)  # noqa: E703
    plt.show()


def plot_hist_of_each_layer(
            adata: ad.AnnData,
            bins: int = 200,
            yscale: str = "log"
        ) -> None:
    """Plots histograms for each data layer in an adata object.

    This function visualizes the distribution of values in each layer of the
    provided adata object. For each layer, the values are flattened and plotted
    as a histogram with the specified number of bins and y-axis scaling.

    NOTE:
        This visualization helps identify global distributional properties and
        sparsity patterns across data layers. It is intended for diagnostic and
        exploratory analysis.

    Args:
        adata (anndata.AnnData): Adata object containing multiple layers.
        bins (int, optional): Number of bins for the histogram. Defaults to 200.
        yscale (str, optional): Scale for the y-axis ('linear', 'log', etc.).
            Defaults to "log".

    Returns:
        None:
            This function generates plots but does not return a value.

    Raises:
        ValueError: If the provided adata object does not contain any layers.

    TODO:
        Add options for saving plots to disk or handling large datasets
        efficiently.

    Tags:
        histogram, io, layer, visualization
    """
    # #########################################################
    # Check if the adata object has layers
    if not hasattr(adata, "layers") or not adata.layers:
        raise ValueError("The provided adata object does not have any layers.")
    # #########################################################
    # Iterate through all layers in the adata object and plot histograms
    for layer_name, layer_data in adata.layers.items():
        # ##########################################
        # Log the layer name and mean of the sum across axis 1
        logger.debug(f"Layer: {layer_name}, Mean: {layer_data.sum(1).mean()}")
        # ##########################################
        # Plot histogram for dense or sparse matrix
        if isinstance(layer_data, np.ndarray):
            # Use direct numpy flattening for dense arrays
            plt.hist(layer_data.flatten(), bins=bins)  # noqa: E703
        else:
            # Convert sparse matrix to array before flattening
            plt.hist(layer_data.toarray().flatten(), bins=bins)  # noqa: E703
        # ##########################################
        # Configure plot settings and display the histogram
        plt.yscale(yscale)
        plt.title(f"Histogram of layer {layer_name}")
        plt.show()


def plot_per_group_DEG_dotplot_n_gene_dendrogram(
            adata: ad.AnnData,
            config: dict | None = None,
            groupby: str | None = None,
            groups: list[str] | None = None,
            highly_variable: bool = False,
            top_n_genes: int = 5,
            img_suffix: str = "",
            var_dendrogram: bool = False,
            split_on: int = 30,
            part: str = "/downstream/",
            overlap_fill_to_top_n: bool = False,
            deg_key: str = "rank_genes_groups",
            perc: float = 0.15,
            p_val_cutoff: float = 0.01,
            lfc: float = 0.5,
            direction: str = "force_up_n_down",
            with_figure_explaination: bool = True
        ) -> tuple[list[str], dict[str, list[str]]]:
    """
    Creates a per-cluster DEG dot plot and optionally a gene dendrogram visualization.

    This function generates dot plots for differentially expressed genes (DEGs)
    across groups and, optionally, a gene dendrogram for visualizing
    co-expression and relationships. It supports hierarchical cluster ordering,
    customizable split ranges, DEG filtering criteria, and multiple export
    options.

    NOTE:
        This function depends on precomputed DEGs in ``adata.uns[deg_key]``
        and assumes the cluster hierarchy is stored or computable from
        ``adata.uns["config"]["general"]``.

    Args:
        adata (anndata.AnnData): Adata object.
        config (dict | None, optional): Configuration dictionary with plotting
            and analysis settings. uns["config"]`` if None. Defaults to None.
        groups (list[str] | None, optional): List of groups to consider.
            Defaults to None.
        groupby (str | None, optional): Optional key to group data by, must be a
            column in adata.obs None will use the
            config["general"]["cluster_algorithm"]. Defaults to None.
        highly_variable (bool, optional): Whether to subset only to highly
            variable genes. Not implemented. Defaults to False.
        top_n_genes (int, optional): Number of top DEGs to include per cluster.
            Defaults to 5.
        img_suffix (str, optional): Optional suffix for image filenames.
            Defaults to "".
        var_dendrogram (bool, optional): Whether to generate a gene dendrogram
            plot. Defaults to False.
        split_on (int, optional): Target number of groups per plot split.
            Defaults to 30.
        part (str, optional): Path under which output files are saved.
            Defaults to "/downstream/".
        overlap_fill_to_top_n (bool, optional): Whether to ensure top_n coverage
            despite overlap. Defaults to False.
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        perc (float, optional): Percentile cutoff used during DEG selection.
            Defaults to 0.15.
        p_val_cutoff (float, optional): P-value cutoff for DEG filtering.
            Defaults to 0.01.
        lfc (float, optional): Log fold change threshold for DEG filtering.
            Defaults to 0.5.
        direction (str, optional): DEG direction strategy.
            Defaults to "force_up_n_down".
        with_figure_explaination (bool, optional): Whether to log figure save
            information. Defaults to True.

    Returns:
        tuple[list[str], dict[str, list[str]]]:

            - all_variables_per_cluster (list[str]): All selected genes across
              groups.
            - variables_per_cluster (dict[str, list[str]]): Mapping of cluster
                to its selected genes.

    Raises:
        NotImplementedError: If ``highly_variable`` is True.

    Calls:
        create_gene_dendrogram, get_DEGs_per_group_for_plotting, get_colormap,
        get_save_path, validate_groupby_column

    Called By:
        run_downstream

    TODO:
        Implement support for highly variable genes. Extend visualization
        flexibility for interactive plotting and alternative backends.
    """
    # ########################################################################################
    # Initialize configuration if not provided
    if config is None:
        config = adata.uns["config"]
    # ########################################################################################
    # get the clustering algorithm key
    if groupby is None:
        groupby = config["general"]["cluster_algorithm"]
    # get the groups
    if groups is None:
        # groups = list(adata.uns[key]["names"].dtype.names)
        groups = adata.obs[groupby].unique().tolist()
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=False, groups=groups)
    # #########################################################
    # Calculate splits for large numbers of groups
    n_groups = len(groups)
    splits = np.linspace(
        0, n_groups, int(n_groups * top_n_genes / split_on + 0.5) + 1
    ).astype(int)
    # ########################################################################################
    # Handle highly variable genes option (currently not implemented)
    if highly_variable:
        raise NotImplementedError("This is not implemented yet, look at the code and finish it!")
    # ########################################################################################
    # Retrieve DEGs per cluster for plotting
    variables_per_cluster, all_variables_per_cluster = get_DEGs_per_group_for_plotting(
        adata=adata, top_n_genes=top_n_genes, groups=groups,
        overlap_fill_to_top_n=overlap_fill_to_top_n, deg_key=deg_key,
        perc=perc, p_val_cutoff=p_val_cutoff, lfc=lfc, direction=direction)
    # ########################################################################################
    # Generate dot plots for each split of groups
    for i in range(len(splits) - 1):
        # ##########################################
        # Select the subset of groups for this split
        cats = adata.obs[config["general"]["cluster_algorithm"]].cat.categories.tolist()[
            splits[i]:splits[i+1]]
        save_path = get_save_path(
            f"cluster_DEG_dotplot{img_suffix}_top{top_n_genes}_{i}.pdf", config, part=part)
        if with_figure_explaination:
            logger.info(f'The figure is saved as {save_path}')
        sub_dict = {k: v for k, v in variables_per_cluster.items() if k in cats}
        # ##########################################
        # Create dot plot with optional dendrogram for groups
        sc.tl.dendrogram(
            adata, config["general"]["cluster_algorithm"], **config["tl"]["dendrogram"])
        # Get the args form the config (remove the already used first)
        dotplot_args = [
            "var_names", "groupby", "dendrogram", "cmap", "save", "use_raw",
            "log", "layer", "expression_cutoff", "swap_axes"]
        run_dict = {k: v for k, v in config["pl"]["dotplot"].items() if k not in dotplot_args}
        # TODO: use sc_plot.dotplot!
        sc.pl.dotplot(
            adata, sub_dict, config["general"]["cluster_algorithm"],
            # dendrogram=True, cmap=get_colormap(cmap=config["general"]["cmap"]),
            dendrogram=True, cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]),
            save=save_path, use_raw=False, log=False,
            layer="log2norm_counts", expression_cutoff=0,
            swap_axes=config["to_plot"]["down"]["cluster_dotplot"]["swap_axes"],
            **run_dict)
    # ########################################################################################
    # Generate gene dendrogram if specified
    if var_dendrogram:
        create_gene_dendrogram(
            adata, all_variables_per_cluster, config,
            fig_name=f"DEG_per_cluster_hierarchical_clustering{img_suffix}",
            top_n_genes=top_n_genes, figsize=(10, 5), part=part)

    return all_variables_per_cluster, variables_per_cluster


def plot_per_group_stacked_violins(
            adata: ad.AnnData,
            config: dict | None = None,
            highly_variable: bool = False,
            top_n_genes: int = 5,
            part: str = "/downstream/",
            overlap_fill_to_top_n: bool = False,
            deg_key: str = "rank_genes_groups",
            perc: float = 0.15,
            p_val_cutoff: float = 0.01,
            lfc: float = 0.5,
            direction: str = "force_up_n_down",
            with_figure_explaination: bool = True
        ) -> None:
    """
    Creates stacked violin plots of differentially expressed genes (DEGs) per cluster.

    This function generates and saves stacked violin plots for each cluster to
    visualize DEG expression across groups of cells. DEGs are selected based on
    statistical thresholds and optional overlap controls. The resulting plots
    are useful for comparative expression analysis.

    Args:
        adata (anndata.AnnData): Adata object.
        config (dict | None, optional): Configuration dictionary with plotting
            and filtering parameters. If None, uns["config"]``. Defaults to
            None.
        highly_variable (bool, optional): Whether to restrict to highly variable
            genes. Not yet implemented. Defaults to False.
        top_n_genes (int, optional): Number of top DEGs to display per cluster.
            Defaults to 5.
        part (str, optional): Path for saving the plots.
            Defaults to "/downstream/".
        overlap_fill_to_top_n (bool, optional): If True, fill top_n genes even
            if there is overlap between clusters. Defaults to False.
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        perc (float, optional): Percentile cutoff used for DEG filtering.
            Defaults to 0.15.
        p_val_cutoff (float, optional): P-value threshold for filtering DEGs.
            Defaults to 0.01.
        lfc (float, optional): Log fold change threshold for selecting DEGs.
            Defaults to 0.5.
        direction (str, optional): Direction of DEGs to include ("up", "down",
            or "force_up_n_down"). Defaults to "force_up_n_down".
        with_figure_explaination (bool, optional): Whether to log where plots
            are saved. Defaults to True.

    Returns:
        None

    Calls:
        get_DEGs_per_group_for_plotting, get_save_path

    Called By:
        run_downstream

    TODO:
        Add support for highly variable gene filtering to improve
        interpretability and speed.
    """
    # ########################################################################################
    # Initialize configuration if not provided
    if config is None:
        config = adata.uns["config"]
    # ########################################################################################
    # Determine the order of groups for plotting
    groupby = config["general"]["cluster_algorithm"]
    groups = adata.obs[groupby].cat.categories.tolist()
    # ########################################################################################
    # Handle the option to subset to highly variable genes
    if highly_variable:
        raise NotImplementedError("This is not implemented yet, look at the code and finish it!")
    # ########################################################################################
    # Retrieve DEGs for each cluster for plotting
    variables_per_cluster, _ = get_DEGs_per_group_for_plotting(
        adata=adata, top_n_genes=top_n_genes, groups=groups,
        overlap_fill_to_top_n=overlap_fill_to_top_n, deg_key=deg_key,
        perc=perc, p_val_cutoff=p_val_cutoff, lfc=lfc, direction=direction)
    # ########################################################################################
    # Generate and save stacked violin plots for each cluster
    for cluster, ref in variables_per_cluster.items():
        # ##########################################
        # Define the save path for each plot
        save_path = get_save_path(
            f"cluster_DEG_dotplot_cluster_{cluster}_top{top_n_genes}.pdf", config, part)
        # ##########################################
        # Optionally log information about the saved figure
        if with_figure_explaination:
            logger.info(f'The figure is saved as {save_path}')
        # ##########################################
        # Create and save the stacked violin plot
        sc.pl.stacked_violin(
            adata, ref, save=save_path,
            groupby=groupby,
            show=config["general"]["show_plots"],
            title=f"cluster_DEG_dotplot_cluster_{cluster}_top{top_n_genes}",
            dendrogram=False)


def plot_per_group_DEG_umaps(
            adata: ad.AnnData,
            config: dict | None = None,
            top_n_genes: int = 5,
            groupby: str | None = None,
            groups: list[str] | None = None,
            part: str = "/downstream/",
            deg_key: str = "rank_genes_groups",
            overlap_fill_to_top_n: bool = False,
            perc: float = 0.15,
            p_val_cutoff: float = 0.01,
            lfc: float = 0.01,
            direction: str = "force_up_n_down",
            with_figure_explaination: bool = True,
            highlighers: str = "#" * 120
        ) -> None:
    """
    Creates UMAP visualizations for Differentially Expressed Genes (DEGs) per cluster.

    This function generates and saves UMAP plots of DEGs identified per cluster.
    The genes are selected based on p-value, log fold change, and percentile
    filtering criteria. It supports both continuous and discrete gradient
    visualization styles and can handle upregulated, downregulated, or both
    types of DEGs depending on the ``direction``.

    NOTE:
        Requires UMAP coordinates and precomputed DEGs in the ``adata`` object.

    Args:
        adata (anndata.AnnData): Adata object containing DEGs.
        config (dict | None, optional): Configuration dictionary. If None, uses
            ``adata.uns["config"]``. Defaults to None.
        top_n_genes (int, optional): Number of top genes to include per cluster.
            Defaults to 5.
        groupby (str | None, optional): Optional key to group data by, must be a
            column in adata.obs. Defaults to None.
        groups (list[str] | None, optional): Specific groups to analyze. If
            None, uses cluster hierarchy from config. Defaults to None.
        part (str, optional): Output directory path for saved UMAP plots.
            Defaults to "/downstream/".
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        overlap_fill_to_top_n (bool, optional): If True, allows overlapping
            genes between groups to still be filled up to top_n. Defaults to
            False.
        perc (float, optional): Percentile threshold for DEG filtering.
            Defaults to 0.15.
        p_val_cutoff (float, optional): P-value cutoff for DEG significance.
            Defaults to 0.01.
        lfc (float, optional): Log fold change threshold. Defaults to 0.01.
        direction (str, optional): DEG direction to use ("up", "down", or
            "force_up_n_down"). Defaults to "force_up_n_down".
        with_figure_explaination (bool, optional): Whether to log saved figure
            paths. Defaults to True.
        highlighers (str, optional): String used for log emphasis.
            Defaults to "#" * 120.

    Returns:
        None:
            This function produces and saves UMAP plots without returning a
            value.

    Calls:
        continuos_umap_helper, discrete_umap_helper, get_DEGs_per_group,
        validate_groupby_column

    Called By:
        run_downstream

    TODO:
        Improve performance for large numbers of groups and optionally combine
        UMAPs into a single figure.
    """
    # ####################################################
    # Initialize configuration if not provided
    if config is None:
        config = adata.uns["config"]
    # ####################################################
    # get the clustering algorithm key
    if groupby is None:
        groupby = config["general"]["cluster_algorithm"]
    # get the groups
    if groups is None:
        # groups = list(adata.uns[key]["names"].dtype.names)
        groups = adata.obs[groupby].unique().tolist()
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=False, groups=groups)
    # ####################################################
    # Initialize dictionaries to store variables per cluster
    variables_per_cluster = {}
    all_variables_per_cluster = []
    # ####################################################
    # Get the genes per cluster based on the specified direction
    if direction == "force_up_n_down":
        variables_up = get_DEGs_per_group(
            adata, groups=groups, n_genes=1e200, perc=perc,
            deg_key=deg_key, p_val_cutoff=p_val_cutoff,
            lfc=lfc, direction="up")
        variables_down = get_DEGs_per_group(
            adata, groups=groups, n_genes=1e200, perc=perc,
            deg_key=deg_key, p_val_cutoff=p_val_cutoff,
            lfc=lfc, direction="down")

        for c in variables_up.keys():
            if overlap_fill_to_top_n:
                genes_to_add_up = [
                    x for x in variables_up[c] if x not in all_variables_per_cluster]
                genes_to_add_down = [
                    x for x in variables_down[c] if x not in all_variables_per_cluster]
            else:
                genes_to_add_up = [
                    x for x in variables_up[c][:top_n_genes]
                    if x not in all_variables_per_cluster]
                genes_to_add_down = [
                    x for x in variables_down[c][:top_n_genes]
                    if x not in all_variables_per_cluster]

            genes_to_add = (genes_to_add_up[:top_n_genes // 2]
                            + genes_to_add_down[:top_n_genes // 2])

            if len(genes_to_add) > 0:
                all_variables_per_cluster.extend(genes_to_add)
                variables_per_cluster[c] = genes_to_add

    else:
        variables = get_DEGs_per_group(
            adata, groups=groups, n_genes=None, perc=perc,
            deg_key=deg_key, p_val_cutoff=p_val_cutoff,
            lfc=lfc, direction=direction)

        for c in variables.keys():
            if overlap_fill_to_top_n:
                genes_to_add = [
                    x for x in variables[c] if x not in all_variables_per_cluster]
            else:
                genes_to_add = [
                    x for x in variables[c][:top_n_genes]
                    if x not in all_variables_per_cluster]

            genes_to_add = genes_to_add[:top_n_genes]

            if len(genes_to_add) > 0:
                all_variables_per_cluster.extend(genes_to_add)
                variables_per_cluster[c] = genes_to_add
    # ####################################################
    # Create the UMAPs for the DEGs per cluster using the selected configuration
    if config["to_plot"]["down"]["gradients_umap"]["discretize"]:
        for k, ref in variables_per_cluster.items():
            discrete_umap_helper(
                adata, keys=ref, part=part,
                with_figure_explaination=with_figure_explaination,
                name=f'{k}_DEGs', highlighers=highlighers, config=config)
    if config["to_plot"]["down"]["gradients_umap"]["continuos"]:
        for k, ref in variables_per_cluster.items():
            continuos_umap_helper(
                adata, keys=ref, part=part,
                with_figure_explaination=with_figure_explaination,
                name=f'{k}_DEGs', highlighers=highlighers, config=config)


def plot_ridgeline(
            data: list[list[float]],
            overlap: float = 0,
            fill: bool = True,
            labels: list[str] | None = None,
            n_points: int = 150
        ) -> None:
    """Plots a ridgeline visualization from multiple data distributions.

    This function creates a ridgeline plot using a list of 1D data arrays. Each
    distribution is smoothed using kernel density estimation and vertically
    offset to create a layered view. This can highlight overlap, separation, or
    trends across multiple groups.

    NOTE:
        This implementation is adapted from:
        https://glowingpython.blogspot.com/2020/03/ridgeline-plots-in-pure-matplotlib.html

    Args:
        data (list[list[float]]): A list where each sublist contains values for
            a distribution.
        overlap (float, optional): Degree of vertical overlap between plots, in
            [0, 1]. Higher values reduce vertical spacing. Defaults to 0.
        fill (bool, optional): Whether to fill the area under each density
            curve. Defaults to True.
        labels (list[str] | None, optional): Y-axis labels corresponding to each
            distribution. If None, no labels are shown. Defaults to None.
        n_points (int, optional): Number of interpolation points for density
            curves. Defaults to 150.

    Returns:
        None:
            The function generates a matplotlib plot in-place and does not
            return a value.

    Raises:
        ValueError: If ``overlap`` is not between 0 and 1 (inclusive).

    Called By:
        show_category_over_umap

    TODO:
        Add support for custom colors, axis scaling, and automated subplot
        generation for grouped distributions.
    """
    # ###########################################################
    # Validate the overlap parameter
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0, 1]")
    # ###########################################################
    # Generate a range of x values covering all data distributions
    xx = np.linspace(
        np.min(np.concatenate(data)),
        np.max(np.concatenate(data)),
        n_points)
    # ###########################################################
    # Initialize an empty list to store y-axis positions of distributions
    ys = []
    # ###########################################################
    # Iterate over the data and plot each distribution
    for i, d in enumerate(data):
        # TODO: Replace the gaussian_kde with the fftkde
        pdf = gaussian_kde(d)  # Estimate the probability density function
        y = i * (1.0 - overlap)  # Adjust y position based on overlap
        ys.append(y)  # Store the y-axis position
        # ##########################################
        # Plotting the curve, filling if specified
        curve = pdf(xx)
        if fill:
            plt.fill_between(
                xx, np.ones(n_points) * y,
                curve + y, zorder=len(data) - i + 1, color=fill)
        plt.plot(xx, curve + y, c='k', zorder=len(data) - i + 1)
    # ###########################################################
    # Set y-axis labels if provided
    if labels:
        plt.yticks(ys, labels)


def plot_hierarchical_heatmap(
            hallmark_counts_df: pd.DataFrame,
            hallmark_col: str = "normalized_by_genes",
            plot_clusters_sorted: bool = False,
            figsize: Sequence[float | int] = (10, 7),
            show_values: bool = True,
            cmap: Colormap | None = None,
            save_path: str | None = None,
            **kwargs
        ) -> None:
    """Generates a hierarchical heatmap from a hallmark count DataFrame.

    This function creates a heatmap based on values in ``hallmark_counts_df``,
    using hierarchical clustering of rows and optionally of columns. If
    ``plot_clusters_sorted`` is enabled, clustering of columns is disabled and
    they are sorted alphabetically. Color gradients are customized depending on
    whether values are positive, negative, or mixed.

    NOTE:
        Keyword arguments passed to this function are forwarded to ``sns.clustermap``.

        Special options include:
            - z_score (int | None): Apply z-scoring across rows (0) or columns(1).
            - standard_scale (int | None): Apply scaling across rows (0) or
              columns (1).

    Args:
        hallmark_counts_df (pandas.DataFrame): Input DataFrame with columns
            ``hallmark``, ``group``, and a value column.
        hallmark_col (str, optional): Column to plot in the heatmap. Defaults to
            ``normalized_by_genes``. Defaults to "normalized_by_genes".
        plot_clusters_sorted (bool, optional): If True, columns are sorted
            alphabetically instead of clustered. Defaults to False.
        figsize (Sequence[float | int], optional): Size of the heatmap figure.
            Defaults to (20, 15).
        show_values (bool, optional): If True, displays values inside heatmap
            cells. Defaults to True.
        cmap (Colormap | None, optional): Colormap for cell values. If None, a
            suitable colormap is selected based on the column name. Defaults to
            None.
        save_path (str | None, optional): File path to save the output figure.
            Defaults to None.
        **kwargs: Additional keyword arguments for ``sns.clustermap``, such as
            ``z_score`` or ``standard_scale``.

    Returns:
        None:
            This function displays and optionally saves the heatmap.

    Raises:
        ValueError: If ``hallmark_col`` is not found in ``hallmark_counts_df``.

    Called By:
        plot_hallmark_group_heatmap

    TODO:
        Consider adding options for exporting the heatmap as interactive HTML.
    """
    # #########################################################
    # Pivot the DataFrame to create the data structure needed for the heatmap
    heatmap_data = hallmark_counts_df.pivot(index="hallmark", columns="group", values=hallmark_col)
    # #########################################################
    # Handle missing data by filling NaN values with 0
    heatmap_data = heatmap_data.fillna(0)
    # #########################################################
    # Generate the linkage matrix for hierarchical clustering of rows using Ward's method
    row_linkage = linkage(heatmap_data, method="ward", metric="euclidean")
    # #########################################################
    # Handle column clustering or sorting based on the plot_clusters_sorted flag
    if plot_clusters_sorted:
        # ##########################################
        # Sort the columns by group names if sorting is enabled
        sorted_groups = [str(x) for x in sorted(hallmark_counts_df["group"].unique().astype(int))]
        heatmap_data = heatmap_data[sorted_groups]
        col_linkage = None  # Disable column clustering when sorted by name
    else:
        # ##########################################
        # Perform hierarchical clustering on columns
        col_linkage = linkage(heatmap_data.T, method="ward", metric="euclidean")
    # #########################################################
    # Determine the colormap to be used for the heatmap
    if cmap is None:
        if "_pos_" in hallmark_col:
            cmap = LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"])
        elif "_neg_" in hallmark_col:
            cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#0000ff", "#ffffff"])
        else:
            cmap = LinearSegmentedColormap.from_list(
                "white_to_blue", ["#0000ff", "#ffffff", "#ff0000"])

    # #########################################################
    # set Defaults
    kwargs.setdefault("dendrogram_ratio", (0.05, 0.05))
    # #########################################################
    # Create the clustered heatmap with the specified settings
    ax = clustermap(
        heatmap_data,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cmap=cmap,
        annot=show_values,
        fmt=".2f",
        figsize=figsize,
        col_cluster=not plot_clusters_sorted,
        xticklabels=True,
        yticklabels=True,
        **kwargs)
    # move the plot down to dodge the title
    ax.fig.subplots_adjust(top=0.9)
    # [x, y, width, height] in figure fraction
    ax.cax.set_position([1.03, 0.3, 0.02, 0.4])

    # #########################################################
    # Add a title to the heatmap
    ax.figure.suptitle(
        f'Hierarchical Heatmap of Hallmark Counts Normalized by '
        f'{hallmark_col.replace("_", " ").title()}')
    # #########################################################
    # Save the figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    # #########################################################
    # Display the heatmap
    plt.show()


def plot_hierarchical_heatmap_(
            hallmark_counts_df: pd.DataFrame,
            hallmark_col: str = "normalized_by_genes",
            plot_clusters_sorted: bool = False,
            figsize: Sequence[float | int] = (15, 12),
            show_values: bool = True,
            cmap: Colormap | None = None,
            save_path: str | None = None,
            **kwargs
        ) -> None:
    """Generates a hierarchical heatmap from a hallmark count DataFrame.

    This function creates a heatmap based on values in ``hallmark_counts_df``,
    using hierarchical clustering of rows and optionally of columns. If
    ``plot_clusters_sorted`` is enabled, clustering of columns is disabled and
    they are sorted alphabetically. Color gradients are customized depending on
    whether values are positive, negative, or mixed.

    NOTE:
        Keyword arguments passed to this function are forwarded to ``Heatmap``.

        Special options include:
            - z_score (int | None): Apply z-scoring across rows (0) or columns(1).
            - standard_scale (int | None): Apply scaling across rows (0) or
              columns (1).

    Args:
        hallmark_counts_df (pandas.DataFrame): Input DataFrame with columns
            ``hallmark``, ``group``, and a value column.
        hallmark_col (str, optional): Column to plot in the heatmap. Defaults to
            ``normalized_by_genes``. Defaults to "normalized_by_genes".
        plot_clusters_sorted (bool, optional): If True, columns are sorted
            alphabetically instead of clustered. Defaults to False.
        figsize (Sequence[float | int], optional): Size of the heatmap figure.
            Defaults to (20, 15).
        show_values (bool, optional): If True, displays values inside heatmap
            cells. Defaults to True.
        cmap (Colormap | None, optional): Colormap for cell values. If None, a
            suitable colormap is selected based on the column name. Defaults to
            None.
        save_path (str | None, optional): File path to save the output figure.
            Defaults to None.
        **kwargs: Additional keyword arguments for ``Heatmap``.

    Returns:
        None:
            This function displays and optionally saves the heatmap.

    Raises:
        ValueError: If ``hallmark_col`` is not found in ``hallmark_counts_df``.

    Called By:
        plot_hallmark_group_heatmap

    TODO:
        Consider adding options for exporting the heatmap as interactive HTML.
    """
    # #########################################################
    # Pivot the DataFrame to create the data structure needed for the heatmap
    heatmap_data = hallmark_counts_df.pivot(index="hallmark", columns="group", values=hallmark_col)
    # #########################################################
    # Handle missing data by filling NaN values with 0
    heatmap_data = heatmap_data.fillna(0)
    # #########################################################
    # Generate the linkage matrix for hierarchical clustering of rows using Ward's method
    row_linkage = linkage(heatmap_data, method="ward", metric="euclidean")
    # #########################################################
    # Handle column clustering or sorting based on the plot_clusters_sorted flag
    if plot_clusters_sorted:
        # ##########################################
        # Sort the columns by group names if sorting is enabled
        sorted_groups = [str(x) for x in sorted(hallmark_counts_df["group"].unique().astype(int))]
        heatmap_data = heatmap_data[sorted_groups]
        col_linkage = None  # Disable column clustering when sorted by name
    else:
        # ##########################################
        # Perform hierarchical clustering on columns
        col_linkage = linkage(heatmap_data.T, method="ward", metric="euclidean")
    # #########################################################
    # Determine the colormap to be used for the heatmap
    if cmap is None:
        if "_pos_" in hallmark_col:
            cmap = LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"])
        elif "_neg_" in hallmark_col:
            cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#0000ff", "#ffffff"])
        else:
            cmap = LinearSegmentedColormap.from_list(
                "white_to_blue", ["#0000ff", "#ffffff", "#ff0000"])
    # #########################################################
    # Create the clustered heatmap with the specified settings
    h = Heatmap(
        heatmap_data,
        cmap=cmap,
        annot=show_values,
        fmt=".2f",
        width=figsize[0],
        height=figsize[1],
        **kwargs)
    if row_linkage is not None:
        h.add_dendrogram("left", linkage=row_linkage)
    if col_linkage is not None:
        h.add_dendrogram("top", linkage=col_linkage)
    # #########################################################
    # Add a title to the heatmap
    h.add_title(
        top=f'Hierarchical Heatmap of Hallmark Counts Normalized by '
            f'{hallmark_col.replace("_", " ").title()}',
        align="center")
    h.add_legends()
    h.render()
    # #########################################################
    # Save the figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    # #########################################################
    # Display the heatmap
    plt.show()


def plot_hallmark_group_heatmap(
            adata: ad.AnnData,
            score_to_use: str = "normalized_total_scores_by_genes",
            use_minmax: bool = True,
            deg_df: pd.DataFrame | None = None,
            hallmark_df: pd.DataFrame | None = None,
            subset_pct: float = 0.1,
            plot_clusters_sorted: bool = False,
            show_values: bool = True,
            organism: str = "human",
            deg_key: str = "rank_genes_groups",
            return_df: bool = False,
            **kwargs
        ) -> None:
    """
    Generates a clustered heatmap of hallmark gene set scores across cell groups.

    This function retrieves differential expression results from an adata
    object, maps them to hallmark gene sets from MSigDB, computes enrichment
    scores, and visualizes them in a hierarchical heatmap. Users can filter
    low-variance hallmark pathways and optionally sort clusters instead of
    performing dendrogram-based clustering.

    NOTE:
        - Ensure that the adata object contains differential expression results
          under the key provided in adata.uns[deg_key].
        - save_path is possible here via kwargs

    Args:
        adata (anndata.AnnData): Adata object containing DEGs.
        score_to_use (str, optional): Whether to annotate cells with their
            numeric values. Defaults to "normalized_total_scores_by_genes".

                - "total_count": The total count of Genes (pos - neg)
                - "total_scores": The total count of wilcoxon scores (pos - neg)
                - "total_normalized_by_genes": like total_count but normalized
                    to the number of genes in the geneset.
                - "normalized_total_scores_by_genes": like total_scores but
                    normalized to the number of genes in the geneset.

        use_minmax (bool, optional): Whether to scale the data between [-1, 1].
            Defaults to True.
        deg_df (pandas.DataFrame | None, optional): Optional DEG DataFrame.
            If None, it is retrieved internally if Precalculated.
            Defaults to None.
        hallmark_df (pandas.DataFrame | None, optional): DataFrame with hallmark
            definitions containing "GeneSet" and "geneSymbols" columns. If None,
            pulled from MSigDB for the specified organism. Defaults to None.
        subset_pct (float, optional): Percentage of top varying hallmarks to
            keep (by std deviation). Hallmarks below this variability are
            excluded. Defaults to 0.1.
        plot_clusters_sorted (bool, optional): If True, cluster columns are
            sorted alphabetically instead of hierarchically clustered. Defaults
            to False.
        show_values (bool, optional): Whether to annotate cells with their
            numeric values. Defaults to True.
        organism (str, optional): Organism name used to retrieve MSigDB gene
            sets. Defaults to "human".
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        return_df (bool, optional): If True, returns the sorted Jaccard matrix
            DataFrame. Defaults to False.
        **kwargs: Additional keyword arguments passed to
            ``plot_hierarchical_heatmap``.

    Returns:
        None:
            This function creates and displays a heatmap. It does not return
            any value.

    Raises:
        ValueError: If differential expression data or hallmark gene sets cannot
            be found or parsed.

    Calls:
        count_genesets_per_group, get_deg_df, get_msigdb_df,
        map_geneset_to_degs, plot_hierarchical_heatmap, subset_degs
    """
    # #########################################################
    # Retrieve and filter the differential expression data
    deg_df = get_deg_df(adata, deg_key=deg_key)
    # ##########################################
    # Apply filtering criteria to the DEGs
    deg_df = subset_degs(
        deg_df, n_genes=1e300, perc=0.15, p_val_cutoff=0.0001,
        lfc=0.5, direction="up_n_down")
    # #########################################################
    # Retrieve and filter the MSigDB hallmark gene sets
    if hallmark_df is None:
        hallmark_df = get_msigdb_df(organism=organism, only_hallmarks=True)
    # ##########################################
    # Map the hallmarks to the filtered differential expression data
    updated_deg_df = map_geneset_to_degs(deg_df, geneset_df_row_wise=hallmark_df)
    # #########################################################
    # Calculate and normalize the hallmark scores
    hallmark_counts_df = count_genesets_per_group(updated_deg_df)
    data = hallmark_counts_df[score_to_use]
    use_norm = True
    if use_minmax:
        use_col = score_to_use + "_minmax"
        # Normalize total scores between -1 and 1

        # hallmark_counts_df[use_col] = (
        #     (data - data.min()) / (data.max() - data.min()) * 2 - 1)
        # Above-zero rows → rescale only those to max = 1
        ids_ = data > 0
        hallmark_counts_df.loc[ids_, use_col] = (
            data[ids_] / data[ids_].max())

        # Below-zero rows → rescale only those to min = -1
        ids_ = data < 0
        hallmark_counts_df.loc[ids_, use_col] = (
            data[ids_] / data[ids_].min() * -1)
        # for the 2 slope norm
        vmin = -1.0
        vmax = 1.0
        vcenter = 0.0
        use_norm = True

    else:
        use_col = score_to_use

        vmin = float(data.min())
        vmax = float(data.max())
        vcenter = 0.0
        if vmin < 0 and vmax > 0:
            use_norm = True

    if use_norm:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        kwargs["norm"] = norm

    # #########################################################
    # Filter hallmarks based on standard deviation and subset percentage
    vc = hallmark_counts_df.groupby("hallmark")[score_to_use].apply(lambda x: x.std()).sort_values()
    good_hm = list(vc[vc > vc.max() * subset_pct].index)
    hallmark_counts_df = hallmark_counts_df[hallmark_counts_df["hallmark"].isin(good_hm)]
    if hallmark_counts_df.shape[0] == 0:
        raise ValueError("There seems to be no overlap between the DEGs and the Hallmarks!")
    # #########################################################
    # Plot the hierarchical heatmap
    plot_hierarchical_heatmap(
        hallmark_counts_df,
        hallmark_col=use_col,
        plot_clusters_sorted=plot_clusters_sorted,
        show_values=show_values,
        **kwargs)
    # #########################################################
    # Return if desired
    if return_df:
        return hallmark_counts_df


def plot_jaccard_heatmap_cluster_comparison(
            labels_1: np.ndarray | pd.Series,
            labels_2: np.ndarray | pd.Series,
            figsize: Sequence[float | int] = (20, 10),
            return_df: bool = False,
            return_mapping: bool = False,
            show: bool = True,
            scale_axis: int | None = None,
            save_path: str = "",
        ) -> list[pd.DataFrame | dict[str, str] | plt.Axes]:
    """
    Plots a heatmap of Jaccard scores between two cluster label sets,
    aligning highest values near the diagonal.

    Args:
        labels_1 (numpy.ndarray): Cluster labels from the first clustering.
        labels_2 (numpy.ndarray): Cluster labels from the second clustering.
        figsize (Sequence[float | int], optional): Figure size for the heatmap.
            Defaults to (20, 10).
        return_df (bool, optional): If True, returns the sorted Jaccard matrix
            DataFrame. Defaults to False.
        return_mapping (bool, optional): If True, returns a mapping from
            clustering 2 labels to clustering 1 labels. Defaults to False.
        show (bool, optional): If True executes plt.show(). Defaults to True.
        scale_axis (int | None, optional): If not None, applies min-max scaling
            to the Jaccard matrix along the given axis (0=row-wise, 1=col-wise).
            Defaults to None.
        save_path (str, optional): The directory path where the experiment data
            should be saved. Defaults to "".

    Returns:
        list[pandas.DataFrame | dict[str, str] | matplotlib.pyplot.Axes]:
            List containing any of these in that order dependant on the args:

                - Jaccard score DataFrame
                - Mapping from clustering 2 labels to clustering 1 labels
                - Matplotlib Axes object (only if ``show`` is False)
    """
    # ###############################################################
    # Fix input to be processed in the same way
    # normalize labels_1
    if isinstance(labels_1, pd.Series) and pd.api.types.is_categorical_dtype(labels_1):
        mapping_1 = dict(enumerate(labels_1.cat.categories))
        labels_1 = labels_1.cat.codes.to_numpy()
    elif not np.issubdtype(np.asarray(labels_1).dtype, np.integer):
        labels_1 = pd.Series(labels_1, dtype="category")
        mapping_1 = dict(enumerate(labels_1.cat.categories))
        labels_1 = labels_1.cat.codes.to_numpy()
    else:
        labels_1 = np.asarray(labels_1)
        mapping_1 = {int(x): int(x) for x in np.unique(labels_1)}
    # normalize labels_2
    if isinstance(labels_2, pd.Series) and pd.api.types.is_categorical_dtype(labels_2):
        mapping_2 = dict(enumerate(labels_2.cat.categories))
        labels_2 = labels_2.cat.codes.to_numpy()
    elif not np.issubdtype(np.asarray(labels_2).dtype, np.integer):
        labels_2 = pd.Series(labels_2, dtype="category")
        mapping_2 = dict(enumerate(labels_2.cat.categories))
        labels_2 = labels_2.cat.codes.to_numpy()
    else:
        labels_2 = np.asarray(labels_2)
        mapping_2 = {int(x): int(x) for x in np.unique(labels_2)}

    # ###############################################################
    # Calculate the Jaccard score between the two clusterings for each cluster pair
    unique_labels_1 = np.unique(labels_1)
    unique_labels_2 = np.unique(labels_2)
    # ###############################################################
    # Calculate the Jaccard score between the two clusterings for each cluster pair
    jaccard_matrix = np.zeros((len(unique_labels_1), len(unique_labels_2)))

    for i, label_1 in enumerate(unique_labels_1):
        for j, label_2 in enumerate(unique_labels_2):
            cluster_1 = (labels_1 == label_1).astype(int)
            cluster_2 = (labels_2 == label_2).astype(int)
            score = jaccard_score(cluster_1, cluster_2)
            jaccard_matrix[i, j] = score
    # ##############################################################
    # Use original labels for index/columns and create a dataframe
    df = pd.DataFrame(
        jaccard_matrix,
        index=[mapping_1[i] for i in unique_labels_1],
        columns=[mapping_2[j] for j in unique_labels_2])
    # ##############################################################
    # SCALE if requested
    if scale_axis is not None:
        min_max_scale_axis(df, axis=scale_axis, inplace=True)
    # ##############################################################
    sorted_df_rows = df.loc[df.max(axis=1).sort_values(ascending=False).index]
    row_clusters = sorted_df_rows.index.tolist()
    max_col_indices = sorted_df_rows.idxmax(axis=0)
    rows_max = {k: [] for k in row_clusters}
    for k, v in max_col_indices.items():
        rows_max[v].append(k)

    rows_max_sorted = copy(rows_max)
    best_match_sum = {}
    for k, v in rows_max.items():
        if len(v) > 1:
            rows_max_sorted[k] = np.array(rows_max[k])[np.argsort(sorted_df_rows.loc[k, v])[::-1]].tolist()
        best_match_sum[k] = sorted_df_rows.loc[k, v].values.flatten().sum()

    sort_cols = []
    for v in rows_max_sorted.values():
        sort_cols.extend(v)

    sorted_df_rows = sorted_df_rows[sort_cols]

    plt.figure(figsize=figsize)
    ax = heatmap(
        sorted_df_rows, annot=True, cmap='viridis', fmt=".2f",
        cbar=True)
    plt.title('Aligned Diagonal Heatmap of Jaccard Scores')
    plt.xlabel('Cluster Labels of Clustering 2')
    plt.ylabel('Cluster Labels of Clustering 1')

    to_return = []

    if return_df:
        to_return.append(sorted_df_rows)
    if return_mapping:
        to_return.append({str(v): str(k) for k, v in sorted_df_rows.idxmax().items()})

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        to_return.append(ax)

    return to_return


def plot_umap_cat_splitting(
            adata: ad.AnnData,
            keys: list[str],
            plot_densitys: bool = True,
            figsize: Sequence[float | int] = (8, 6),
            dpi: int = 100,
            umap_kwargs: dict = {},
            density_kwargs: dict = {}
        ) -> None:
    """Visualizes UMAP with clustering and optionally density plots.

    NOTE:
        The overall notebook display size is controlled by `figsize` and
        `dpi`. For subfigure sizing, use the `umap` and `density` kwargs.
        These kwargs are not listed under `Args`.

    Args:
        adata (anndata.AnnData): Adata object.
        keys (list[str]): Features or groups to color UMAP by.
        plot_densitys (bool, optional): If True, plots density plots for
            categorical vars. Defaults to True.
        figsize (Sequence[float | int], optional): Figure size in inches for the
            overall figure. Together with `dpi`, it controls the notebook display
            size. Defaults to (8, 6).
        dpi (int, optional): Resolution of the figure in dots per inch. Together
            with `figsize`, it controls the notebook display size. Defaults to 100.
        umap_kwargs (dict): Extra keyword arguments passed to plot_umap.
        density_kwargs (dict): Extra keyword arguments passed to
            plot_embedding_density.

    Returns:
        None

    Calls:
        get_adata_sub_keys, get_categorical_columns,
        plot_embedding_density, plot_umap

    Called By:
        search_for_leiden_resolution, search_for_neighbor_params
    """
    if len(keys) < 2:
        return

    with rc_context({'figure.figsize': figsize, 'figure.dpi': dpi}):
        if not plot_densitys:
            plot_umap(adata, keys=keys, s=20, **umap_kwargs)
        else:
            obsm_keys = umap_kwargs.get("obsm_keys", None)
            temp_data = get_adata_sub_keys(adata, keys, obsm_keys=obsm_keys)
            categorical, non_categorical = get_categorical_columns(temp_data)

            if non_categorical:
                plot_umap(adata, keys=non_categorical, s=20, **umap_kwargs)

            for key in categorical:
                plot_embedding_density(adata, groupby=key, **density_kwargs)


def calc_hierarchical_order(
            arrays: list[np.ndarray],
            axis: int = 0,
            substitute: float | None = 0.0,
            return_linkage: bool = False,
            **kwargs
        ) -> list[int] | tuple[list[int], linkage]:
    """
    Computes a dendrogram-based ordering of either rows or columns across one
    or more matrices.

    Args:
        arrays (list[np.ndarray]): List of 2D arrays to combine for clustering.
        axis (int, optional): 0 = cluster rows, 1 = cluster columns (default:
            0). Defaults to 0.
        substitute (float | None, optional): Value to replace NaNs. None = no
            substitution. Defaults to 0.0.
        return_linkage (bool, optional): If True, also return the linkate
            Defaults to False.
        **kwargs: Additional arguments passed to
            ``scipy.cluster.hierarchy.linkage``.

    Returns:
        list[int] | tuple[list[int], linkage]:

            - list[int]: If not return_linkage, Ordered indices from hierarchical
              clustering.
            - tuple[list[int], linkage]: If return_linkage,
              Ordered indices from hierarchical clustering and Linkage

    Called By:
        plot_dotplot, plot_split_dotplot
    """
    if axis == 0:
        combined_data = np.hstack(arrays)
    elif axis == 1:
        combined_data = np.hstack([arr.T for arr in arrays])
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    if substitute is not None:
        if not np.issubdtype(combined_data.dtype, np.floating):
            combined_data = combined_data.astype(float)
        combined_data = np.nan_to_num(combined_data, nan=substitute)

    kwargs.setdefault("method", "ward")
    kwargs.setdefault("metric", "euclidean")

    linkage_matrix = linkage(combined_data, **kwargs)
    if not return_linkage:
        return leaves_list(linkage_matrix).tolist()
    else:
        return leaves_list(linkage_matrix).tolist(), linkage_matrix


# ###################################################################################################
# Dotplot
def make_circle_segment_markers(
            n_segments: int,
            n_points: int = 100,
            offset: float = 0.0,
        ) -> list[mpl_path]:
    """Create equal-size circular sector markers (pie slices).

    Each marker represents one circular sector of equal angle. The apex
    can optionally be shifted outward along the bisector.

    Args:
        n_segments (int): Number of equal circle sectors. Must be >= 1.
        n_points (int, optional): Number of points along each arc.
            Defaults to 100.
        offset (float, optional): Distance to move the apex outward along
            the sector bisector. Defaults to 0.0.

    Returns:
        list[MplPath]: List of Path objects usable as matplotlib markers.

    Raises:
        ValueError: If ``n_segments`` is less than 1.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    markers: list[mpl_path] = []

    # apply rotation of pi/2 so n=2 is vertical
    rotation = np.pi / 2

    for i in range(n_segments):
        theta_start = 2 * np.pi * i / n_segments + rotation
        theta_end = 2 * np.pi * (i + 1) / n_segments + rotation
        theta_mid = (theta_start + theta_end) / 2
        theta = np.linspace(theta_start, theta_end, n_points)

        arc = [(np.cos(t), np.sin(t)) for t in theta]

        # apex shifted outward along bisector
        apex = (offset * np.cos(theta_mid), offset * np.sin(theta_mid))

        verts = [apex] + arc + [apex]
        codes = [mpl_path.MOVETO] + [mpl_path.LINETO] * len(arc) + [mpl_path.CLOSEPOLY]

        # enforce full circle bounding box with dummy points
        verts += [(1, 0), (-1, 0), (0, 1), (0, -1)]
        codes += [mpl_path.CLOSEPOLY] * 4

        markers.append(mpl_path(verts, codes))

    return markers


def _setup_norm_and_legend(
            data_list: list[pd.DataFrame],
            norm: bool | None | Normalize,
            vmin: float | None,
            vmax: float | None,
            title: str,
            legend_kws: dict = {},
        ) -> tuple[Normalize | None, dict | None]:
    """Handle normalization and legend title logic for dotplot.

    Args:
        data_list (list[pandas.DataFrame]): List of DataFrames with values
            for this feature.
        norm (bool | None | Normalize): Normalization control.
            - True: Create a Normalize from values in ``data_list``.
            - False: Disable normalization.
            - None: Let downstream handle defaults.
            - Normalize: Use user-specified object directly.
        vmin (float, optional): Minimum value for normalization.
            Defaults to None.
        vmax (float, optional): Maximum value for normalization.
            Defaults to None.
        title (str): Base title string (e.g., "Expr" or "PCT").
        legend_kws (dict, optional): Legend keyword arguments provided
            by caller. Defaults to {}}.

    Returns:
        tuple[Normalize | None, dict | None]: Normalization object and
        legend keyword arguments ready for use.

    Raises:
        ValueError: If ``data_list`` is empty when norm=True.
    """
    if isinstance(norm, Normalize):
        # User provided custom Normalize to trust it
        return norm, legend_kws

    if norm is True:
        all_vals = np.concatenate([df.values.ravel() for df in data_list])
        vmin = vmin if vmin is not None else all_vals.min()
        vmax = vmax if vmax is not None else all_vals.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        if vmin is not None or vmax is not None:
            label = f"{title} [{'' if vmin is None else f'min={vmin:.2f}'}" \
                    f"{', ' if vmin is not None and vmax is not None else ''}" \
                    f"{'' if vmax is None else f'max={vmax:.2f}'}]"
            legend_kws = {"title": label}
            legend_kws.setdefault("title", label)
        else:
            legend_kws.setdefault("title", title)
        return norm, legend_kws

    if norm is False:
        return None, legend_kws

    # norm=None to let SizedMesh handle defaults
    return None, legend_kws


def _get_legend_flags(
            norm: bool | None | Normalize,
            size_norm: bool | None | Normalize,
            n_segments: int,
            i: int,
        ) -> bool:
    """Handle normalization and legend title logic for dotplot.

    Args:
        data_list (list[pandas.DataFrame]): List of DataFrames with values
            for this feature.
        norm (bool | None | Normalize): Normalization control.
            - True: Create a Normalize from values in ``data_list``.
            - False: Disable normalization.
            - None: Let downstream handle defaults.
            - Normalize: Use user-specified object directly.
        vmin (float, optional): Minimum value for normalization.
            Defaults to None.
        vmax (float, optional): Maximum value for normalization.
            Defaults to None.
        legend_kws (dict, optional): Legend keyword arguments provided
            by caller. Defaults to None.
        title (str): Base title string (e.g. "Expr" or "PCT").

    Returns:
        tuple[Normalize | None, dict | None]: Normalization object and
        legend keyword arguments ready for use.

    Raises:
        ValueError: If ``data_list`` is empty when ``norm`` is True.
    """
    norm_is_active = norm is True or isinstance(norm, Normalize)
    size_norm_is_active = size_norm is True or isinstance(size_norm, Normalize)

    # Case 6: N=1 to at most one legend
    if n_segments == 1:
        return True
    # Case 1: both shared to only first has legends
    if norm_is_active and size_norm_is_active:
        return i == 0
    # Case 2: shared size_norm, no color norm to force shared color norm
    if not norm_is_active and size_norm_is_active:
        return i == 0
    # Case 3: shared color norm, no size norm to force shared size norm
    if norm_is_active and not size_norm_is_active:
        return i == 0
    # Case 4: both False to each gets its own legends
    if norm is False and size_norm is False:
        return True
    # Case 5: both None to defer to defaults
    if norm is None and size_norm is None:
        return True

    return False


def plot_group_dotplot(
            mean_expr: pd.DataFrame | list[pd.DataFrame],
            frac_expr: pd.DataFrame | list[pd.DataFrame],
            cmap: str = "turbo",
            grid: bool = False,
            boundary: bool = True,
            linewidth: float = 1.0,
            linecolor: str = "black",
            dendrogram_left: bool = True,
            dendrogram_top: bool = True,
            save_path: str = "",
            minmax_scale_axis: int | None = None,
            pct_min: float | None = None,
            pct_max: float | None = None,
            norm: bool | None | Normalize = True,
            vmin: float | None = None,
            vmax: float | None = None,
            size_norm: bool | None | Normalize = True,
            sizes: tuple[float, float] = (1, 200),
            smin: float | None = None,
            smax: float | None = None,
            show: bool = True,
            figsize: tuple[float, float] | bool | None = True,
            color_legend_kws: dict = {},
            size_legend_kws: dict = {},
            **kwargs
        ) -> None | list[plt.Axes]:
    """Plot group dotplot (optionally split into halves for two conditions).

    NOTE:
        - dpi should be set via the rc_context
        - Figsize =
    Args:
        mean_expr (pandas.DataFrame): Mean expression per
            cluster × gene (color).
        frac_expr (pandas.DataFrame): Fraction expressed per
            cluster × gene (size).
        mean_expr_2 (pandas.DataFrame | list[pandas.DataFrame] | None):
            Second condition mean expression.
        frac_expr_2 (pandas.DataFrame | list[pandas.DataFrame] | None):
            Second condition fraction expressed.
        cmap (str, optional): Colormap for first condition. Defaults to "turbo".
        grid (bool, optional): Show cell grid lines. Defaults to False.
        boundary (bool, optional): Draw boundary rectangle. Defaults to True.
        linewidth (float, optional): Line width for grid/boundary.
            Defaults to 1.0.
        linecolor (str, optional): Line color for grid/boundary.
            Defaults to "black".
        dendrogram_left (bool, optional): Add left dendrogram. Defaults to True.
        dendrogram_top (bool, optional): Add top dendrogram. Defaults to True.
        save_path (str, optional): File path to save figure. Empty = no save.
        minmax_scale_axis (int | None): Axis for min-max normalization of color.
        pct_min (float | None): Minimum dot size fraction.
        pct_max (float | None): Maximum dot size fraction.
        norm (bool | None | Normalize): Color normalization.
        vmin (float | None): Min value for color normalization.
        vmax (float | None): Max value for color normalization.
        size_norm (bool | None | Normalize): Size normalization.
        sizes (tuple, optional): Min and max dot sizes. Defaults to (1, 200).
        smin (float | ): Min value for size normalization.
        smax (float | None): Max value for size normalization.
        show (bool, optional): Whether to show the plot. Defaults to True.
        figsize (bool | Sequence[float] | None, optional): Figure size control.
            Defaults to True.

                - False: use rcParams["figure.figsize"]
                - True: auto calculate via calc_dotplot_figsize(...)
                - None: use Marsilea default [None, None]
                - Sequence: explicit (width, height)
                - Other: raise AttributeError

        color_legend_kws (dict, optional): Keyword arguments forwarded
            to the color legend (expression legend). Merged internally with
            defaults from `_setup_norm_and_legend`. Defaults to {}.
        size_legend_kws (dict, optional): Keyword arguments forwarded
            to the size legend (fraction legend). Merged internally with
            defaults from `_setup_norm_and_legend`.
        **kwargs: Extra kwargs for SizedMesh. Defaults to {}.

    Returns:
        None | list[matplotlib.axes.Axes]:
            None if show=True, else the Axes.

    Raises:
        AttributeError:
            If figsize has an invalid type or value.
    """
    # ################################################
    # IO checks
    if isinstance(mean_expr, pd.DataFrame):
        mean_expr = [mean_expr]
    elif not isinstance(mean_expr, list):
        raise TypeError("mean_expr must be DataFrame or list of DataFrames.")
    if isinstance(frac_expr, pd.DataFrame):
        frac_expr = [frac_expr]
    elif not isinstance(frac_expr, list):
        raise TypeError("frac_expr must be DataFrame or list of DataFrames.")
    if len(mean_expr) != len(frac_expr):
        # Except for the DEG dotplot the number must be the same
        if len(mean_expr) == 1 and len(frac_expr) == 2:
            mean_expr = [mean_expr[0], mean_expr[0]]
        else:
            raise AttributeError(
                "mean_expr and frac_expr must have equal lengths.")
    if not all(isinstance(x, pd.DataFrame) for x in mean_expr + frac_expr):
        raise TypeError("All items in mean_expr and frac_expr must be DataFrames.")
    if not all(m.shape == mean_expr[0].shape for m in mean_expr + frac_expr):
        raise AttributeError("All matrices must share identical shapes.")
    # ################################################
    # Create the markers, based on the number of input matrices
    n_segments = len(mean_expr)
    markers = make_circle_segment_markers(n_segments)
    # ################################################
    # Min-max scaling for color
    if minmax_scale_axis is not None:
         mean_expr = min_max_scale_axis(
            data_list= mean_expr, axis=minmax_scale_axis)
    # ################################################
    # Clip size fractions
    for i, df in enumerate(frac_expr):
        if pct_min is not None:
            df = np.maximum(df, pct_min)
        if pct_max is not None:
            df = np.minimum(df, pct_max)
        frac_expr[i] = df
    # ################################################
    # Norms
    norm, color_legend_kws = _setup_norm_and_legend(
         mean_expr, norm, vmin, vmax, "Expr", color_legend_kws)
    size_norm, size_legend_kws = _setup_norm_and_legend(
        frac_expr, size_norm, smin, smax, "PCT", size_legend_kws)
    # ################################################
    # Figsize handling
    if figsize is False:
        # figsize handling from rc_context
        figsize = plt.rcParams["figure.figsize"]
    elif figsize is True:
        # ###########################
        # The calculation of the sizes for the figure
        figsize = calc_dotplot_figsize(
                shape=mean_expr[0].shape,
                xlabels=mean_expr[0].columns.tolist(),
                ylabels=mean_expr[0].index.tolist())
    elif figsize is None:
        # The marsilea default will be used.
        figsize = [None, None]
    elif isinstance(figsize, Sequence) or figsize is None:
        # If actual figsize is parsed
        pass
    else:
        raise AttributeError("Please check the figsize arg!")
    # ################################################
    # Base heatmap
    h = Heatmap(
         mean_expr[0].values, vmin=0, vmax=0,
        cmap="Greys", legend=False,
        linewidth=linewidth if grid else 0,
        linecolor=linecolor if grid else None,
        width=figsize[0], height=figsize[1])
    # ################################################
    # Dendrogram handling with precomputed linkage
    if dendrogram_left and df.shape[0] > 1:
        _, row_linkage = calc_hierarchical_order(
            [df.values for df in  mean_expr],
            axis=0, return_linkage=True)
        h.add_dendrogram("left", linkage=row_linkage)
    if dendrogram_top and df.shape[1] > 1:
        _, col_linkage = calc_hierarchical_order(
            [df.values for df in  mean_expr],
            axis=1, return_linkage=True)
        h.add_dendrogram("top", linkage=col_linkage)
    # ################################################
    # Dots
    for i, (color, size, marker) in enumerate(
            zip(mean_expr, frac_expr, markers)):
        layer_kwargs = dict(kwargs)

        if n_segments == 1:
            # single to allow user marker, fallback "o"
            if "marker" not in layer_kwargs:
                layer_kwargs["marker"] = "o"
        else:
            # multi to force our generated marker
            layer_kwargs["marker"] = marker

        dots = SizedMesh(
            color=color.values,
            size=size.values,
            cmap=cmap,
            norm=norm,
            sizes=sizes,
            size_norm=size_norm,  # TODO: Check when marsilea fixed the bug!
            color_legend_kws=color_legend_kws,
            size_legend_kws=size_legend_kws,
            legend=_get_legend_flags(norm, size_norm, n_segments, i),
            # size_legend_kws={"title": "Point Size"},
            # color_legend_kws={"title": "Color Scale"},
            **layer_kwargs)
        h.add_layer(dots)
    # ################################################
    # Add the legend
    h.add_legends(box_padding=1.2, stack_size=2)
    # ################################################
    # Add labels
    h.add_left(Labels(mean_expr[0].index.tolist()))
    h.add_bottom(Labels(mean_expr[0].columns.tolist(), rotation=90))
    # h.add_left_title(y_label)   # y-axis label
    # h.add_bottom_title(x_label)    # x-axis label
    # ################################################
    # Add a boundary
    h.render()
    if boundary:
        ax = h.get_main_ax()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ptp_x = xmax - xmin
        ptp_y = ymax - ymin
        offset = .00
        xmin = xmin - (ptp_x * offset)
        xmax = xmax + (ptp_x * offset)
        ymin = ymin - (ptp_y * offset)
        ymax = ymax + (ptp_y * offset)
        rect = mpl_rectangle(
            (xmin, ymin),
            xmax - xmin, ymax - ymin,
            fill=False, color=linecolor, linewidth=linewidth)
        ax.add_patch(rect)
    # ################################################
    # Save/return/show the plot
    fig = plt.gcf()
    axes = fig.get_axes()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return axes


def plot_dotplot(
            adata: ad.AnnData,
            keys: list[str],
            groupby: str,
            condition_obs_key: str | None = None,
            vs: tuple[str, str] | None = None,
            expression_cutoff: float = 0.0,
            title: str | None = None,
            obsm_keys: list[str] | None = None,
            varm_keys: list[str] | None = None,
            plot_group_dotplot_kwargs: dict = {},
            extract_kwargs: dict = {},
            save_path: str = "",
            show: bool = True,
            figsize: tuple[float, float] | bool | None = True,
        ) -> None | list[plt.Axes]:
    """Wrapper to compute dotplot data and call plot_group_dotplot.

    NOTE:
        - dpi should be set via the rc_context
        - figsize CAN bes set via the rc_context with figsize = False.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        keys (list[str]): Genes to visualize.
        groupby (str): Grouping column in obs.
        condition_obs_key (str | None, optional): Optional condition column in obs.
            Defaults to None.
        vs (tuple[str, str] | None, optional): Optional pair of conditions to select.
            Defaults to None.
        expression_cutoff (float, optional): Expression threshold.
            Defaults to 0.0.
        title (str, optional): Plot title. Defaults to None.
        obsm_keys (list[str] | None, optional): Keys from obsm. Defaults to None.
        varm_keys (list[str] | None, optional): Keys from varm. Defaults to None.
        plot_group_dotplot_kwargs (dict, optional): Extra kwargs forwarded to
            plot_group_dotplot. Defaults to {}.
        extract_kwargs (dict, optional): Extra kwargs forwarded to
            get_adata_sub_keys. Defaults to {}.
        save_path (str, optional): Path to save figure.
            Defaults to "".
        show (bool, optional): Whether to show the plot. Defaults to True.
        figsize (bool | Sequence[float] | None, optional): Figure size control.
            Defaults to True.

                - False: use rcParams["figure.figsize"]
                - True: auto calculate via calc_dotplot_figsize(...)
                - None: use Marsilea default [None, None]
                - Sequence: explicit (width, height)
                - Other: raise AttributeError

    Returns:
        None | list[matplotlib.axes.Axes]: None if show=True, else the Axes.
    """
    # #########################################################
    plot_group_dotplot_kwargs |= {
        "save_path": save_path, "show": show, "figsize": figsize}
    # #########################################################
    # Validate the args
    validate_groupby_column(adata.obs, groupby, check_categorical=True)
    if not keys:
        raise ValueError("The keys list is empty.")
    # #########################################################
    # Handle groupby and condition
    if condition_obs_key is None:
        # #####################################
        # single condition
        df_expr = get_adata_sub_keys(
            adata, keys, obsm_keys, varm_keys,
            groupby=[groupby],
            **extract_kwargs)
        frac = df_expr.groupby(groupby, observed=True)[keys].apply(
            lambda x: (x > expression_cutoff).sum() / len(x))
        mean = df_expr.groupby(groupby, observed=True)[keys].apply(
            lambda x: x.mask(x <= expression_cutoff).mean())
        if title is None:
            title = groupby
        return plot_group_dotplot(mean, frac, **plot_group_dotplot_kwargs)
    else:
        # #####################################
        # multiple conditions
        validate_groupby_column(adata.obs, condition_obs_key, check_categorical=True)
        conds = list(adata.obs[condition_obs_key].cat.categories)
        if vs is not None:
            if not isinstance(vs, (tuple, list)) or len(vs) != 2:
                raise ValueError("'vs' must be a tuple of exactly two condition labels.")
            conds = list(vs)
        if title is None:
            title = " vs ".join(conds)
        df_expr = get_adata_sub_keys(
            adata, keys, obsm_keys, varm_keys,
            groupby=[condition_obs_key, groupby],
            **extract_kwargs)
        mean_list, frac_list = [], []
        for cond in conds:
            df_c = df_expr[df_expr[condition_obs_key] == cond].copy()
            frac = df_c.groupby(groupby, observed=False)[keys].apply(
                lambda x: (x > expression_cutoff).sum() / len(x))
            mean = df_c.groupby(groupby, observed=False)[keys].apply(
                lambda x: x.mask(x <= expression_cutoff).mean())
            frac.fillna(0, inplace=True)
            mean.fillna(0, inplace=True)
            frac_list.append(frac)
            mean_list.append(mean)
        return plot_group_dotplot(
            mean_list,
            frac_list,
            **plot_group_dotplot_kwargs)


def plot_DEG_dotplot(
            adata: ad.AnnData,
            groupby: str = None,
            condition_obs_key: str | None = None,
            deg_key: str | None = None,
            deg_df: pd.DataFrame | None = None,
            plot_group_dotplot_kwargs: dict = {},
            subset_kwargs: dict = {"n_genes": 5},
            save_path: str = "",
            show: bool = True,
            figsize: tuple[float, float] | bool | None = True,
        ) -> None | list[plt.Axes]:
    """Plot DEG dotplot with log fold-change as color and group/ref pct as sizes.

    NOTE:
        - The default plot uses "rank_genes_groups" in case of
          condition_obs_key=None, else
          f"{condition_obs_key}_per_{groupby}_rank_genes_groups"
          as produced by calc_DEGs_multi_group()
        - For the automated usage of condition_obs_key you need to run
          sc_utils.calc_DEGs_multi_group first!
        - dpi should be set via the rc_context

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        groupby (str | None, optional): Column in obs specifying clusters.
            Defaults to config["general"]["cluster_algorithm"].
        condition_obs_key (str | None, optional): Column in obs specifying
            conditions. Defaults to None.
        deg_key (str | None, optional): Key in adata.uns with DEG results.
            Defaults to None.
        deg_df (pd.DataFrame | None, optional): Precomputed DEG DataFrame.
            Defaults to None.
        plot_group_dotplot_kwargs (dict, optional): Extra kwargs
            forwarded to plot_group_dotplot. Defaults to None.
        subset_kwargs (dict, optional): Kwargs parsed to subset_degs.
            If you want no subsetting at all, parse an empty dict.
            Defaults to {"n_genes": 5}.
        save_path (str, optional): Path to save figure. Empty = no save.
            Defaults to "".
        show (bool, optional): Whether to show the plot. Defaults to True.
        figsize (bool | Sequence[float] | None, optional): Figure size control.
            Defaults to True.

                - False: use rcParams["figure.figsize"]
                - True: auto calculate via calc_dotplot_figsize(...)
                - None: use Marsilea default [None, None]
                - Sequence: explicit (width, height)
                - Other: raise AttributeError

    Returns:
        None | list[matplotlib.axes.Axes]: None if show=True, else the Axes.
    """
    # #########################################################
    # Inject save/show/figsize into kwargs
    plot_group_dotplot_kwargs |= {
        "save_path": save_path, "show": show, "figsize": figsize}
    # ####################################################
    # groupby and condition_obs_key validity check
    # get the clustering algorithm key
    if groupby is None:
        # Try to get the default clustering key
        group_name = (
                adata.uns
                .get("config", {})
                .get("general", {})
                .get("cluster_algorithm"))
        groupby = group_name
    # Validate groupby and condition_obs_key
    validate_groupby_column(
        adata.obs, groupby=groupby, check_categorical=True)
    if condition_obs_key is not None:
        # Check if groupby is properly setup
        validate_groupby_column(
            adata.obs, condition_obs_key, check_categorical=False)
    # ####################################################
    # Get the deg_df
    is_default_group_cond = False
    if deg_df is None:
        if deg_key is None:
            if condition_obs_key is None:
                deg_key = "rank_genes_groups"
            else:
                deg_key = f"{condition_obs_key}_per_{groupby}_rank_genes_groups"
                is_default_group_cond = True
        if deg_key not in adata.uns:
            if is_default_group_cond:
                raise KeyError(
                    f"DEG key '{deg_key}' not found in adata.uns run "
                    "sc_utils.calc_DEGs_multi_group(.., condition_obs_key=...) "
                    "first.")
            else:
                raise KeyError(f"DEG key '{deg_key}' not found in adata.uns")
        deg_df = adata.uns[deg_key]
    if not isinstance(deg_df, pd.DataFrame):
        raise TypeError("deg_df must be a pandas.DataFrame")
    # #########################################################
    # Subset deg_df and check required columns
    deg_df = subset_degs(deg_df, **subset_kwargs)  # Returns a copy

    required_cols = {
        "group",
        "reference",
        "names",
        "logfoldchanges",
        "pct_nz_group",
        "pct_nz_reference",
    }
    if not required_cols.issubset(deg_df.columns):
        raise ValueError(f"deg_df must contain columns {required_cols}")

    deg_df["comparison"] = deg_df["group"].astype(str) + "_vs_" + deg_df["reference"].astype(str)
    # #########################################################
    # pivot into comparison × gene matrices
    lfc = deg_df.pivot(index="comparison", columns="names",
                       values="logfoldchanges").fillna(0)
    pct_group = deg_df.pivot(index="comparison", columns="names",
                             values="pct_nz_group").fillna(0)
    pct_ref = deg_df.pivot(index="comparison", columns="names",
                           values="pct_nz_reference").fillna(0)
    # #########################################################
    # Ensure colormap centered at 0 using actual min/max
    lfc_min = lfc.min().min()
    lfc_max = lfc.max().max()
    if lfc_min >= 0:
        lfc_min = -.1
    if lfc_max <= 0:
        lfc_max = .1
    plot_group_dotplot_kwargs["norm"] = TwoSlopeNorm(
        vmin=lfc_min, vcenter=0, vmax=lfc_max)
    # #########################################################
    # return or plot
    return plot_group_dotplot(
        mean_expr=lfc,
        frac_expr=[pct_group, pct_ref],
        **plot_group_dotplot_kwargs)


# ###################################################################################################
# multi comparison plots
def plot_umap_sbs(
            adata: ad.AnnData,
            genes: list[str],
            groupby: str,
            groups: list[str | int] = None,
            obsm_keys: str | list | None = None,
            base_figsize: Sequence[float | int] = (4, 4),
            fix_cmap: bool = True,
            extract_kwargs: dict = {},
            fill_background_alpha: None | float = None,
            save_path: str = "",
            show: bool = True,
            **kwargs,
        ) -> None:
    """Plot a grid of UMAPs showing gene expression across metadata groups.

    This function generates a subplot grid of UMAP plots. Each row corresponds
    to a gene, and each column corresponds to a specific group value from
    ``adata.obs[groupby]``. It supports using ``.obsm`` keys via
    ``get_adata_sub_keys``.

    NOTE:
        The size parameter for sc.pl.umap is determined automatically by
        scanpy, if you use the sub_ids you may need to set the ``size`` parameter,
        otherwise, the plots will have different sized spots.

    Args:
        adata (anndata.AnnData): Adata object containing UMAP coordinates.
        genes (list[str]):
            List of gene names to plot (must be in ``adata.var_names``).
        groupby (str):
            Name of ``adata.obs`` column to group by (e.g., "condition").
        groups (list[str | int], optional):
            Subset to the groups in the groupby (must exist in
            ``adata.obs[groupby]``). Defaults to None.
        obsm_keys (str | list | None, optional): Name of ``.obsm`` key to extract
            gene values from. X). Defaults to None.
        base_figsize (Sequence[float | int], optional):
            Base size for each subplot as (width, height). Defaults to (4, 4).
        fix_cmap (bool):
            If True, uses the same color scale for each gene across all
            ``groups``.
        extract_kwargs (dict, optional): Additional keyword arguments forwarded
            to ``sc_utils.get_adata_sub_keys``.
        fill_background_alpha (dict, optional): plot the background cells gray
            with alpha. Defaults to None
        save_path (str, optional): Save the figure. Defaults to ""
        show (bool, optional): If True executes plt.show(). Defaults to True.
        **kwargs:
            Additional keyword arguments passed to ``sc_plots.plot_umap``.

    Returns:
        None

    Raises:
        ValueError: If genes or group values are not present in the data.

    Calls:
        get_adata_sub_keys, plot_umap, validate_groupby_column
    """
    # #########################################################
    # Check if groupby and condition_obs_key is properly setup
    if groupby is not None:
        validate_groupby_column(
            adata.obs, groupby, check_categorical=True, groups=groups)

    # TODO: Add more IO checks
    # ##############################################
    # kwarg checks
    # Remove user-provided 'ax' because it doesn't make sense here and we use them
    kwargs.pop("ax", None)
    # ##############################################
    # Extract the data
    df = get_adata_sub_keys(
        adata,
        keys=genes,
        obsm_keys=obsm_keys,
        groupby=groupby,
        var_keys_only=False,
        **extract_kwargs,)

    if groups is None:
        groups = df[groupby].unique().tolist()

    missing_groups = [g for g in groups if g not in df[groupby].unique()]

    if missing_groups:
        raise ValueError(f"Groups not found in adata.obs['{groupby}']: {missing_groups}")
    # ##############################################
    # Determin figsize
    n_genes = len(genes)
    n_groups = len(groups)

    figsize = (base_figsize[0] * n_groups, base_figsize[1] * n_genes)
    _, axes = plt.subplots(nrows=n_genes, ncols=n_groups, figsize=figsize)
    # Determine the axis and fix the plt bug of one ax object to list
    if n_genes == 1:
        axes = [axes]
    if n_groups == 1:
        axes = [[a] for a in axes]
    # ##############################################
    # Fix genes to intersection with df columns (excluding groupby column)
    valid_genes = list(set(genes).intersection(df.columns) - {groupby})

    if isinstance(fill_background_alpha, float):
        adata.obs["123_ranodm_dummy_123"] = ""

    for i, gene in enumerate(valid_genes):
        # Compute vmin/vmax if needed
        vmin, vmax = None, None
        if fix_cmap:
            v = df.loc[df[groupby].isin(groups), gene].values
            vmin = np.min(v)
            vmax = np.max(v)

        for j, group in enumerate(groups):
            sub_ids = df.index[df[groupby] == group].tolist()

            ax = axes[i][j]
            kwargs_local = kwargs.copy()
            if fix_cmap:
                kwargs_local["vmin"] = vmin
                kwargs_local["vmax"] = vmax
            if isinstance(fill_background_alpha, float):
                plot_umap(
                    adata,
                    keys="123_ranodm_dummy_123",
                    alpha=fill_background_alpha,
                    ax=ax,
                    show=False,
                    legend_loc=None,)
            plot_umap(
                adata,
                keys=gene,
                ax=ax,
                show=False,
                sub_ids=sub_ids,
                obsm_keys=obsm_keys,
                **kwargs_local,)
            ax.set_title(f"{gene} - {groupby}={group}")

    plt.tight_layout()

    if isinstance(fill_background_alpha, float):
        del adata.obs["123_ranodm_dummy_123"]

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        return axes


def plot_density_difference(
            adata: ad.AnnData,
            density_uns_key: str,
            basis: str = 'X_umap',
            group_ref: list[tuple[str, str]] | None = None,
            group: str | None = None,
            reference: str | None = None,
            cmap: str = 'turbo',
            ncols: int | None = None,
            base_figsize: Sequence[float | int] = (5, 5),
            save_path: str | None = None,
            dpi: int = 100,
            levels: int = 20,
            clip_range: tuple[float, float] | None = (-1.1, 1.1),
            lower_thresh: float | None = None,
            mask_thresh: float = 1e-3
        ) -> None:
    """Plot density difference maps from precomputed densities in ``adata.uns``.

    Args:
        adata (anndata.AnnData): Adata object with densities in ``.uns``.
        density_uns_key (str): Key in ``adata.uns`` for the density dictionary.
        basis (str, optional): Embedding basis used for the KDE (e.g.,
            'X_umap'). Defaults to 'X_umap'.
        group_ref (list[tuple[str, optional): list of (group, ref) pairs.
        group (str | None, optional): Group name if only one comparison is done.
            Defaults to None.
        ref (str, optional): Reference name if only one comparison is done.
        cmap (str, optional): Colormap to use for the contour difference plot.
            Defaults to 'turbo'.
        ncols (int | None, optional): Number of subplot columns.
            Defaults to None.
        base_figsize (Sequence[float | int], optional): Base figure size for
            each subplot. Defaults to (5, 5).
        save_path (str, optional): Path to save figure.
        dpi (int): Dots per inch for the figure.
        levels (int): Number of contour levels.
        clip_range (tuple[float, float], optional): Range to clip the difference
            values.
        lower_thresh (float): Threshold to binarize densities (0 below, 1 above
            or equal).
        mask_thresh (float): Threshold to mask areas with low density. Defaults
            to 1e-3.

    Returns:
        None

    Raises:
        ValueError: If comparison groups are missing.
        KeyError: If requested groups are not in the stored densities.

    Calls:
        get_rows_cols_figsize
    """
    if group_ref is None:
        if group is None or reference is None:
            raise ValueError("Either provide ``group_ref`` or both ``group`` and ``reference``.")
        group_ref = [(group, reference)]

    densities = adata.uns[density_uns_key]
    all_keys = set(densities.keys())

    missing = [x for pair in group_ref for x in pair if x not in all_keys]
    if missing:
        raise KeyError(f"Missing groups in density data: {missing}")

    n_plots = len(group_ref)
    ncols, nrows, figsize = get_rows_cols_figsize(
        n_categories=n_plots,
        ncols=ncols,
        base_figsize=base_figsize,)

    # Infer shape from one density matrix
    grid_size = next(iter(densities.values())).shape[0]
    x_min, x_max = np.min(adata.obsm[basis][:, 0]), np.max(adata.obsm[basis][:, 0])
    y_min, y_max = np.min(adata.obsm[basis][:, 1]), np.max(adata.obsm[basis][:, 1])
    x_dist = np.ptp([x_min, x_max])
    y_dist = np.ptp([y_min, y_max])
    x_min, x_max = x_min - 0.05 * x_dist, x_max + 0.05 * x_dist
    y_min, y_max = y_min - 0.05 * y_dist, y_max + 0.05 * y_dist

    xx, yy = np.mgrid[
        x_min:x_max:grid_size * 1j,
        y_min:y_max:grid_size * 1j]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(-1)

    if lower_thresh is not None:
        # Custom cmap
        custom_cmap = ListedColormap([
            to_rgba('#00ff00', alpha=0.5),  # green, center, full opacity
            to_rgba('#ff00ff', alpha=1.),  # purple, low value, semi-transparent
            to_rgba('#ff8000', alpha=0.5),  # orange, high value, semi-transparent
        ])

        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, custom_cmap.N)
    else:
        custom_cmap = cmap
        norm = None

    for idx, (grp, ref) in enumerate(group_ref):
        ax = axes[idx]

        # Binarize the density for clean view
        if lower_thresh is not None:
            grp_dens = (densities[grp] >= lower_thresh).astype(float)
            ref_dens = (densities[ref] >= lower_thresh).astype(float) * -1
            diff = grp_dens + ref_dens
        else:
            diff = densities[grp] - densities[ref]

        # Optional: mask regions where both group and reference densities are near-zero
        mask = (densities[grp] < mask_thresh) & (densities[ref] < mask_thresh)
        diff[mask] = np.nan

        if clip_range:
            diff = np.clip(diff, clip_range[0], clip_range[1])

        cs = ax.contourf(xx, yy, diff, levels=levels if norm is None else bounds, cmap=custom_cmap, norm=norm)
        ax.set_title(f'{grp} vs {ref}', fontsize=12)
        ax.set_xlabel(f'{basis} 1')
        ax.set_ylabel(f'{basis} 2')
        if basis == "spatial":
            ax.invert_yaxis()
        fig.colorbar(cs, ax=ax, label='Density difference', shrink=0.8)

    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ###################################################################################################
# UMAP Helpers for sc_code.run_downstream
def discrete_umap_helper(
            adata: ad.AnnData,
            keys: list[str],
            part: str,
            with_figure_explaination: bool,
            name: str = "",
            highlighers: str = "#" * 80,
            config: dict | None = None
        ) -> None:
    """Helper function to create a discretized UMAP plot for specified keys.

    This function generates a UMAP plot for a set of marker genes within an
    adata object, with the option to log an explanation of the figure being
    saved. The function allows for a custom name in the filename and
    configuration options for the UMAP plotting parameters.

    NOTE:
        This is more a helper for run_downstream.

    Args:
        adata (anndata.AnnData): Adata object.
        keys (list[str]): List of marker genes to be visualized on the UMAP.
        part (str): Directory part of the analysis where this should be saved.
        with_figure_explaination (bool): If True, logs an explanation of the
            figure being saved.
        name (str, optional): Name to be included in the saved file name.
            Defaults to "".
        highlighers (str, optional): Highlighter string for visual emphasis in
            logs. Defaults to "#" * 80.
        config (dict | None, optional): Configuration dictionary. uns["config"]``
            if not provided. Defaults to None.

    Returns:
        None:
            The function directly generates and saves a UMAP plot.

    Calls:
        get_colormap, get_save_path, plot_umap

    Called By:
        plot_per_group_DEG_umaps, run_downstream
    """
    # #########################################################
    # Check if a configuration dictionary is provided, else use the default from ``adata``
    if config is None:
        config = adata.uns["config"]

    # Construct the save path using the provided name and configuration settings
    save_path = get_save_path(f'_{name}_discretized.pdf', config, part)
    # #########################################################
    # Log explanation of the figure being saved if requested
    if with_figure_explaination:
        # ##########################################
        # Log the highlighter string for emphasis
        logger.info(highlighers)
        # Log a description of the UMAP plot being saved
        logger.info(f'This Shows {name} Genes Discretized for each cell over the UMAP')
        # Log the path where the UMAP plot is saved
        logger.info(f'It is saved as: umap{save_path}')
        # Log the highlighter string again for visual emphasis
        logger.info(highlighers)
    # #########################################################
    # Generate and save the UMAP plot with discretized values
    # plot_umap(
    sc.pl.umap(
        adata, color=keys,
        cmap=get_colormap(cmap=config["pl"]["umap"]["cmap"], fade_alpha=True),
        save=save_path, vmin=0, vmax=1, layer="counts",
        **{k: v for k, v in config["pl"]["umap"].items() if k not in ["cmap", "vmin", "vmax", "layer"]})


def continuos_umap_helper(
            adata: ad.AnnData,
            keys: list[str],
            part: str,
            with_figure_explaination: bool,
            name: str = "",
            highlighers: str = "#" * 80,
            config: dict | None = None
        ) -> None:
    """Helper function to create a continuous UMAP plot for specified keys.

    This function generates and saves a UMAP plot visualizing the gradients of
    specified marker genes across cells in the adata. It can optionally log a
    textual explanation of the figure being saved and supports customization via
    a configuration dictionary.

    NOTE:
        This is more a helper for run_downstream.

    Args:
        adata (anndata.AnnData): Adata object.
        keys (list[str]): List of marker genes to be visualized on the UMAP.
        part (str): Directory part of the analysis where this should be saved.
        with_figure_explaination (bool): If True, logs an explanation of the
            figure being saved.
        name (str, optional): Name to be included in the saved file name.
            Defaults to "".
        highlighers (str, optional): Highlighter string for visual emphasis in
            logs. Defaults to 80 "#" characters. Defaults to "#" * 80.
        config (dict | None, optional): Configuration dictionary. uns["config"]``
            if not provided. Defaults to None.

    Returns:
        None:
            The function directly generates and saves a UMAP plot.

    Calls:
        get_colormap, get_save_path, plot_umap

    Called By:
        plot_per_group_DEG_umaps, run_downstream
    """
    # #########################################################
    # Setup configuration and save path
    if config is None:
        config = adata.uns["config"]

    save_path = get_save_path(f'_{name}_continuos.pdf', config, part)
    # #########################################################
    # Log figure explanation if specified
    if with_figure_explaination:
        # ##########################################
        # Log the start of the explanation
        logger.info(highlighers)
        logger.info(f'This Shows {name} Genes gradients for each cell over the UMAP')
        # Log the save path of the figure
        logger.info(f'It is saved as: umap{save_path}')
        logger.info(highlighers)
    # #########################################################
    # Generate and save the UMAP plot
    # plot_umap(
    sc.pl.umap(
        adata, color=keys,
        cmap=get_colormap(cmap=config["pl"]["umap"]["cmap"], fade_alpha=True),
        save=save_path,
        **{k: v for k, v in config["pl"]["umap"].items() if k not in ["cmap"]})


def plot_ref_dotplots(
            adata: ad.AnnData,
            config: dict | None = None,
            part: str = "/downstream/",
            with_figure_explaination: bool = True
        ) -> None:
    """
    Create dotplots for marker genes per cluster and/or subset of ref/markers.

    This function generates dotplots for selected marker genes within clusters
    or subsets of ref/markers based on the provided configuration. The function
    can save the generated plots and optionally log their locations.

    NOTE:
        This is more a helper for run_downstream use the plot_dotplot for
        regular plots.

    Args:
        adata (anndata.AnnData): Adata object.
        config (dict | None, optional): Configuration dictionary with parameters
            for generating dotplots. If not provided, it uns["config"]``.
            Defaults to None.
        part (str, optional): Path to the directory where plots will be saved.
            Default is "/downstream/". Defaults to "/downstream/".
        with_figure_explaination (bool, optional): Whether to log the file path
            where the figure is saved. Default is True. Defaults to True.

    Returns:
        None:
            This function saves or displays dotplots but returns no object.

    Raises:
        KeyError: If required keys are missing in the configuration dictionary.
        ValueError: If the data or configuration settings are invalid.

    Calls:
        get_colormap, get_save_path, ref_dict_long_value_split

    Called By:
        run_downstream
    """
    # #########################################################
    # Handle the configuration setup
    if config is None:
        config = adata.uns["config"]
    # #########################################################
    # Generate dendrogram for clustering
    sc.tl.dendrogram(adata, config["general"]["cluster_algorithm"],
                     **config["tl"]["dendrogram"])
    # #########################################################
    # dotplots for RNA marker subsets per cluster
    if (
            len(config["to_plot"]["down"]["marker_subset"]) > 0
            and "rna" in config["to_plot"]["down"]["cluster_dotplot"]["marker_subs_mod"]):
        if "rna" in config["to_plot"]["down"]["cluster_dotplot"]["marker_subs_mod_per_cluster"]:
            # ##########################################
            # Loop through clusters and generate dotplots
            for c in adata.uns["ref_dict_upd_subs"].keys():
                save_path = get_save_path(
                        f'rna_cluster_marker_expression_selected_only_group_{c}.pdf', config, part)
                if with_figure_explaination:
                    logger.info(f'The figure is saved as {save_path}')
                # #####################
                # The plot can't handle more than 350 genes, use the first 350 for now
                c_dict = ref_dict_long_value_split(
                        {k: v for k, v in adata.uns["ref_dict_upd_subs"].items() if k == c})
                # #####################
                for k, v in c_dict.items():
                    # TODO: use sc_plot.dotplot!
                    sc.pl.dotplot(
                            adata, {k: v}, config["general"]["cluster_algorithm"],
                            cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]), save=save_path,
                            **{k: v for k, v in config["pl"]["dotplot"].items() if k not in ["cmap"]})
        else:
            # ##########################################
            # Generate a single dotplot for all clusters
            save_path = get_save_path(
                    "rna_cluster_marker_expression_selected_only.pdf", config, part)
            if with_figure_explaination:
                logger.info(f'The figure is saved as {save_path}')
            # #####################
            # The plot can't handle more than 350 genes, use the first 350 for now
            c_dict = ref_dict_long_value_split(adata.uns["ref_dict_upd_subs"].items())
            # #####################
            for k, v in c_dict.items():
                # TODO: use sc_plot.dotplot!
                sc.pl.dotplot(
                    adata, {k: v}, config["general"]["cluster_algorithm"],
                    cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]), save=save_path,
                    **{k: v for k, v in config["pl"]["dotplot"].items() if k not in ["cmap"]})
    # #########################################################
    # Generate dotplots for RNA marker modules per cluster
    if "rna" in config["to_plot"]["down"]["cluster_dotplot"]["marker_mod"]:
        if "rna" in config["to_plot"]["down"]["cluster_dotplot"]["marker_mod_per_cluster"]:
            # ##########################################
            # Step 4a: Loop through clusters and generate dotplots for marker modules
            for c in adata.uns["ref_dict_upd"].keys():
                save_path = get_save_path(
                        f'rna_cluster_marker_expression_group_{c}.pdf', config, part)
                if with_figure_explaination:
                    logger.info(f'The figure is saved as {save_path}')
                # #####################
                # The plot can't handle more than 350 genes, use the first 350 for now
                c_dict = ref_dict_long_value_split({k: v for k, v in adata.uns["ref_dict_upd"].items() if k == c})
                # #####################
                for k, v in c_dict.items():
                    # TODO: use sc_plot.dotplot!
                    sc.pl.dotplot(
                        adata, {k: v}, config["general"]["cluster_algorithm"],
                        cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]), save=save_path,
                        **{k: v for k, v in config["pl"]["dotplot"].items() if k not in ["cmap"]})
        else:
            # ##########################################
            # Generate a single dotplot for all marker modules across clusters
            save_path = get_save_path(
                    "rna_cluster_marker_expression.pdf", config, part)
            if with_figure_explaination:
                logger.info(f'The figure is saved as {save_path}')
            # #####################
            # The plot can't handle more than 350 genes, use the first 350 for now
            c_dict = ref_dict_long_value_split(adata.uns["ref_dict_upd"])
            # #####################
            for k, v in c_dict.items():
                # TODO: use sc_plot.dotplot!
                sc.pl.dotplot(
                    adata, {k: v}, config["general"]["cluster_algorithm"],
                    dendrogram=True, cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]),
                    standard_scale="var", save=save_path,  # use_raw=config["use_raw"], TODO: AHH WHAT IS GOING ON
                    swap_axes=config["to_plot"]["down"]["cluster_dotplot"]["swap_axes"])


def plot_ref_stacked_violins(
            adata: ad.AnnData,
            config: dict | None = None,
            part: str = "/downstream/",
            with_figure_explaination: bool = True
        ) -> None:
    """
    Generates and saves stacked violin plots for marker genes per cluster
    and/or a subset of ref/markers.

    This function creates and saves stacked violin plots using the ``scanpy``
    library, based on the specified configuration for RNA data within an adata
    object. The plots are grouped by clusters or other specified groupings and
    can be tailored to either a full set of ref/markers or a subset.

    NOTE:
        This is more a helper for run_downstream.

    Args:
        adata (anndata.AnnData): Adata object.
        config (dict | None, optional): Configuration for marker plotting.
            uns["config"]``. Defaults to None.
        part (str, optional): Directory path to save the plots.
            Defaults to "/downstream/".
        with_figure_explaination (bool, optional): Whether to log saved figure
            paths. Defaults to True.

    Returns:
        None:
            The function saves the plot(s) and does not return any value.

    Raises:
        KeyError: If the required keys are missing in the ``config`` or
            ``adata.uns`` dictionaries.
        ValueError: If the data or ref/markers specified in the config are not
            compatible or missing.

    Calls:
        get_colormap, get_save_path

    Called By:
        run_downstream
    """
    # #########################################################
    # Initialize the configuration if not provided
    if config is None:
        config = adata.uns["config"]
    # #########################################################
    # Generate and save stacked violin plots for a subset of ref if specified
    if (
            len(config["to_plot"]["down"]["marker_subset"]) > 0
            and "rna" in config["to_plot"]["down"]["cluster_violins"]["marker_subs_mod"]):
        if "ref_dict_upd" in adata.uns.keys():
            for c, ref in adata.uns["ref_dict_upd_subs"].items():
                save_path = get_save_path(
                        f'rna_cluster_marker_expression_selected_only_group_{c}.pdf', config, part)
                if with_figure_explaination:
                    logger.info(f'The figure is saved as {save_path}')
                # ##########################################
                # Plot and save the violin plot for the current group of ref
                sc.pl.stacked_violin(
                    adata, ref, groupby=config["general"]["cluster_algorithm"],
                    title=f'stacked_violin_rna_cluster_marker_selected_group_{c}',
                    use_raw=False, save=save_path,
                    cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]))
                # ##########################################
        else:
            # TODO: write correct error and add the raise to the docstring
            raise AttributeError('You set the config["to_plot"]["down"]["cluster_violins"]["marker_mod"] sub...')
    # #########################################################
    # Generate and save stacked violin plots for all ref if specified
    if "rna" in config["to_plot"]["down"]["cluster_violins"]["marker_mod"]:
        if "ref_dict_upd" in adata.uns.keys():
            # print(part)
            for c in adata.uns["ref_dict_upd"].keys():
                save_path = get_save_path(
                        f'stacked_violin_rna_cluster_marker_expression_group_{c}.pdf', config, part)
                if with_figure_explaination:
                    logger.info(f'The figure is saved as {save_path}')
                # ##########################################
                # Plot and save the violin plot for the current group of ref
                sc.pl.stacked_violin(
                    adata, adata.uns["ref_dict_upd"][c], groupby="leiden",
                    title=f'stacked_violin_rna_cluster_marker_expression_group_{c}',
                    use_raw=False, save=save_path,
                    cmap=get_colormap(cmap=config["pl"]["dotplot"]["cmap"]))
                # break
                # ##########################################
        else:
            # TODO: write correct error and add the raise to the docstring
            raise AttributeError('You set the config["to_plot"]["down"]["cluster_violins"]["marker_mod"] sub...')


# ###########################################################################################################
# QC
def plot_filtering(
            adata: ad.AnnData,
            before: bool = True,
            prot: bool = False,
            part: str = "/qc/"
        ) -> None:
    """Plots the main QC steps in violins and scatterplots.

    This function generates visualizations of quality control (QC) steps for
    single-cell RNA sequencing data. It provides violin plots and scatterplots
    that illustrate various QC metrics, such as the number of genes, counts, and
    other quality scores. The function allows customization of whether the plots
    are generated before or after filtration and supports data-specific
    adjustments, such as for Citeseq data.

    Args:
        adata (anndata.AnnData): Adata object.
        before (bool, optional): Indicates whether the plotting is done before
            or after filtration. Defaults to True.
        prot (bool, optional): Specifies whether the data is Citeseq data.
            Defaults to False.
        part (str, optional): Specifies the part of the analysis for which the
            plotting is intended, which determines the save directory. Defaults
            to "/qc/".

    Returns:
        None:
            The function generates plots and does not return any values.

    Raises:
        None: This function does not raise any exceptions.

    Calls:
        plot_qc_scatter, plot_qc_scatter_combinations, plot_qc_violins
    """
    # #########################################################
    # Exit the function if only analysis is required
    if adata.uns["config"]["general"]["analysis_only"]:
        return
    # #########################################################
    # Set the key to indicate whether plotting is done before or after filtering
    if before:
        b = "before"
    else:
        b = "after"
    logger.info(f"Plots {b} filtering")
    # #########################################################
    # Generate violin plots for the number of genes, counts, and QC variables if specified in the config
    if adata.uns["config"]["to_plot"]["qc"]["all_violins"]:
        plot_qc_violins(adata, b, part)
    # #########################################################
    # Generate scatterplots for the number of counts versus the number of genes with various QC scores
    if adata.uns["config"]["to_plot"]["qc"]["all_gene_counts_scatters"]:
        plot_qc_scatter(adata, b, part)
    # #########################################################
    # Plot all possible combinations of QC measures if specified in the config
    if adata.uns["config"]["to_plot"]["qc"]["all_combination_scatters"]:
        plot_qc_scatter_combinations(adata, part)


def plot_qc_violins(
            adata: ad.AnnData,
            keys: list[str] | None = None,
            qc_config_key: str = "keys_for_qc",
            timepoint_name: str | None = None,
            part: str = "/qc/",
            plot_in_one: bool = True,
            groupby: str | None = None,
            var_keys_only: bool = False,
            save_path: str = "",
            ncols: int = 5,
            clip_intervals: list | None = None,
            ignore_zeros: bool = True,
            do_remove: bool = False,
            show_thresholds: bool = True,
            **kwargs
        ) -> None:
    """Plots quality control (QC) violin plots for the given data.

    This function generates violin plots to visualize quality control metrics in
    the data. If no specific keys are provided, it defaults to using the keys
    specified in the ``adata.uns["config"]["general"]["keys_for_qc"]["obs"]``
    configuration. Users can choose to plot all violins in one figure or in
    separate figures based on the ``plot_in_one`` flag.

    Args:
        adata (anndata.AnnData): Adata object.
        keys (list[str] | None, optional): List of keys to plot. If None, it
            uses the keys from
            ``adata.uns["config"]["general"]["keys_for_qc"]["obs"]``. Defaults to
            None.
        qc_config_key (str, optional): The key of adata.uns["config"]["general"]
            to use (keys_for_qc or keys_for_qc_all). Defaults to "keys_for_qc".
        timepoint_name (str | None, optional): Timepoint description, e.g.,
            "before" or "after" clustering. Defaults to None.
        part (str, optional): Directory part of the analysis where this should
            be saved. Defaults to "/qc/".
        plot_in_one (bool, optional): If True, plots all violins in one figure.
            Defaults to True.
        groupby (str | None, optional): The groupby argument for violin plots.
            Defaults to None.
        var_keys_only (bool, optional): If you want to plot adata.var keys.
            Defaults to False.
        save_path (str, optional): Path to save the generated plot image.
            Defaults to "".
        clip_intervals (list | None, optional): Intervals for value clipping.
            Defaults to None.
        ignore_zeros (bool, optional): If True, ignores 0 as a value for
            percentiles. Defaults to True.
        do_remove (bool, optional): If True, values outside clip_intervals are
            set to NaN. Defaults to False.
        do_remove (bool, optional): If True, shows the already defined
            Thresholds. Defaults to False.
        kwargs (dict): Parsed to plot_violin.

    Returns:
        None:
            The function directly generates and displays the plots.

    Calls:
        get_save_path, plot_violin

    Called By:
        plot_filtering

    TODO:
        Some are actually kwargs like clip_intervals, ncols. still nice to
        have the doc, maybe?!?

    Tags:
        QC, config, groupby, obs, var, visualization
    """
    # #########################################################
    # Check if groupby is properly setup
    if groupby is None and not var_keys_only:
        groupby = adata.uns["config"]["pl"]["violin"]["groupby"]
    # #########################################################
    # Retrieve keys for QC plotting from configuration if not provided
    if keys is None:
        if not var_keys_only:
            if qc_config_key in adata.uns["config"]["general"].keys():
                keys = adata.uns["config"]["general"][qc_config_key]["obs"].copy()
            else:
                raise KeyError(
                    f"qc_config_key '{qc_config_key}' not found in "
                    "adata.uns['config']['general']")
        else:
            if qc_config_key in adata.uns["config"]["general"].keys():
                keys = adata.uns["config"]["general"][qc_config_key]["var"].copy()
            else:
                raise KeyError(
                    f"qc_config_key '{qc_config_key}' not found in "
                    "adata.uns['config']['general']")
    # #########################################################
    # Set the plotting thresholds
    # NOTE: This is a little inefficient but save, maybe think about a better solution
    if show_thresholds:
        if not var_keys_only:
            key_filtered_qc_vars = {
                k: v
                for k, v in adata.uns["config"]["pp"]["filter_qc_var_obs"].items()
                if any(sub in k for sub in keys)}
            # plot_thresholds = get_thresholds(keys, key_filtered_qc_vars)
            use_obs = True
        else:
            key_filtered_qc_vars = {
                k: v
                for k, v in adata.uns["config"]["pp"]["filter_qc_var_var"].items()
                if any(sub in k for sub in keys)}
            use_obs = False
            # plot_thresholds = get_thresholds(keys, key_filtered_qc_vars)
        plot_thresholds = {}
        for k in keys:
            thresholds = get_thresholds(
                [k], key_filtered_qc_vars, use_obs=use_obs)
            plot_thresholds[k] = thresholds[0] if thresholds else (None, None)
    # ###########################################################
    # Plot all violins in one figure if plot_in_one is set to True
    if plot_in_one:
        # Extract the needed kwargs and discard the ones we will set
        violin_kwargs = {
            **adata.uns["config"]["pl"]["violin"],
            **kwargs}  # assumes kwargs is already defined
        if "rotation" in violin_kwargs and "xlabel_rotation" not in violin_kwargs:
            violin_kwargs["xlabel_rotation"] = violin_kwargs["rotation"]
        for key in (
                "keys", "groupby", "ncols", "save_path",
                "var_keys_only", "clip_intervals", "ignore_zeros", "do_remove", "log",
                "use_raw", "stripplot", "size", "scale", "order", "multi_panel",
                "xlabel", "ylabel", "rotation"):
            violin_kwargs.pop(key, None)
        # #############################
        # Ensure show and save_path are handled correctly
        show_plot = violin_kwargs.pop("show", True)
        # #############################
        plot_violin(
            adata, keys=keys, groupby=groupby,
            ncols=ncols, save_path=save_path,
            var_keys_only=var_keys_only, clip_intervals=clip_intervals,
            ignore_zeros=ignore_zeros, do_remove=do_remove, show=show_plot,
            plot_thresholds=plot_thresholds, **violin_kwargs)
        plt.show()
    # ###########################################################
    # Plot each key in separate figures if plot_in_one is set to False
    else:
        for key in keys:
            save_path = get_save_path(
                f"_{timepoint_name}_filtering_{key}.pdf",
                adata.uns["config"], part=part)
            # Extract the needed kwargs and discard the ones we will set
            violin_kwargs = {
                **adata.uns["config"]["pl"]["violin"],
                **kwargs}
            if "rotation" in violin_kwargs and "xlabel_rotation" not in violin_kwargs:
                violin_kwargs["xlabel_rotation"] = violin_kwargs["rotation"]
            for key in ("keys", "save", "groupby", "log", "use_raw"
                        "stripplot", "size", "scale", "order", "multi_panel",
                        "xlabel", "ylabel", "rotation"):
                violin_kwargs.pop(key, None)

            plot_violin(
                adata, keys=key, save=save_path, groupby=groupby,
                plot_thresholds=plot_thresholds,
                **violin_kwargs)


def plot_qc_scatter(
            adata: ad.AnnData,
            before: str = "before",
            part: str = "/qc/"
        ) -> None:
    """Generate scatter plots with all quality control (QC) keys for coloring.

    This function creates scatter plots of the number of counts vs. number of
    genes, using all keys specified in
    ``adata.uns["config"]["general"]["keys_for_qc"]`` for coloring. It saves the
    plots in the specified directory. The function only includes keys that are
    present in ``adata.obs`` and excludes specific keys such as 'n_counts' and
    'n_genes'.

    Args:
        adata (anndata.AnnData): Adata object.
        before (str, optional): Prefix for the file name of the saved plots,
            indicating the stage of analysis. Defaults to "before".
        part (str, optional): Directory part of the analysis where this should
            be saved. Defaults to "/qc/".

    Returns:
        None

    Raises:
        ValueError: If no valid QC keys are found in ``adata.obs``.

    Calls:
        get_save_path

    Called By:
        plot_filtering

    Tags:
        QC, config, obs, visualization
    """
    # ###########################################################
    # Filter QC keys to include only those present in ``adata.obs``
    # and exclude 'n_counts' and 'n_genes'
    keys_for_qc_updated = [
        x for x in adata.uns["config"]["general"]["keys_for_qc"]
        if x in adata.obs.keys() and x not in ["n_counts", "n_genes"]]

    # Check if there are any valid keys left after filtering
    if len(keys_for_qc_updated) == 0:
        keys_for_qc_updated = [None]
    # ###########################################################
    # Plot scatter for each valid QC key and save the plots
    for key in keys_for_qc_updated:
        # Generate the file save path based on the provided configuration
        save_path = get_save_path(
            f'_{before}_filtering_{key}.pdf', adata.uns["config"], part=part)
        # ##########################################
        # Create and save the scatter plot for the current QC key
        sc.pl.scatter(
            adata, x='n_counts', y='n_genes', color=key, save=save_path,
            **adata.uns["config"]["pl"]["scatter"])


def plot_qc_scatter_combinations(
            adata: ad.AnnData,
            part: str = "/qc/"
        ) -> None:
    """Scatterplots all combinations of the given attributes.

    This function generates scatterplots for all combinations of the attributes
    specified in ``adata.uns["config"]["general"]["keys_for_qc"]``, provided that
    these keys exist in ``adata.obs``. It then saves the plots to the specified
    directory.

    Args:
        adata (anndata.AnnData): Adata object.
        part (str, optional): Directory part of the analysis where the plots
            should be saved. Defaults to "/qc/".

    Returns:
        None

    Raises:

        Warning: Logs a warning if no valid keys from ``adata.obs`` are provided
            in the configuration.

    Calls:
        get_save_path

    Called By:
        plot_filtering

    Tags:
        QC, config, obs, visualization
    """
    # ###########################################################
    # Validate the presence of keys in ``adata.obs`` and filter out non-existent keys
    keys_for_qc_updated = [
        x for x in adata.uns["config"]["general"]["keys_for_qc"]
        if x in adata.obs.keys()]

    if len(keys_for_qc_updated) == 0:
        # ##########################################
        # Log a warning if no valid keys are found
        logger.warning(
            "OOPS! You didn't pass valid ``adata.obs.keys()`` in the attributes "
            "for plot_qc_scatter_combinations!")
        return
    # ###########################################################
    # Generate and save scatterplots for all combinations of the given attributes
    for i, j, k in combinations(keys_for_qc_updated, 3):
        title = f"plot for {i} vs {j} with {k} coloring"
        this_path = "_".join(title.split(" ")) + ".pdf"
        # ##########################################
        # Determine the save path for the plot
        save_path = get_save_path(this_path, adata.uns["config"], part=part)
        # ##########################################
        # Plot and save the scatterplot
        sc.pl.scatter(
            adata, x=i, y=j, color=k, title=title, save=save_path,
            show=adata.uns["config"]["pl"]["scatter"]["show"])


# ###########################################################################################################
# CMO/Hashes
def plot_hash_violins(
            adata: ad.AnnData,
            obsm_key: str = "hashes",
            hash_keys: list[str] = None,
            logs: list[str] = None,
            norms: list[str] = None,
            figsize: Sequence[float | int] = (10, 6)
        ) -> None:
    """Plot violin plots per log + norm combination for hash keys from ``obsm``.

    By default, plots all logs and all norms detected in ``obsm``, including raw
    and clipped variants.

    Args:
        adata (anndata.AnnData): Adata object containing ``obsm`` dataframe with
            hash data.
        obsm_key (str, optional): Key in ``obsm`` where features are stored.
            Defaults to "hashes".
        hash_keys (list[str], optional): Hash keys to plot (base names without
            transforms). If None, auto-detects. Defaults to None.
        logs (list[str], optional): List of logs to plot (e.g., ["log10",
            "log1p"]). If None, auto-detects all present. Defaults to None.
        norms (list[str], optional): List of norms to plot (e.g.,
            ["norm_separately", "lognorm"]). If None, auto-detects all present.
            Defaults to None.
        figsize (Sequence[float | int], optional): Figure size.
            Defaults to (10, 6)

    Returns:
        None:
            None
            visualization

    Tags:
        visualization
    """
    df = adata.obsm[obsm_key].copy()

    # Auto-detect hash keys if not provided
    if hash_keys is None:
        all_cols = df.columns
        hash_keys = list(set(col.split("_")[0] for col in all_cols if "_" in col))

    # Auto-detect logs if not provided
    if logs is None:
        logs_detected = set()
        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 2 and parts[1] in ["log10", "log1p"]:
                logs_detected.add(parts[1])
        if any(f"{hk}_raw" in df.columns for hk in hash_keys):
            logs_detected.add("raw")
        if any(f"{hk}_raw_norm" in df.columns for hk in hash_keys):
            logs_detected.add("raw_norm")
        logs = sorted(list(logs_detected))

    # Auto-detect norms if not provided
    if norms is None:
        norms_detected = set()
        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 3 and parts[1] in ["log10", "log1p"]:
                norms_detected.add("_".join(parts[2:]))
            elif len(parts) == 3 and parts[1] == "raw":
                norms_detected.add(parts[2])
        norms = sorted(list(norms_detected))
        if not norms:
            norms = [None]

    # Plot for each combination of log and norm
    for log_method in logs:
        for norm_method in norms:
            plot_df = pd.DataFrame()

            for hk in hash_keys:
                # Build column name
                if log_method in ["raw", "raw_norm"]:
                    col_name = f"{hk}_{log_method}"
                else:
                    col_name = f"{hk}_{log_method}"
                    if norm_method:
                        col_name += f"_{norm_method}"

                if col_name in df.columns:
                    temp_df = pd.DataFrame({
                        "value": df[col_name],
                        "hash": hk
                    })
                    plot_df = pd.concat([plot_df, temp_df], axis=0)

            if plot_df.empty:
                logger.info(f"No data for log: {log_method}" + (f", norm: {norm_method}" if norm_method else ""))
                continue

            plt.figure(figsize=figsize)
            violinplot(data=plot_df, x="hash", y="value", inner="quartile")
            title = f"Violin plot {log_method}"
            if norm_method:
                title += f" + {norm_method}"
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


# ###########################################################################################################
# Downstream
def plot_mediods_heatmap(
            adata: ad.AnnData,
            ref: list[str] | dict[str, list[str]],
            show_min: int = 20,
            use_scale: bool = False,
            scaler: float = 0.05,
            cluster_key: str = "leiden",
            distance_metric: str = "manhattan",
            use_medioids: bool = True,
            show_duplicates: bool = False,
            show_classes: bool = False,
            use_means: bool = False,
            part: str = "/downstream/",
            layer: str = "log2norm_counts"
        ) -> None:
    """
    Creates a subsetted gene heatmap while maintaining variance based on the
    specified clustering key.

    This function generates a heatmap by clustering cells based on the specified
    ``cluster_key``, allowing for the visualization of medioids or means of
    subclusters. It can scale the clusters proportionally and handle various
    configurations for displaying marker genes.

    Args:
        adata (anndata.AnnData): Adata object.
        ref (list[str] | dict[str): List or dictionary of marker genes to be
            displayed in the heatmap. If a dictionary is provided, keys are
            classes, and values are lists of ref/markers..
        show_min (int, optional): Minimum number of cells or mediods/means to
            display per cluster. Defaults to 20.
        use_scale (bool, optional): If True, scales the number of centroids to
            show based on the ``scaler`` value. Defaults to False.
        scaler (float, optional): Scaling factor for the number of centroids to
            display, used if ``use_scale`` is True. Must be between 0 and 1.
            Defaults to 0.05.
        cluster_key (str, optional): Key in ``adata.obs`` used for clustering the
            cells. Defaults to "leiden".
        distance_metric (str, optional): Distance metric used for subclustering.
            "manhattan" corresponds to mediods, "euclidean" to means.
            Defaults to "manhattan".
        use_medioids (bool, optional): If True, uses the mediods of the
            subclusters for the heatmap. Defaults to True.
        use_means (bool, optional): If True, uses the means of the subclusters
            for the heatmap. Defaults to False.
        show_duplicates (bool, optional): If True, allows marker genes to appear
            multiple times if they are in ref for different clusters. Defaults
            to False.
        show_classes (bool, optional): If True, displays clusters or classes in
            the heatmap. Defaults to False.
        part (str, optional): Directory path for saving the resulting heatmap.
            Defaults to "/downstream/".
        layer (str, optional): The layer of the ``adata`` object to use for the
            analysis. Defaults to "log2norm_counts".

    Returns:
        None

    Raises:
        AttributeError: If the specified ``layer`` is not found in ``adata.layers``.

    Calls:
        deduplicate_ref_dict, get_all_markers_from_ref_dict, get_colormap,
        get_save_path

    Called By:
        run_downstream

    Tags:
        annotation, clustering, config, obs, visualization
    """
    # #########################################################
    # Sanity check for the specified layer and temporarily replace adata.X with the desired layer
    if layer not in adata.layers.keys():
        raise AttributeError(f"layer {layer} not in adata.layers!")
    prev_layer = adata.X
    adata.X = adata.layers[layer]
    # #########################################################
    # Process the ref and handle different marker formats (list or dict)
    if isinstance(ref, dict):
        if not show_duplicates:
            ref = deduplicate_ref_dict(ref)
        if not show_classes:
            ref = get_all_markers_from_ref_dict(ref, not show_duplicates)
    else:
        if show_classes:
            logger.info("NOTE: The passed ref is a list, for visualising the classes,"
                        " pass a dict with classes as keys and features as a list for the values")
        if not show_duplicates:
            ref = pd.unique(ref)
    # #########################################################
    # Create a copy of the adata object to store the plotting results
    median_plotting = adata[[False, ] * adata.shape[0]].copy()
    # #########################################################
    # Iterate through each cluster and process the subclustering
    for c in adata.obs[cluster_key].unique():
        # ##########################################
        # Subset the data based on the current cluster
        cluster_view = adata[adata.obs[cluster_key] == c]
        cluster_size = cluster_view.shape[0]
        # ##########################################
        # Handle cases where cluster size is greater than show_min
        if cluster_size > show_min:
            if use_scale:
                n_to_show = int(cluster_size * scaler + .5)
            else:
                n_to_show = show_min

            # Apply KMedoids subclustering
            model = KMedoids(metric=distance_metric, n_clusters=n_to_show)
            subclustering = model.fit_predict(cluster_view.X)
            # ##########################################
            # Handle the use_medioids option
            if use_medioids:
                median_subclustering = cluster_view[model.medoid_indices_, :].copy()
                median_plotting = ad.concat([median_plotting, median_subclustering])
            # ##########################################
            # Handle the use_means option
            if use_means:
                median_mean_subclustering = cluster_view[model.medoid_indices_, :].copy()
                for i in sorted(np.unique(subclustering)):
                    cluster_view[subclustering == i].X.mean(0)
                    median_mean_subclustering.X[i, :] = cluster_view[subclustering == i].X.mean(0)
                median_plotting = ad.concat([median_plotting, median_mean_subclustering])
        # ##########################################
        # If the cluster size is less than show_min, use all cells
        else:
            median_plotting = ad.concat([median_plotting, cluster_view.copy()])
    # #########################################################
    # Generate and save the heatmap
    save_path = get_save_path("marker_genes_medioids_subclustering.pdf", adata.uns["config"], part)
    sc.pl.heatmap(median_plotting, ref, groupby=cluster_key, save=save_path,
                  cmap=adata.uns["config"].get_colormap(cmap=adata.uns["config"]["pl"]["heatmap"]["cmap"]),
                  **{k: v for k, v in adata.uns["config"]["pl"]["heatmap"].items() if k not in ["cmap"]})
    # #########################################################
    # Restore the original layer in adata.X
    adata.X = prev_layer


# ################################################################
# Feature improvement plots
def check_scoring_validity(
            adata: ad.AnnData,
            key: str = "n_unique",
        ) -> None:
    """Check and visualize the unique counts within the given adata object.

    NOTE:
        - This is for developing a scoring system for the n_unique, but possibly
          usable for any other scoring.
        - The keys "n_counts", "n_genes" "pct_MT" must be present in the obs!
        - Automatically uses "{key}_score" and "{key}_score_norm" if present.

    This function processes the adata object ``adata`` to visualize the
    distribution of unique counts across different attributes of cells and
    genes. The function creates histograms for various metrics such as
    'n_unique_score' and 'n_unique_score_norm', both for obs and var in the
    adata object. Additionally, it generates scatter plots to explore the
    relationships between these metrics and other features like 'n_counts',
    'pct_MT', and 'n_genes'.

    Args:
        adata (anndata.AnnData): Adata object.
        key (str, optional): A obs key to check. Defaults to "n_unique".

    Returns:
        None: The function creates and displays plots, but does not return any
            value.

    Calls:
        get_n_unique

    Tags:
        QC, config, obs, var, visualization
    """
    # #########################################################
    # Calculate unique counts if not present
    if "n_unique" not in adata.obs.keys():
        # This function processes the adata object to calculate unique counts.
        get_n_unique(adata)
    # ##########################################
    # Plot histograms for 'n_unique_score' across cells (obs)
    if f"{key}_score" in adata.obs.keys():
        plt.hist(adata.obs[f"{key}_score"], bins=100)
        plt.show()

    if f"{key}_score_norm" in adata.obs.keys():
        plt.hist(adata.obs[f"{key}_score_norm"], bins=100)
        plt.show()
    # ##########################################
    # Plot histograms for 'n_unique_score' across genes (var)
    if f"{key}_score" in adata.var.keys():
        plt.hist(adata.var[f"{key}_score"], bins=100)
        plt.show()

    if f"{key}_score_norm" in adata.var.keys():
        plt.hist(adata.var[f"{key}_score_norm"], bins=100)
        plt.show()
    # #########################################################
    # Identify and log genes with low unique counts
    low_genes = adata.var_names[adata.var[f"{key}"] < 3]
    logger.debug(len(low_genes))  # Log the number of genes with low unique counts.
    # #########################################################
    # Plot distribution of unique counts across genes
    plt.hist(adata.var[f"{key}"], bins=100)
    plt.yscale("log")  # Set y-axis to logarithmic scale for better visualization of distribution.
    plt.show()

    plt.hist(adata.var[f"{key}_score"], bins=100)
    plt.show()

    plt.hist(adata.var[f"{key}_score_norm"], bins=100)
    plt.show()
    # #########################################################
    # Generate scatter plots to explore relationships between features
    color = f"{key}"
    sc.pl.scatter(adata, "n_counts", "pct_MT", color=color)
    sc.pl.scatter(adata, "n_genes", "pct_MT", color=color)
    sc.pl.scatter(adata, "n_counts", "n_genes", color=color)

    color = f"{key}_score"
    sc.pl.scatter(adata, "n_counts", "pct_MT", color=color)
    sc.pl.scatter(adata, "n_genes", "pct_MT", color=color)
    sc.pl.scatter(adata, "n_counts", "n_genes", color=color)

    color = f"{key}_score_norm"
    sc.pl.scatter(adata, "n_counts", "pct_MT", color=color)
    sc.pl.scatter(adata, "n_genes", "pct_MT", color=color)
    sc.pl.scatter(adata, "n_counts", "n_genes", color=color)
