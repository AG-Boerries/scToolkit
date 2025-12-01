'''Here you see the major functions to call for a pre defined single cell experiment.

Core API for executing standardized single-cell RNA-seq analysis workflows
using Scanpy. This module wraps preprocessing, filtering, clustering, and
visualization steps in a modular and reproducible fashion.

Key Concepts:
-------------
- ``setup_adata(adata, **config_kwargs)``: Initializes ``adata`` with
  configuration, stats, and GPU setup. This replaces manual calls to
  ``sc_config.get_config(...)`` and associated setup logic.

- adata.uns["config"]:
    Stores all analysis parameters (e.g., dataset name, paths, clustering settings)
    as a centralized config object created via sc_config. Used throughout the pipeline
    for consistent behavior.

- adata.uns["stats"]:
    Contains dataset-level summary statistics (e.g., gene/cell counts, QC metrics)
    computed during initialization. Serves as a reference for filtering and diagnostics.

- adata.uns["genesets"]:
    A dictionary where keys are gene set names and values are lists of genes.
    Used for running gene set enrichment methods across the dataset.

- adata.uns["ref_dict"] / ["ref_dict_upd"] / ["ref_dict_upd_subs"]:
    Contain reference gene set definitions and their filtered or updated versions
    used for enrichment benchmarking and comparison.

- Conventions for Function/Method arguments:
    - ``key``: Single reference to ``adata.obs``, ``adata.obsm``, ``adata.var_names``, etc.
      Usually the doc specifies, which of them are usable.
    - ``keys``: List or dict of keys, grouped logically.
    - ``ref``: Mostly dict, but also sometimes list for group to gene/markers mapping.
      This can be genes, or whatever modality you have the features.
      This is different from the keys because it is not related to the obs,
      only to the adata.var_names. DON'T confuse this with ``reference``
    - ``ref_dict``: Like ref, but specific for dictionarys mapping group to fetures.
    - ``ref_df``: Like ref, but specific for a datafame.
    - ``obs_key``: A single column in ``adata.obs``.
    - ``obs_keys``: A list of columns in ``adata.obs``.
    - ``groupby``: Categorical column in ``adata.obs``, used for grouping.
    - ``reference``: Used for comparisons, there reference marks the reference,
      or for the general term refernce sort according to reference.
      THIS has nothing to do with ``ref``.

- Conventions for Function/Method name prefixes:
    - ``plot_``: plotting function start with it usually they have saving
      and are not to confuse with ``show_`` which are checkers mostly.
    - ``print_``: Printing helpers for too long lines or dataframes in chunks
      etc.
    - ``get_``: getters start.
    - ``extract_``: Usually reserved for subsetters or refiners instead of get.
    - ``calc_``: calculator start.
    - ``search_``: Package or database search helpers .
    - ``check_``: Checkers, to for validation or quick checks.
    - ``ref_``: Handling helpers for ``ref`` like remove genes not present in
      adata.
    - ``df_``: Dataframe handling helpers e.g for conversion to ref_dict.
    - ``update_``: Updating functions for config or nested structures
    - ``replace_``: For functions that stack replace things.
    - ``filter_`` or ``tag_``: For functions that filter or tag (Mostly QC)
    - Possibly more and if you find a function, that is not according to
      conventions, please report it.

Typical Workflow:
-----------------

.. code::

    import scanpy as sc
    from scToolkit import sc_code
    from scToolkit.sc_utils import initialize_cuda

    adata = sc.read_h5ad("your_file.h5ad")
    sc_code.setup_adata(
        adata,
        param_setup="default",
        dataset_name="your_dataset",
        save_path="results/",
        use_GPU=True)
    sc_code.all_qc_preprocessing(adata)
    sc_code.plot_filtering(adata, before=True)
    adata = sc_code.all_filters_tagging(adata, apply_filters=True)
    sc_code.plot_filtering(adata, before=False)
    adata = sc_code.Run_all_prep_steps_clustering(adata)
    adata = cluster_SC_scanpy_like(adata)
    adata = run_downstream(adata)

Summary:
--------
This module supports execution of all steps required for a full single-cell
workflow. It is tightly coupled with ``sc_config.py`` for dynamic config handling
and expects ``adata.uns["config"]`` to be initialized via ``setup_adata``.

This Module provides the general functions to call for a single cell experiment.

General conventions for the package:
    key: usually can be one of adata.obsm.keys(), adata.obs.keys, adata.var_names
    (sometimes also the adata.varm.keys and adata.obs_names)

    keys: like key but a list or dict with key as groups.

    obs_key: A single key in the adata.obs.keys()
    obs_keys: a list of adata.obs.keys()


    group is the general synonyme for categorical columns cluster/celltype/ 2 different arguments arrise
    groupby: therefore must be a single key in adata.obs.keys() and categorical
    (adata.obs[groupby].dtype.name == "category")

══════════════════════════════════════════════════
'''

# Standard Library Imports
from . import (
    os, time, pickle, lzma, gzip, sub, copy, rc_context,
    plt, warnings, Sequence,  # noqa: F401
    # json_dump, plt,  # noqa: F401
    # Pandas and NumPy for data manipulation and analysis
    np, pd,
    # Scanpy and adata for single-cell data analysis
    sc, ad,
    # Muon for multi-omics data analysis
    # pt,
    # Handling for the buggy CUDA error message code
    rsf,
    # ##########################
    # scToolkit specifics
    # Utility Functions
    df_split_col,
    # Logging and Configuration Functions
    get_logger, get_config, get_stats,
    # update_nested_dict,
    # Plotting Functions
    plot_ref_dotplots, plot_mediods_heatmap,
    plot_per_group_DEG_dotplot_n_gene_dendrogram,
    plot_per_group_stacked_violins, plot_ref_stacked_violins,
    plot_per_group_DEG_umaps, save_genesets_to_csv,
    # Custom Analysis Functions
    plot_embedding_density, convert_mixed_types, flag_gene_family,
    get_n_unique, convert_to_mem_efficient, setup_image_folder, get_save_path,
    get_adata_by_mod, get_DEGs_per_group, create_group_stats_df,
    get_shape_diff, filter_genes_multicall, filter_cells_multicall,
    get_highly_variable, filter_genes_n_families_regex, filter_genes_n_families,
    mark_highly_variable_genes, initialize_cuda, get_valide_ref_dicts,
    get_all_markers_from_ref_dict, get_DEG_gene_csvs,
    calc_highly_variable_genes_unique_based, ALL_PATHS,
    GENESET_FILENAMES,
    get_print_highlighter, continuos_umap_helper, discrete_umap_helper,
    ref_dict_sort_values_hvg_like, calc_DEGs, plot_umap,
    # replace_np_array_with_list_recursive, plot_umap,  # noqa: F401
    # get_categorical_columns, get_adata_sub_keys,  # noqa: F401
    plot_umap_cat_splitting, get_group_hierarchy, calc_top_n_ratio,
    get_colormap, plot_jaccard_heatmap_cluster_comparison)

# Initialize the logger
logger = get_logger(name="sc_code")


def test_logger() -> None:
    """Helper"""
    logger.info("bla")


# ##########################################################################################################
# Basic Handling
# (adata, param_setup="default", dataset_name="default", save_path="analysis",
#                        use_GPU=True, overwrite=False, organism="human",
#                        selected_gene_markers=[]):
def setup_adata(
            adata: ad.AnnData,
            overwrite: bool = False,
            **config_kwargs
        ) -> None:
    """
    Prepares the adata object for downstream analysis by configuring and
    initializing necessary settings.

    This function adds configurations, statistical summaries, and GPU
    initialization to the adata object to prepare it for further processing. If
    configurations or stats are already present in ``adata``, they will only be
    overwritten if ``overwrite`` is set to True.

    NOTE:
        If the GPU initialization fails, you may need to rerun this function
        with ``use_GPU=False`` and ``overwrite=True`` to bypass GPU setup.

    Args:
        adata (anndata.AnnData): Adata object to be configured and prepared for
            analysis.
        overwrite (bool, optional): If True, existing configurations and stats
            in ``adata`` will be overwritten. Defaults to False.
        **config_kwargs: Additional configuration parameters passed to
            sc_config.get_config().

    Returns:
        None

    Raises:
        KeyboardInterrupt: Raised if GPU initialization is interrupted manually.
        Exception: Raised if GPU initialization fails due to any other issue.

    Calls:
        get_config, get_stats, initialize_cuda

    Tags:
        config, pipeline, stats
    """
    # #########################################################
    # Configuration handling: Check if a configuration already exists, and if not or if forced, set it up.
    if "config" in adata.uns.keys() and not overwrite:
        logger.warning('You already have a adata.uns["config"]!')
    else:
        adata.uns["config"] = get_config(**config_kwargs)
    # #########################################################
    # Stats handling: Check if statistical summaries are already present, and if not or if forced, create them.
    if "stats" in adata.uns.keys() and not overwrite:
        logger.warning('You already have a adata.uns["stats"]!')
    else:
        adata.uns["stats"] = get_stats()
    # #########################################################
    # GPU support handling: Initialize GPU support if specified in the configuration.
    if "use_GPU" in config_kwargs.keys():
        if config_kwargs["use_GPU"]:
            try:
                initialize_cuda()
            except KeyboardInterrupt:
                logger.warning("The GPU is not initialized!")
            except Exception:  # noqa: E722
                logger.warning(
                    "The GPU failed to initialize, please run sc_utils.initialize_cuda() to see the error!\n"
                    "If you don't manage to fix the issue, you need to rerun setup_adata\n"
                    "with use_GPU=False, overwrite=True!")


# ###########################################################################################################
# QC
def all_qc_preprocessing(
            adata: ad.AnnData,
            part: str = "/qc/",
            prot: bool = False,
            try_rerun: bool = False,
            apply_filters: bool = False,
        ) -> None:
    """
    This function annotates the quality control (QC) genes and calculates
    the QC variables.

    This function annotates QC-related genes and calculates various QC metrics.
    It does not utilize a GPU due to the overhead outweighing the performance
    gains. Additionally, it focuses on percentages rather than raw counts for QC
    measures and removes the ``raw`` layer from the adata object, as it does not
    provide additional utility and can interfere with certain functionalities.

    NOTE:
        1. This function doesn't use a GPU by now, because the overhead of
            moving the data to a GPU is larger
            than the speed benefit of computing on the GPU.
        2. The percentage of major genes is removed, and also the actual counts
            of the QC measure, because we are
            only interested in percentages. (may change, because the actual
            counts can be informative for duplicates?!?)
        3. This function deletes the raw layer, because it has no use and
            interferes with some functionality.

    The default created obs keys are:
        - n_genes: Number of genes a cell has.
        - n_counts: Total counts of a cell.
        - n_unique: Number of unique count values a cell/gene has.
        - n_unique_score: Scored uniqueness, considering the number of unique
          values and the uniqueness of each value. (Experimental)
        - pct_MT: Percentage of mitochondrial content.
        - pct_RBS: Percentage of ribosomal content.
        - pct_MTRBS: Percentage of mitochondrial ribosome content. (Deprecated,
          has no information??)
        - pct_RBS_MTRBS: pct_RBS + pct_MTRBS (Deprecated, has no information??)

    Args:
        adata (anndata.AnnData): Adata object.
        part (str, optional): Path to save the generated figures.
            Defaults to "/qc/".
        prot (bool, optional): Indicates if the data is Citeseq data.
            Defaults to False.
        apply_filters (bool, optional): If true, it applys the filters directly.
            Defaults to False.

    Returns:
        None

    Calls:
        all_filters_tagging, convert_to_mem_efficient, flag_gene_family,
        get_n_unique, setup_image_folder

    TODO:
        - Check if it really is the same object before and afterwards!!!!!
        - pct_MTRBS/pct_RBS + pct_MTRBS (Deprecated, has no information??)

    Tags:
        QC, config, obs, pipeline, stats, var
    """
    # #########################################################
    # Check if the preprocessing was already run
    if "all_qc_preprocessing" in adata.uns["stats"]["preprocessors_ran"]:
        logger.warning("You already ran the all_qc_preprocessing, skipping re-execution!")
        if not try_rerun:
            return

    start = time()
    # #########################################################
    # Delete the raw layer if it exists
    if adata.raw is not None:
        logger.warning("Deleted the adata.raw!")
        del adata.raw
    # #########################################################
    # Set up the paths for saving figures
    setup_image_folder(adata.uns["config"], part)
    # #########################################################
    # Flag gene families on CPU, as there is no advantage in using GPU here
    flag_gene_family(adata, prot=prot)
    logger.info(f"flag_gene_family: {time() - start}")
    start = time()
    # #########################################################
    # Remove duplicate keys before running scanpy's QC metrics calculation
    for key in [f"total_counts_{key}" for key in adata.uns["config"]["pp"]["calculate_qc_metrics"]["qc_vars"]]:
        if key in adata.obs.keys():
            del adata.obs[key]

    for key in [f"pct_{key}" for key in adata.uns["config"]["pp"]["calculate_qc_metrics"]["qc_vars"]]:
        if key in adata.obs.keys():
            del adata.obs[key]

    # Run scanpy QC metrics calculation
    sc.pp.calculate_qc_metrics(adata, **adata.uns["config"]["pp"]["calculate_qc_metrics"])
    logger.info(f"QC_metrics: {time() - start}")
    start = time()
    # #########################################################
    keys_for_qc = ["n_genes", "n_counts", "n_unique", "max_count"]
    keys_for_qc = keys_for_qc + [f'top_{i}_ratio' for i in [5, 10]]

    # Rename or delete annotations, remove total keys since only percentages are needed
    for key in [f"total_counts_{key}" for key in adata.uns["config"]["pp"]["calculate_qc_metrics"]["qc_vars"]]:
        if key in adata.obs.keys():
            if not adata.uns["config"]["general"]["del_qc_counts"]:
                this_key = sub("total_counts_", "n_", key)
                adata.obs[this_key] = adata.obs[key]
                keys_for_qc.append(this_key)
            del adata.obs[key]

    # Rename keys to remove unnecessary information
    adata.obs.rename(
        columns={k: k.replace("_counts", "") for k in adata.obs.keys() if "_counts_" in k}, inplace=True)
    # #########################################################
    # Remove unnecessary dropout keys
    if "pct_dropout_by_counts" in adata.var.keys():
        del adata.var["pct_dropout_by_counts"]
    # #########################################################
    # Clean var and obs to remove duplicate columns
    for key in ["n_genes", "n_counts"]:
        if key in adata.obs.keys():
            del adata.obs[key]

    for key in ["n_cells", "n_counts"]:
        if key in adata.var.keys():
            del adata.var[key]
    # #########################################################
    # Rename columns to more meaningful names
    adata.obs.rename(columns={"n_genes_by_counts": "n_genes", "total_counts": "n_counts"}, inplace=True)
    adata.var.rename(columns={"n_cells_by_counts": "n_cells", "total_counts": "n_counts"}, inplace=True)
    # #########################################################
    # Calculate unique counts per gene and cell
    logger.info(f"After_flag_gene_family: {time() - start}")
    start = time()
    get_n_unique(adata)
    logger.info(f"get_n_unique_n_max_count: {time() - start}")
    start = time()
    # #########################################################
    # Get the max count
    adata.obs["max_count"] = adata.X.max(1).toarray().flatten()
    for i in [5, 10]:
        adata.obs[f'top_{i}_ratio'] = calc_top_n_ratio(adata, i)
    # #########################################################
    # Convert data types to more memory-efficient ones
    # TODO: Consider the possibility of overflows during summation.
    for key in ["n_genes", "n_counts", "n_unique", "max_count"]:
        adata.obs[key] = convert_to_mem_efficient(adata.obs[key])
    # #########################################################
    # Collect QC keys for visualization and reduce precision of percentages
    if not prot:
        pct_keys = [f"pct_{k}" for k in adata.uns["config"]["pp"]["calculate_qc_metrics"]["qc_vars"]]
        keys_for_qc.extend(pct_keys)

        # Decrease precision of percentage keys
        temp_keys_to_del = []
        for key in pct_keys:
            if len(adata.obs[key].unique()) <= 3:
                temp_keys_to_del.append(key)
                del adata.obs[key]
                keys_for_qc.remove(key)
            else:
                adata.obs[key] = adata.obs[key].astype(np.float16)

        if temp_keys_to_del:
            logger.warning(f'Removing {", ".join(temp_keys_to_del)} because '
                           'they have less than 3 unique values!')
    # #########################################################
    # Delete unnecessary var keys after QC preprocessing
    if adata.uns["config"]["general"].get(
            "keep_qc_var_gene_identifiers", False):
        for key in adata.uns["config"]["pp"]["calculate_qc_metrics"]["qc_vars"]:
            if key in adata.var.keys():
                del adata.var[key]
    # #########################################################
    # Add the percentage of cells to the var
    adata.var["pct_cells"] = (adata.var["n_cells"] / adata.shape[0]).astype(np.float16)
    # #########################################################
    # Get the max count
    adata.var["max_count"] = adata.X.max(0).toarray().flatten()
    # #########################################################
    # Save the all qc keys and the most usefull ones
    # #################
    # Create the keys
    adata.uns["config"]["general"]["keys_for_qc"] = {"obs": [], "var": []}
    adata.uns["config"]["general"]["keys_for_qc_all"] = {"obs": [], "var": []}
    # #################
    # Fill all and copy for usefull ones
    adata.uns["config"]["general"]["keys_for_qc_all"]["obs"] = keys_for_qc
    var_keys_for_qc = np.intersect1d([
            "n_cells", "n_counts", "n_unique", "pct_cells", "max_count"
            # "means", "variances", "variances_norm"
            ], adata.var.keys()).tolist()
    adata.uns["config"]["general"]["keys_for_qc_all"]["var"] = var_keys_for_qc
    # #################
    # Fill usefull by taking the config["general"]["qc_keys_to_keep"]["obs"]
    # OBS
    if adata.uns["config"]["general"]["qc_keys_to_keep"]["obs"]:
        valid_qc_obs = np.intersect1d(
            keys_for_qc,
            adata.uns["config"]["general"]["qc_keys_to_keep"]["obs"]).tolist()
        if valid_qc_obs:
            adata.uns["config"]["general"]["keys_for_qc"]["obs"] = valid_qc_obs
        else:
            raise ValueError(
                "Malformed configuration in adata.uns['config']['general']: "
                "none of the obs keys defined in "
                "'qc_keys_to_keep[\"obs\"]' are present in "
                "'keys_for_qc[\"obs\"]'. Please ensure that at least one key "
                "listed in qc_keys_to_keep['obs'] also appears in "
                "keys_for_qc['obs'].")
    else:
        adata.uns["config"]["general"]["keys_for_qc"]["obs"] = keys_for_qc
    # VAR
    if adata.uns["config"]["general"]["qc_keys_to_keep"]["var"]:
        var_keys_for_qc = np.intersect1d(
            var_keys_for_qc,
            adata.uns["config"]["general"]["qc_keys_to_keep"]["var"]).tolist()
        if var_keys_for_qc:
            adata.uns["config"]["general"]["keys_for_qc"]["var"] = var_keys_for_qc
        else:
            raise ValueError(
                "Malformed configuration in adata.uns['config']['general']: "
                "none of the var keys defined in "
                "'qc_keys_to_keep[\"var\"]' are present in "
                "'keys_for_qc[\"var\"]'. Please ensure that at least one key "
                "listed in qc_keys_to_keep['var'] also appears in "
                "keys_for_qc['var'].")
    else:
        adata.uns["config"]["general"]["keys_for_qc"]["var"] = var_keys_for_qc
    # #########################################################
    # Tag low-quality cells and genes
    logger.info(f"after_get_n_unique: {time() - start}")
    start = time()
    all_filters_tagging(adata, apply_filters=apply_filters)
    logger.info(f"all_filters_tagging: {time() - start}")
    start = time()

    logger.info(f'Tagged {(~adata.obs["good_quality"]).sum()} cells and '
                f'{(~adata.var["good_quality"]).sum()} genes as bad quality.')
    # #########################################################
    # Log the QC keys used for visualization
    logger.info(
        f'QC keys for visualization: '
        f'{adata.uns["config"]["general"]["keys_for_qc"]["obs"]}.\n'
        'You can change them by accessing it via: '
        'adata.uns["config"]["general"]["keys_for_qc"]["obs"] '
        'and the corresponding regex '
        'adata.uns["config"]["qc"]["gene_identifiers"]')
    # #########################################################
    # Mark this preprocessing step as complete
    adata.uns["stats"]["preprocessors_ran"].append("all_qc_preprocessing")


def all_filters_tagging(
            adata: ad.AnnData,
            qc_key: str = "good_quality",
            reset: bool = False,
            return_filter_log: bool = False,
            apply_filters: bool = False
        ) -> pd.DataFrame | tuple[ad.AnnData, pd.DataFrame] | ad.AnnData | None:
    """Runs all filtering criteria defined in ``adata.uns["config"]["pp"][keys]``.

    The filtering keys include: ``filter_cells``, ``filter_genes``, and
        ``filter_qc_var``.

    Args:
        adata (anndata.AnnData): Adata object.
        qc_key (str, optional): Key in ``adata.obs`` and ``adata.var`` to mark good
            quality cells/genes. Default is "good_quality". Defaults to
            "good_quality".
        reset (bool, optional): If True, resets the qc_key values before
            applying new filters. Default is False. Defaults to False.
        return_filter_log (bool, optional): If True, returns a DataFrame with
            filter effect summary. Default is False. Defaults to False.
        apply_filters (bool, optional): If True, returns a adata object with
            filter effect summary. Default is False. Defaults to False.

    Returns:
        pandas.DataFrame | tuple[anndata.AnnData, pandas.DataFrame] | anndata.AnnData | None:
            - If ``return_filter_log`` is True and ``apply_filters`` is False:
                Returns a ``pandas.DataFrame`` with filter statistics.
            - If both ``return_filter_log`` and ``apply_filters`` are True:
                Returns a tuple ``(filtered AnnData, filter log DataFrame)``.
            - If only ``apply_filters`` is True:
                Returns filtered ``anndata.AnnData``.
            - Otherwise:
                Returns ``None``.

    Raises:
        KeyError: If the required keys are not present in
            ``adata.uns["config"]["pp"]``.

        Warning: If specific keys in the filtering criteria are not present in
            ``adata.obs`` or ``adata.var``.

    Called By:
        all_qc_preprocessing

    Tags:
        QC, config, obs, pipeline, stats, var
    """
    start_sub = time()
    # #########################################################
    # Ensure stats dictionary exists in adata.uns
    if "stats" not in adata.uns.keys():
        adata.uns["stats"] = {}
    # #########################################################
    # Store the raw shape of the data before filtering
    if "raw_shape" not in adata.uns["stats"].keys():
        adata.uns["stats"]["raw_shape"] = [adata.shape[0], adata.shape[1]]
    # #########################################################
    # Prepare obs-level quality control (QC) filtering
    # If qc_key is not already present or reset is True, initialize it
    if qc_key not in adata.obs.keys() or reset:
        adata.obs[qc_key] = True

    filter_log = []
    original_obs_mask = adata.obs[qc_key].copy()
    # #########################################################
    # Apply filtering criteria to obs-level data (obs)
    for key in adata.uns["config"]["pp"]["filter_qc_var_obs"]:
        # The key has to start with "min_" or "max_" followed by an obs key
        subset_direction, obs_key = key[:4], key[4:]

        # Check if the obs_key exists in adata.obs
        if obs_key not in adata.obs.keys():
            logger.warning(f"Can't process qc obs variable key: {obs_key}, because the key doesn't "
                           "contain a valid obs key in the string!")
            continue

        # Report that there are Nans
        if adata.obs[obs_key].isna().any():
            logger.warning(f"NaN values detected in obs key '{obs_key}', excluded from filtering.")
        # Get the subsetter for the nans
        obs_invalid = adata.obs[obs_key].isna()

        if "min_" == subset_direction:
            new_mask = (adata.obs[obs_key] > adata.uns["config"]["pp"]["filter_qc_var_obs"][key]) | obs_invalid
        elif "max_" == subset_direction:
            new_mask = (adata.obs[obs_key] < adata.uns["config"]["pp"]["filter_qc_var_obs"][key]) | obs_invalid
        else:
            logger.info(f"Can't process qc variable key: {key}, because no min_ or max_ in the string!")
            continue

        filtered_count = int((original_obs_mask & ~new_mask).sum())
        adata.obs[qc_key] = adata.obs[qc_key] & new_mask

        filter_log.append({
            "axis": "obs",
            "filter": key,
            "threshold": adata.uns["config"]["pp"]["filter_qc_var_obs"][key],
            "n_removed": filtered_count})
    # #########################################################
    # Prepare variable-level quality control (QC) filtering
    # If qc_key is not already present, initialize it
    if qc_key not in adata.var.keys() or reset:
        adata.var[qc_key] = True

    original_var_mask = adata.var[qc_key].copy()
    # #########################################################
    # Apply filtering criteria to variable-level data (var)
    for key in adata.uns["config"]["pp"]["filter_qc_var_var"]:
        # The key has to start with "min_" or "max_" followed by a var key
        subset_direction, var_key = key[:4], key[4:]
        # Check if the var_key exists in adata.var
        if var_key not in adata.var.keys():
            logger.info(f"Can't process qc var variable key: {key}, because the key doesn't "
                        "contain a valid var key in the string!")
            continue

        # Report that there are Nans
        if adata.var[var_key].isna().any():
            logger.warning(f"NaN values detected in var key '{var_key}', excluded from filtering.")
        var_invalid = adata.var[var_key].isna()

        if "min_" == subset_direction:
            new_mask = (adata.var[var_key] > adata.uns["config"]["pp"]["filter_qc_var_var"][key]) | var_invalid
        elif "max_" == subset_direction:
            new_mask = (adata.var[var_key] < adata.uns["config"]["pp"]["filter_qc_var_var"][key]) | var_invalid
        else:
            logger.info("Can't process qc variable key: {key}, because no min_ or max_ in the string!")
            continue
        filtered_count = int((original_var_mask & ~new_mask).sum())
        adata.var[qc_key] = adata.var[qc_key] & new_mask

        filter_log.append({
            "axis": "var",
            "filter": key,
            "threshold": adata.uns["config"]["pp"]["filter_qc_var_var"][key],
            "n_removed": filtered_count
        })
    # #########################################################
    # Save filtered shape statistics and log processing time
    good_qc_view = adata[adata.obs[qc_key], adata.var[qc_key]]
    bad_qc_view = adata[~adata.obs[qc_key], ~adata.var[qc_key]]
    # ###############################
    # Update adata.uns["stats"] with shape info, appending if existing
    for key, shape in {
            "filtered_shape": [good_qc_view.shape[0], good_qc_view.shape[1]],
            "bad_qc_shape": [bad_qc_view.shape[0], bad_qc_view.shape[1]],
            }.items():
        if key in adata.uns["stats"]:
            if isinstance(adata.uns["stats"][key], list):
                adata.uns["stats"][key].append(shape)
            else:
                adata.uns["stats"][key] = [adata.uns["stats"][key], shape]
        else:
            adata.uns["stats"][key] = [shape]

    logger.info(
        f"Number of good quality (cells, genes): {good_qc_view.shape}; "
        f"bad quality: {bad_qc_view.shape}; "
        f"time: {time() - start_sub:.2f}s")
    # #########################################################
    # Return what is needed
    if return_filter_log and not apply_filters:
        return pd.DataFrame(filter_log)
    if return_filter_log and apply_filters:
        adata = adata[adata.obs["good_quality"], adata.var["good_quality"]].copy()
        return adata, pd.DataFrame(filter_log)
    if apply_filters:
        adata = adata[adata.obs["good_quality"], adata.var["good_quality"]].copy()
        return adata


def all_filters(
            adata: ad.AnnData,
            prot: bool = False,
            post_filter: bool = True
        ) -> ad.AnnData:
    """Runs all filtering criteria defined in ``adata.uns["config"]["pp"][keys]``.

    The function filters cells, genes, and quality control variables in an adata
    object based on the configuration provided in ``adata.uns["config"]["pp"]``.
    The keys used for filtering are ``filter_cells``, ``filter_genes``, and
    ``filter_qc_var``.

    NOTE:
        Deprecated, use the all_filters_tagging, it doesn't rely on the slow
        scanpy functions anymore.

    Args:
        adata (anndata.AnnData): Adata object that contains the dataset to be
            filtered.
        prot (bool, optional): Specifies if Citeseq data is included.
            Defaults to False.
        post_filter (bool, optional): Filters out empty cells and rows after
            general filtration. Defaults to True.

    Returns:
        anndata.AnnData:
            The filtered adata object. The function does not
            deepcopy the object, and the filtering is done inplace.

    Calls:
        filter_cells_multicall, filter_genes_multicall, get_shape_diff

    Tags:
        QC, config, obs, stats, var
    """
    start_sub = time()
    # #####################################################
    # Initialize stats dictionary if not already present
    if "stats" not in adata.uns.keys():
        adata.uns["stats"] = {}
    # #####################################################
    # Add raw_shape key to stats if it is not already present
    if "raw_shape" not in adata.uns["stats"].keys():
        adata.uns["stats"]["raw_shape"] = [adata.shape[0], adata.shape[1]]
    # #####################################################
    # Filter based on cells using the criteria in adata.uns["config"]["pp"]["filter_cells"]
    filter_cells_multicall(adata, **adata.uns["config"]["pp"]["filter_cells"])
    # #####################################################
    # Filter based on quality control variables for obs
    for key in adata.uns["config"]["pp"]["filter_qc_var_obs"]:
        shape_before = adata.shape
        # The key has to begin with "min_" or "max_" and the rest must be an obs key
        subset_direction, obs_key = key[:4], key[4:]

        # Check if the obs_key is in adata.obs
        if obs_key not in adata.obs.keys():
            logger.warning(
                f"Can't process qc obs variable key: {key}, because the key doesn't "
                "contain a valid obs key!")
            continue

        if "min_" == subset_direction:
            adata = adata[adata.obs[obs_key] > adata.uns["config"]["pp"]["filter_qc_var_obs"][key], :].copy()
        elif "max_" == subset_direction:
            adata = adata[adata.obs[obs_key] < adata.uns["config"]["pp"]["filter_qc_var_obs"][key], :].copy()
        else:
            logger.info(f"Can't process qc variable key: {key}, because no min_ or max_ in the string!")
            continue

        get_shape_diff(adata, f'pp_filter_qc_var_obs_{key}', shape_before, adata.shape)
    # #####################################################
    # After filtering cells, filter the genes using the criteria in adata.uns["config"]["pp"]["filter_genes"]
    filter_genes_multicall(adata, **adata.uns["config"]["pp"]["filter_genes"])
    # #####################################################
    # Filter based on quality control variables for variables (var)
    for key in adata.uns["config"]["pp"]["filter_qc_var_var"]:
        shape_before = adata.shape
        # The key has to begin with "min_" or "max_" and the rest must be a var key
        subset_direction, var_key = key[:4], key[4:]

        # Check if the var_key is in adata.var
        if var_key not in adata.var.keys():
            logger.info(f"Can't process qc var variable key: {key}, because the key doesn't contain a valid var key!")
            continue

        if "min_" == subset_direction:
            adata = adata[:, adata.var[var_key] > adata.uns["config"]["pp"]["filter_qc_var_var"][key]].copy()
        elif "max_" == subset_direction:
            adata = adata[:, adata.var[var_key] < adata.uns["config"]["pp"]["filter_qc_var_var"][key]].copy()
        else:
            logger.info(f"Can't process qc variable key: {key}, because no min_ or max_ in the string!")
            continue

        get_shape_diff(adata, f'pp_filter_qc_var_var_{key}', shape_before, adata.shape)
    # #####################################################
    # Save and print the shape after filtering along with the processing time
    adata.uns["stats"]["filtered_shape"] = [adata.shape[0], adata.shape[1]]
    # #####################################################
    # Apply post-filtering to remove empty cells and rows if post_filter is True
    if post_filter:
        adata = adata[np.array(adata.X.sum(1)).flatten() != 0, np.array(adata.X.sum(0)).flatten() != 0]

    logger.info(f"Shape after Filtering obs: {adata.shape}, time: {time() - start_sub}")
    return adata


# ###########################################################################################################
# Clustering
def run_pca_on_subset(
            adata: ad.AnnData,
            config: dict | None = None
        ) -> None:
    """Runs PCA on a subset of the adata object based on selected genes and
    adds the PCA results to the original adata object.

    Args:
        adata (anndata.AnnData): Adata object to be modified in-place.
        config (dict | None, optional): Configuration dictionary containing PCA
            parameters. Defaults to None.

    Returns:
        None:
            None (modifies adata in place)

    TODO:
        This is a temporary workaround due to bugs in the ``mask_var`` function
        from Rapids.

    Tags:
        clustering, config
    """
    if config is None:
        config = adata.uns["config"]

    # Subset the data based on the given genes
    adata_subset = adata[:, adata.var["genes_for_pca"].astype(bool)].copy()

    n_components = min(config["pp"]["pca"]["n_comps"], min(adata_subset.shape) - 1)
    # Run PCA
    rsf.pp.pca(
                adata_subset,  # mask_var="genes_for_pca",
                n_comps=n_components,
                **{k: v for k, v in config["pp"]["pca"].items() if k not in [
                    "svd_solver", "random_state", "return_info", "dtype", "copy",
                    "use_highly_variable", "method", "n_comps",
                ]})
    # Store PCA results back in the original object
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, n_components))
    adata.obsm["X_pca"][:, :adata_subset.obsm["X_pca"].shape[1]] = adata_subset.obsm["X_pca"]

    # Save variance explained ratio
    adata.uns["pca_variance_ratio"] = adata_subset.uns["pca"]["variance_ratio"]

    # Save the PCA components
    adata.varm["PCs"] = np.zeros((adata.n_vars, n_components))
    adata.varm["PCs"][adata.var["genes_for_pca"].astype(bool), :] = adata_subset.varm["PCs"]

    return adata


def cluster_SC_scanpy_like(  # noqa: C901
            adata: ad.AnnData,
            gene_subeset: list[str] = [],
            run_pca: bool = True,
            run_neighbors: bool = True,
            run_clustering: bool = True,
            run_umap: bool = True,
            hierarchical_clustering: bool = True,
        ) -> ad.AnnData:
    """
    Runs PCA, neighborhood graph, unsupervised clustering (leiden),
    and UMAP on an adata object.

    This function is designed to perform a sequence of preprocessing and
    clustering steps on single-cell
    RNA sequencing data stored in an adata object.

    It includes the following steps:
        - Principal Component Analysis (PCA),
        - neighborhood graph construction,
        - clustering using the Leiden algorithm,
        - Uniform Manifold Approximation and Projection (UMAP) for visualization.

    NOTE:
        Use ``adata = cluster_SC_scanpy_like(adata)`` because this function
        operates in-place, but it might updates the symlink for GPU computation.

    Args:
        adata (anndata.AnnData): Adata object.
        gene_subeset (list[str], optional): Gene set to run the PCA on. If not
            set, the function uses ``adata.uns["highly"]``. Defaults to [].
        run_pca (bool, optional): Whether to (re-)calculate PCA.
            Defaults to True.
        run_neighbors (bool, optional): Whether to (re-)calculate neighbors.
            Defaults to True.
        run_clustering (bool, optional): Whether to (re-)calculate unsupervised
            clustering. Defaults to True.
        run_umap (bool, optional): Whether to (re-)calculate UMAP.
            Defaults to True.
        hierarchical_clustering (bool, optional): If a post hierarchical
            clustering of the clusters should be performed. Defaults to True.

    Returns:
        anndata.AnnData:
            Returns the updated adata object with all the performed
            computations.

    Raises:
        AttributeError: If an invalid algorithm is specified for
            ``sc.pp.neighbors``.
        AttributeError: If an outdated or unsupported clustering algorithm is
            chosen.

    Calls:
        get_group_hierarchy, get_highly_variable, set_neighbors_n_pcs

    Called By:
        search_for_leiden_resolution, search_for_neighbor_params

    Tags:
        clustering, config, pipeline, visualization
    """
    # ############################################################
    # Get the config object from the adata object
    config = adata.uns["config"].copy()
    # ############################################################
    # Check for genes with zero counts; PCA cannot run on these genes
    if "counts" in adata.layers.keys():
        non_z_genes = np.array(adata.layers["counts"].sum(0) != 0).flatten()
        if sum(non_z_genes) != adata.shape[1]:
            adata = adata[:, non_z_genes].copy()
            logger.warning("There were empty genes, we removed them, otherwise PCA cannot run!")
    else:
        logger.warning(
                "NO Check for empty genes was performed. "
                "If an error occurs, please check for empty genes!!!")
    # ############################################################
    # Select the genes for PCA
    if isinstance(gene_subeset, list):
        if len(gene_subeset) == 0:
            gene_subeset = get_highly_variable(adata, return_adata=False, return_genes=True)
    elif isinstance(gene_subeset, str):
        if gene_subeset == "HVG":
            gene_subeset = get_highly_variable(adata, return_adata=False, return_genes=True)
            # print("hvg genes")
        elif gene_subeset == "all":
            gene_subeset = adata.var_names
            # print("all genes")
        else:
            logger.warning("Please provide a valid gene set or string for gene_subeset!")

    adata.var["genes_for_pca"] = adata.var_names.isin(gene_subeset)
    # ############################################################
    # RUN PCA
    if run_pca:
        # ############################################################
        # Adjust the number of principal components if necessary
        if config["pp"]["pca"]["n_comps"] is None:
            config["pp"]["pca"]["n_comps"] = len(gene_subeset) - 1
        if config["pp"]["pca"]["n_comps"] > len(gene_subeset):
            config["pp"]["pca"]["n_comps"] = len(gene_subeset) - 1
            logger.info(
                "ATTENTION!: The number of genes is less than pca_n_comps. We decrease it "
                f"to {len(gene_subeset)} to match the number of genes.")
        if config["general"]["use_GPU"] and config["pp"]["pca"]["method"] == "rapids":
            rsf.pp.pca(
                adata, mask_var="genes_for_pca",
                n_comps=min(config["pp"]["pca"]["n_comps"], min(adata.shape) - 1),
                **{k: v for k, v in config["pp"]["pca"].items() if k not in [
                    "svd_solver", "random_state", "return_info", "dtype", "copy",
                    "use_highly_variable", "method", "n_comps",
                ]})
        else:
            # NOTE: We fix the maximum of pcs to the minimum of the cells/genes - 1
            sc.pp.pca(
                adata, n_comps=min(config["pp"]["pca"]["n_comps"], min(adata.shape) - 1),
                random_state=config["general"]["seed"], mask_var="genes_for_pca")
        # ############################################################
        # Fix the explained variance calculation
        adata.uns["pca"]["variance_ratio"] = adata.uns["pca"]["variance"]
        adata.uns["pca"]["variance_ratio"][adata.uns["pca"]["variance_ratio"] < 1] = 0
        adata.uns["pca"]["variance_ratio"] = (
                adata.uns["pca"]["variance_ratio"] / sum(adata.uns["pca"]["variance_ratio"]))
        set_neighbors_n_pcs(adata)
        # Reset the config view, somehow it is overwritten
        config = adata.uns["config"].copy()
    # ############################################################
    # RUN NEIGHBORS
    if run_neighbors:
        # TODO: This is legacy, maybe remove it some day.
        if "transformer" not in config["pp"]["neighbors"]:
            if config["general"]["use_GPU"]:
                config["pp"]["neighbors"]["transformer"] = "rapids"
                adata.uns["config"] = config
                logger.warning(
                    'You enabled GPU support but '
                    'config["pp"]["neighbors"]["transformer"] was not set, so you likely use an old object '
                    'We will set to to the default "rapids" now!')
            else:
                config["pp"]["neighbors"]["transformer"] = "sklearn"
                adata.uns["config"] = config
                logger.warning(
                    'You disbled GPU support but '
                    'config["pp"]["neighbors"]["transformer"] was not set, so you likely use an old object '
                    'We will set to to the default "sklearn" now!')
        if (
                (config["general"]["use_GPU"]
                 and config["pp"]["neighbors"]["transformer"] == "rapids")):
            # print("Neighbors")
            rsf.pp.neighbors(
                adata, algorithm="brute",
                **{k: v for k, v in config["pp"]["neighbors"].items()
                    if k not in ["transformer", "knn", "method"]})
        else:
            if config["pp"]["neighbors"]["transformer"] == "rapids":
                logger.warning(
                    'You disabled GPU support but '
                    'config["pp"]["neighbors"]["transformer"] == "rapids" '
                    'We will set ti to the default "sklearn" now!')
                # Maybe not necessary but we set the original config also
                # adata.uns["config"]["pp"]["neighbors"]["transformer"] = "sklearn"
                config["pp"]["neighbors"]["transformer"] = "sklearn"
                adata.uns["config"] = config
            if config["pp"]["neighbors"]["method"] in ["umap", "gauss"]:
                if config["pp"]["neighbors"]["transformer"] in ["pynndescent", "sklearn", "rapids"]:
                    sc.pp.neighbors(adata, **config["pp"]["neighbors"])
                else:
                    raise AttributeError('Please choose a valid "transformer" of scanpy.pp.neighbors()!')
            else:
                raise AttributeError('Please choose a valid "method" of scanpy scanpy.pp.neighbors()!')
    # ############################################################
    # RUN UNSUPERVISED CLUSTERING
    if run_clustering:
        if config["general"]["cluster_algorithm"] == "leiden":
            if (
                    config["general"]["use_GPU"]
                    and config["tl"]["leiden"]["method"] == "rapids"):
                print("Leiden")
                rsf.tl.leiden(
                    adata, **{k: v for k, v in config["tl"]["leiden"].items() if k not in [
                        "copy", "directed", "restrict_to",
                        "adjacency", "partition_type", "obsp", "use_raw", "n_iterations",
                        "flavor", "method",
                    ]})
            else:
                # Ignore only this specific FutureWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            ".*default backend for leiden will be igraph instead of leidenalg.*"),
                        category=FutureWarning)
                    sc.tl.leiden(
                        adata,
                        **{k: v for k, v in config["tl"]["leiden"].items()
                            if k not in ["method"]})
            if hierarchical_clustering and len(adata.obs["leiden"].cat.categories) > 1:
                get_group_hierarchy(adata)
        # elif config["general"]["cluster_algorithm"] == "louvain":
        #     raise AttributeError(
        #         "Please stay up to date and don't use outdated algorithms! "
        #         "Read: https://www.nature.com/articles/s41598-019-41695-z")
        else:
            raise AttributeError("Please choose a scanpy tl.* algorithm like leiden!")
    # ##################################################
    # RUN UMAP
    if run_umap:
        if (
                config["general"]["use_GPU"]
                or config["tl"]["umap"]["method"] == "rapids"):
            rsf.tl.umap(adata,
                        **{k: v for k, v in config["tl"]["umap"].items()
                           if k not in ["gamma", "method"]})
        else:
            # Maybe not necessary but we set the original config also
            config["tl"]["umap"]["method"] = "umap"
            adata.uns["config"] = config

            sc.tl.umap(adata, **config["tl"]["umap"])

        # if config["general"]["save_umap_object"]["to_save"]:
        #     adata.uns[config["general"]["save_umap_object"]["key"]] = umap
    # ##################################################
    # TODO: Create elbow visualization for PCA variance ratio
    if not config["general"]["analysis_only"] and config["to_plot"]["cl"]["visualize_pca_variance_ratio"]:
        sc.pl.pca_variance_ratio(
            adata, n_pcs=config["pp"]["neighbors"]["n_pcs"] + 30,
            **{k: v for k, v in adata.uns["config"]["pl"]["pca_variance_ratio"].items() if k != "n_pcs"})

    return adata


def Run_all_prep_steps_clustering(
            adata: ad.AnnData,
            filter: bool = True
        ) -> ad.AnnData:
    """Run the preprocessing steps for gene expression data.

    This function performs a series of preprocessing steps on the provided adata
    object, including filtering genes, normalization, log-transforming, scaling,
    and regression. The steps are configurable based on the settings in
    ``adata.uns["config"]``. It also handles saving intermediate results in
    ``adata.layers`` and updating the preprocessing status in
    ``adata.uns["stats"]["preprocessors_ran"]``.

    Args:
        adata (anndata.AnnData): Adata object.
        filter (bool, optional): If True, performs exclusion of MT/Ribos/Pseudo
            genes. Defaults to True.

    Returns:
        anndata.AnnData:
            Updated adata object with the applied preprocessing
            steps.

    Raises:
        NotImplementedError: Raised if regression on CPU is attempted, as it is
            not implemented.

    Calls:
        calc_highly_variable_genes_unique_based, filter_genes_n_families,
        filter_genes_n_families_regex, get_shape_diff, hist_equalize_data,
        mark_highly_variable_genes

    TODO:
        - Test implementation of "filter_highly_variable" and "regress_out" for
          GPU, but note that the speed increase is significant only with
          datasets containing more than 500,000 cells.
        - Implement "regress_out" for CPU.

    Tags:
        config, normalization, pipeline, scaling, stats
    """
    # #######################################################
    # Check if the function has already been run to avoid re-execution
    if "Run_all_prep_steps_clustering" in adata.uns["stats"]["preprocessors_ran"]:
        logger.warning("You already ran the Run_all_prep_steps_clustering, skipping re-execution!")
        return adata
    # #######################################################
    # Retrieve the configuration object from the adata object
    config = adata.uns["config"]

    logger.info("#" * 80 + "\nStarted Raw Count Preprocessing\n" + "#" * 80)
    start = time()  # noqa: F841
    # #######################################################
    # Save the original counts for later use if specified in the configuration
    if config["general"]["save_counts"]:
        adata.layers["counts"] = adata.X.copy()
    # ################################################################
    # Perform gene filtering based on specified criteria if filter is True
    if filter:
        if config["exclude_gene_family_regex"]:
            logger.info("Running Regex gene exclusion")
            adata = filter_genes_n_families_regex(adata, config)
        else:
            adata = filter_genes_n_families(adata, config)
    # ################################################################
    # Placeholder for converting data to desired CPU/GPU object, currently commented out
    # adata = get_desired_cpu_gpu_object(adata)
    # #######################################################
    # Perform normalization steps based on the configuration
    start_sub = time()
    if config["general"]["hist_eq"]:
        adata.log1p()
        adata = hist_equalize_data(adata, config)
        # #######################################################
        # Save normalized log2 data if specified in the configuration
        if config["general"]["save_histeq_counts"]:
            adata.layers['log2norm_counts'] = adata.X.copy()
        # #######################################################
    else:
        if config["general"].get("save_log2_counts", False):
            adata.layers['log2_counts'] = adata.X.copy()
            adata.layers['log2_counts'].data = (
                np.log1p(adata.layers['log2_counts'].data))
        # TODO: Add conditions and logging for cases where normalization or log is not used
        if config["general"]["normalize_cell_size"]:
            sc.pp.normalize_total(adata, **config["pp"]["normalize_total"])
            # #######################################################
            # Save normalized counts for later use if specified in the configuration
            if config["general"]["save_norm_counts"]:
                adata.layers['norm_counts'] = adata.X.copy()
        # #######################################################
        if config["general"]["logarithmize"]:
            sc.pp.log1p(adata, **config["pp"]["log1p"])
            # ##########################################
            # Remove unnecessary 'log1p' entry from adata.uns if present
            if "log1p" in adata.uns.keys():
                del adata.uns["log1p"]
            # #######################################################
            # Save normalized log2 data for later use if specified in the configuration
            if config["general"]["save_normlog2_counts"]:
                adata.layers['log2norm_counts'] = adata.X.copy()
        # #######################################################
    logger.info(f"Normalization time: {time() - start_sub}")
    # #######################################################
    # Identify highly variable genes based on the configuration
    start_sub = time()
    mark_highly_variable_genes(adata)
    # adata = get_desired_cpu_gpu_object(adata)
    # #######################################################
    # Placeholder for saving raw data before filtering, regression, and scaling
    # raw = adata.to_AnnData().copy()
    # #######################################################
    # Filter highly variable genes if specified in the configuration
    if config["general"]["filter_highly_variable"]:
        shape_before = adata.shape
        if config["pp"]["highly_variable_genes"]["flavor"] == "ours":
            calc_highly_variable_genes_unique_based(adata)
        else:
            sc.pp.highly_variable_genes(adata, **config["pp"]["highly_variable_genes"])
        get_shape_diff(adata, "filter_highly_variable", shape_before, adata.shape)
    # #######################################################
    # Perform regression based on the configuration, with GPU implementation placeholder
    if config["general"]["regress_out"]:
        sc.pp.regress_out(adata, **config["pp"]["regress_out"])
        # #######################################################
        # Save regressed out data for later use if specified in the configuration
        if config["general"]["save_regress_out"]:
            adata.layers['regress_out'] = adata.X.copy()
    # #######################################################
    # Scale the data to unit variance and zero mean if specified in the configuration
    # TODO: Consider implementing a GPU-based scaling method
    if config["general"]["scale_data"]:
        logger.info("Scaling data...")
        sc.pp.scale(adata, **config["pp"]["scale"])  # zero_center=True, max_value=10)
        # #######################################################
        # Save scaled data for later use if specified in the configuration
        if config["general"]["save_scaled"]:
            adata.layers['scaled'] = adata.X.copy()
    # ###############################################################
    # Finalization of preprocessing, logging the time taken
    logger.info("#" * 80 + f"\nFinished RNA Preprocessing after {time() - start} seconds\n" + "#" * 80)
    adata.uns["stats"]["preprocessors_ran"].append("Run_all_prep_steps_clustering")
    return adata


# ###########################################################################################################
# Downstream
def downstream_preprocessing(
            adata: ad.AnnData,
            ref_dict: dict = {},
            calc_deg: bool = True,
            score_genes: bool = False,
            deg_key: str = "rank_genes_groups",
            use_whyever_default_scanpy_deg: bool = False,
            layer: str | None = "log2norm_counts",
        ) -> ad.AnnData:
    """Run downstream preprocessing for a single modality.

    NOTE:
        This function will overwrite the X with layer!

    This function handles the downstream preprocessing for a single modality of
    the provided adata object. It includes various steps such as calculating
    highly variable genes, updating marker gene dictionaries, scoring cell types
    cluster-wise, and optionally ranking gene groups.

    Args:
        adata (anndata.AnnData): Adata object containing the data to be
            processed.
        ref_dict (dict, optional): Dictionary of reference marker genes.
            Defaults to {}.
        calc_deg (bool, optional): If True, calculates the DEGs
            (rank_genes_groups). Defaults to True.
        score_genes (bool, optional): If True, scores genes based on predefined
            criteria. Defaults to False.
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        Defaults to "rank_genes_groups".
        layer (str | None, optional): The layer to compute the DEGs for.
            Defaults to "log2norm_counts".

    Returns:
        anndata.AnnData:
            The processed adata object with updated attributes.

    Raises:
        KeyError: If required keys are not found in the adata object.

    Calls:
        calc_DEGs, get_all_markers_from_ref_dict, get_highly_variable,
        get_valide_ref_dicts, ref_dict_sort_values_hvg_like, score_genes

    Called By:
        run_downstream

    TODO:
        Fix the division by 0 error, by skipping/fixing these.

    Tags:
        DEG, annotation, config, pipeline
    """
    start = time()
    # #########################################################
    # Retrieve configuration object from adata
    config = adata.uns["config"]
    cluster_key = config["general"]["cluster_algorithm"]
    # #########################################################
    # Set an empty dict at the default position, if not parsed
    # TODO: the unsetting of the config should be only temporary and
    #       People have to setup the config correcly!
    if len(ref_dict) == 0:
        if "ref_dict" not in adata.uns.keys():
            adata.uns["ref_dict"] = {}
            # Unset the config parameters
            config["to_plot"]["down"]["rank_genes_groups"]["marker_mod"] = []
            config["to_plot"]["down"]["cell_type_score_umap"]["marker_mod"] = []
            config["to_plot"]["down"]["heatmaps"]["marker_mod"] = []
            config["to_create"]["down"]["DEG_csv"]["marker_mod"] = []
            config["to_create"]["down"]["DEG_csv"]["marker_mod_parallel"] = []
            config["to_create"]["down"]["cluster_stats_csv"]["marker_mod"] = []
        if "ref_dict_upd" not in adata.uns.keys():
            adata.uns["ref_dict_upd"] = {}
            config["to_plot"]["down"]["heatmaps"]["marker_subs_mod"] = []
    # #########################################################
    # Determine if log2 normalized counts are available and set the active layer

    if layer in adata.layers.keys():
        adata.X = adata.layers[layer].copy()
    elif layer is None:
        # No layerchanges needed if None is parsed
        pass
    else:
        logger.info(f"Layer {layer} not found in adata.layers, using adata.X instead!")
        layer = None
    # #########################################################
    # Retrieve and store highly variable genes
    highly_variables = list(get_highly_variable(
        adata, return_adata=False, return_genes=True, sort_by_variance=True
    ))
    adata.uns["highly_variables"] = copy(highly_variables)
    # #########################################################
    # Extend marker genes if reference dictionary is provided
    if len(ref_dict) != 0:
        # Update marker gene dictionary with valid genes
        ref_dict_upd, ref_dict_upd_subs = get_valide_ref_dicts(
            adata, ref_dict, return_subset=True)
        # Sort the genes according to highly variability
        ref_dict_upd = ref_dict_sort_values_hvg_like(adata, ref_dict_upd)

        highly_variables.extend(
            get_all_markers_from_ref_dict(ref_dict_upd))
        adata.uns["ref_dict_upd"] = ref_dict_upd

        # Store updated subsets if available
        if ref_dict_upd_subs is not None:
            # Sort the genes according to highly variability
            ref_dict_upd_subs = ref_dict_sort_values_hvg_like(adata, ref_dict_upd_subs)

            adata.uns["ref_dict_upd_subs"] = ref_dict_upd_subs
        else:
            adata.uns["ref_dict_upd_subs"] = ref_dict_upd
    # #########################################################
    # Remove duplicates from the list of highly variable genes
    highly_variables = pd.unique(np.array(highly_variables))
    # #########################################################
    # Score cell type cluster-wise if required
    if calc_deg:
        # #########################################################
        # Score genes for each cluster using the reference dictionary
        if score_genes:
            logger.info("Run score genes...")
            start = time()

            for key in ref_dict_upd.keys():
                sc.tl.score_genes(
                    adata, gene_list=ref_dict_upd[key], score_name=key, ctrl_size=len(highly_variables),
                    n_bins=config["tl"]["score_genes"]["n_bins"],
                    gene_pool=highly_variables,
                    use_raw=config["tl"]["score_genes"]["use_raw"])
            logger.info(f"score genes took {time() - start:.2f}")
        # #########################################################
        # Calculate and update rank gene groups in the adata object
        if config["tl"]["rank_genes_groups"]["key_added"] in adata.uns.keys():
            logger.info(f'Deleting previously calculated {config["tl"]["rank_genes_groups"]["key_added"]} '
                        'and replace it.')
            del adata.uns[deg_key]

        logger.info("Run DEGs...")
        start = time()
        # TODO: Fix the division by 0 error, by skipping/fixing these
        if use_whyever_default_scanpy_deg:
            np.seterr(divide='ignore')
            sc.tl.rank_genes_groups(
                adata, groupby=cluster_key, layer=layer,
                **{k: v for k, v in config["tl"]["rank_genes_groups"].items() if k not in ["layer"]})
            np.seterr(divide=None)
            # Update the groups parameter with correct categories
            adata.uns[deg_key]["params"]["groups"] = adata.obs[cluster_key].cat.categories.tolist()
            # Exclude specific reference cluster if it's used for ranking
            if config["tl"]["rank_genes_groups"]["reference"] != "rest":
                adata.uns[deg_key]["params"]["groups"].remove(
                    config["tl"]["rank_genes_groups"]["reference"])
        else:
            calc_DEGs(
                    adata, n_cores=config["general"]["n_cores"])

        logger.info(f"DEG calculation took {time() - start:.2f}")

    # logger.info("#" * 80, f"\nFinished Downstream Preprocessing after {time() - start:.2f} seconds\n", "#" * 80)
    logger.info(f'{"#" * 80}\nFinished Downstream Preprocessing after {time() - start:.2f} seconds\n{"#" * 80}')
    return adata


'''
The results include one Folder and one HTML/PDF file, In the HTML/PDF file
is in the beginning a more detailed explaination.
For the sake of fast lookup I will also add the general explaination here,
this is also in the bd_rhap_ag_illert_scRNA_with_transgenes.html/pdf


The folder structure is as folowing:

├── DEGs - DEG csv files for all, filtered, unfiltered, up and down
│   ├── all_...filtered.csv
│   └── per_cluster
│       ├── DEG csv files per cluster
│       ├── down
│       │   └── cluster_..._rna.csv
│       └── up
│           └── cluster_..._rna.csv
├── figures - All Figures.
│   └── downstream
│       ├── Dotplots
│       │   ├── DEG
│       │   │   └── dotplot_cluster_DEG_dotplot_top10_3.pdf
│       │   └── markers
│       │       ├── dotplot_rna_cluster_marker_expression_selected_only_group_activated_cells.pdf
│       │       └── dotplot_rna_cluster_marker_expression_selected_only_group_Tumor_cell.pdf
│       ├── UMAP
│       │   ├── DEG_plots
│       │   │   └── umap_..._DEGs_discretized.pdf
│       │   ├── HVG_plots
│       │   │   └── umap_top10_highly_variable_discretized.pdf
│       │   ├── marker_plots
│       │   │   └── umap_..._markers_discretized.pdf
│       │   ├── QC_plots
│       │   │   └── umap_qc.pdf
│       │   └── umap_leiden_clustering.pdf
│       └── Violinplots
│           ├── DEG
│           │   ├── stacked_violin_cluster_DEG_dotplot_cluster_....pdf
│           └── markers
│               └── stacked_violin_stacked_violin_rna_cluster_marker_expression_group_....pdf
└── markers
    ├── cell_type_markers.csv
    └── cell_type_markers_non_overlap.csv  - if there is overlap


The HTML/PDF files are derived from the analysis and include a cell type
estimation section, as well as a section displaying all figures in the folders
with their respective file paths.

This includes key elements such as DEGs, marker genes for cell types, and
Highly Variable Genes (HVG) plotted in UMAP space, as dot plots, and as stacked
violin plots.
You may not be interested in all of them; this depends on your preference.
Choose the ones you prefer, and for the next analysis, I can provide only those.

I acknowledge that some file names are long and overly specific; this is a work
in progress, and I apologize for any inconvenience. Suggestions are welcome.

If you encounter issues opening the .csv files, I can provide them in .xlsx
format. If you have problems with .pdf images, I can convert them to .png.
Opening multiple images simultaneously may slow down your system.
# ##########################################################################
# The following section is starting with
- UMAPs:
    - For the Clustering, the condition Density, the QC.
      NOTE:
        The most Measures in the QC are for testing purposes.
        The most interesting are the
            - n_genes: for total number of genes,
            - n_counts: for total number of UMI counts,
            - n_unique: for total number of unique UMI counts,
            - pct_MT: The percentage of mitochondrial gene expression,
            - pct_RBS: The percentage of ribosomal gene expression,
            - pct_MTRBS: The percentage of Mitoribosomal gene expression,
            - pct_HG: The percentage of Hemoglobin gene expression,
            - pct_IG: The percentage of Hemoglobin gene expression,
            - pct_gtex, pct_hpa, pct_hrt, : Housekeeping genesets for Human from Samples,
            - pct_kjin, pct_cellminer, : Housekeeping genesets for Human from Cellcluture,
            - pct_IG_genes: The percentage of Immunoglobulin gene expression,
            - pct_TR_genes: The percentage of T-cell Receptor gene expression,
            - pct_protein_coding: The percentage of protein-coding genes,
            - pct_processed_transcripts: The percentage of processed Transcripts, so non-protein Genes but processed,
            - pct_TEC_genes: Unvalidated genes/pseudogenes
            - pct_pseudogenes: The percentage of pseudogenes,

     If you see a pattern in one of the others,
     let me know so we can further investigate and maybe refine the gene set.

    - Then we have DEG UMAPs one time
        - in discretized: Log-expression greater than 1 is accounted for 1. This makes it easier to
          spot the cells actually expressing the gene.
        - in continuous: Log-expression. This only makes sense to look at if you have a gene of interest,
          because it is usually hard to spot anything.

    - The same for the Marker Genes.
      Here we have a mixture of cell types and the AML markers which are a combination of
      the genes you mutated in the experiment and General AML markers

    - The same for the Highly variable Genes (HVG)

- Dotplots:
    - One for the DEGs per Cluster (Due to the large size it is split into a max of 30 genes)
    - One for the Marker Genes

- Stacked Violin Plots:
    - Basically the same as the Dotplots but with violins.


Folder Structure and Outputs
# #################################
UMAP Plots: Saved in /downstream/UMAP/
    Leiden and Classification Clustering UMAP: Shows the unsupervised
    clustering of cells in UMAP space.
    Sample/Condition Density UMAP: Visualizes the density of cells for
    each sample or condition.
    Quality Control UMAP: Displays quality control metrics for cells.
    DEG UMAPs: Visualizes gradients over cells based on DEGs, separated
    into upregulated and downregulated genes.
    Marker UMAPs: Displays discrete and continuous UMAP plots for all or
    subsetted cell type markers.
# ################
These plots provide a visual representation of how cells are distributed
and clustered in UMAP space, highlighting important features like
clustering, sample density, and expression gradients.
# #################################
Dotplots: Saved in /downstream/Dotplots/
    DEG Dotplots: Shows dotplots for DEGs per cluster.
    Marker Dotplots: Displays dotplots for markers per cluster.
# ################
Dotplots help to visualize the expression of selected genes across
different clusters, indicating which clusters express certain genes more
prominently.
# #################################
Violin Plots: Saved in /downstream/Violinplots/
    DEG Violinplots: Shows stacked violin plots for DEGs per cluster.
    Marker Violinplots: Displays stacked violin plots for markers per
    cluster.
# ################
Violin plots depict the distribution of gene expression levels across
clusters, allowing for a comparison of expression intensity between
clusters.
# #################################
Rank Genes Groups Plots: Saved in /downstream/DEGs/
    Cluster DEG Plots: Displays DEGs for each cluster.
# ################
These plots highlight the most significant DEGs for each cluster,
helping to identify key genes that define different cell populations.
# #################################
Heatmaps: Saved in /downstream/heatmaps/
    Marker Gene Heatmaps: Displays heatmaps of marker genes.
    DEG Heatmaps: Visualizes heatmaps of DEGs across clusters.
# ################
Heatmaps provide an overview of gene expression patterns across all
cells or clusters, showing how certain genes are upregulated or
downregulated across different conditions.
# #################################
CSV/excel Outputs: Saved in /downstream/DEGs/ and other relevant
directories.
    DEG CSVs: Contains CSV files with DEGs, marker gene and cluster statistics.
# ################
These CSV files contain valuable data about DEGs and cluster statistics,
providing detailed information for further analysis and interpretation.
# ################
The ``run_downstream`` function is designed to perform differential
expression gene (DEG) analysis and generate various visualizations,
based on configuration settings stored in the ``adata`` object. Here's an
explanation of how this function decides whether to plot or not plot
different visualizations and how it handles the creation of DEG files,
based on the provided configuration parameters:

# ## Configuration Parameters Overview
- **``adata.uns["config"]["to_plot"]``**: Controls which plots are
  created.
- **``adata.uns["config"]["to_create"]``**: Controls which files (like
  DEG CSVs) are generated.
- **``deg_key``**: Specifies the key for ranking gene groups.
- **``rerun_DEG``**: Determines whether to rerun DEG calculations.
- **``with_figure_explaination``**: Decides if figure explanations are
  included in the logs.
- **``disable_sc_verbosity``**: Controls whether the verbosity of Scanpy
  (sc) functions is suppressed.

# ## Plotting Logic

1. **UMAP Plots**:
   - **Condition**: ``adata.uns["config"]["to_plot"]["down"]["umap"]``
   - If true, UMAP visualizations are generated.
   - Further sub-configurations:
     - **``adata.uns["config"]["to_plot"]["down"]["visualize_leiden_umap"]``**:
       Controls the generation of UMAPs for leiden clustering.
     - **``adata.uns["config"]["to_plot"]["down"]["visualize_embedding_density_umap"]``**:
       Controls UMAPs visualizing sample density.
     - **``adata.uns["config"]["to_plot"]["down"]["visualize_qc_umap"]``**:
       Controls UMAPs for quality control visualizations.
     - **``adata.uns["config"]["to_plot"]["down"]["gradients_umap"]["include_DEG_variables"]``**:
       Determines if DEG variables are included in UMAP visualizations.
     - **``adata.uns["config"]["to_plot"]["down"]["gradients_umap"]["include_marker"]``**:
       Controls whether markers are visualized in UMAPs (e.g., "all",
       "subset", "ref_dict").
     - **``adata.uns["config"]["to_plot"]["down"]["gradients_umap"]["include_highly_variable"]``**:
       Controls the inclusion of highly variable genes in UMAPs.

2. **Dotplots**:
   - **Condition**: ``adata.uns["config"]["to_plot"]["down"]["cluster_dotplot"]["visualize"]``
   - If true, dot plots for DEG and marker genes per cluster are
     generated.
   - Further sub-configurations:
     - **``adata.uns["config"]["to_plot"]["down"]["cluster_dotplot"]["n_DEGs"]``**:
       Specifies the number of DEGs to visualize in dot plots.
     - **``adata.uns["config"]["to_plot"]["down"]["cluster_dotplot"]["marker_mod"]``**:
       Controls the marker modules visualized in dot plots.
     - **``adata.uns["config"]["to_plot"]["down"]["cluster_dotplot"]["marker_subs_mod"]``**:
       Controls subsets of marker modules visualized in dot plots.

3. **Violinplots**:
   - **Condition**: ``adata.uns["config"]["to_plot"]["down"]["cluster_violins"]["visualize"]``
   - If true, stacked violin plots for DEG and marker genes per cluster
     are generated.
   - Further sub-configurations:
     - **``adata.uns["config"]["to_plot"]["down"]["cluster_violins"]["n_DEGs"]``**:
       Specifies the number of DEGs to visualize in violin plots.
     - **``adata.uns["config"]["to_plot"]["down"]["cluster_violins"]["marker_mod"]``**:
       Controls the marker modules visualized in violin plots.
     - **``adata.uns["config"]["to_plot"]["down"]["cluster_violins"]["marker_subs_mod"]``**:
       Controls subsets of marker modules visualized in violin plots.

4. **Rank Genes Groups Plots**:
   - **Condition**: ``adata.uns["config"]["to_plot"]["down"]["rank_genes_groups"]["visualize"]``
   - If true, visualizations for DEGs per cluster are created.
   - Further sub-configurations:
     - **``adata.uns["config"]["to_plot"]["down"]["rank_genes_groups"]["marker_mod"]``**:
       Specifies the marker modules included in rank genes groups plots.

5. **Heatmaps**:
   - **Condition**: ``adata.uns["config"]["to_plot"]["down"]["heatmaps"]["visualize"]``
   - If true, heatmaps for marker genes and DEGs are generated.
   - Further sub-configurations:
     - **``adata.uns["config"]["to_plot"]["down"]["heatmaps"]["marker_mod"]``**:
       Specifies the marker modules visualized in heatmaps.
     - **``adata.uns["config"]["to_plot"]["down"]["heatmaps"]["marker_subs_mod"]``**:
       Controls subsets of marker modules visualized in heatmaps.
     - **``adata.uns["config"]["to_plot"]["down"]["heatmaps"]["DEG_mod"]``**:
       Specifies the DEG modules visualized in heatmaps.

# ## DEG and CSV Creation Logic

1. **DEG CSVs**:
   - **Condition**: ``adata.uns["config"]["to_create"]["down"]["DEG_csv"]["create"]``
   - If true, the function saves the DEGs to CSV files.

2. **Marker Gene CSVs**:
   - **Condition**: ``adata.uns["config"]["to_create"]["down"]["marker_csv"]["create"]``
   - If true, the function saves marker gene sets to CSV files.

3. **Cluster Stats CSVs**:
   - **Condition**: ``adata.uns["config"]["to_create"]["down"]["cluster_stats_csv"]["create"]``
   - If true, cluster statistics are saved to CSV files.
    However, this part of the code is currently inactive, due to maintainance.

# ## Explanation of the Plotting Process

- The function checks the ``adata.uns["config"]["to_plot"]`` and
  ``adata.uns["config"]["to_create"]`` settings to determine which plots
  and DEG files need to be generated.
- If ``with_figure_explaination`` is true, it logs detailed explanations
  about the figures being generated.
- The actual generation of plots involves various helper functions (e.g.,
  ``sc.pl.umap``, ``plot_per_group_DEG_dotplot_n_gene_dendrogram``),
  which are called depending on the configuration settings.

# ## Conclusion
The ``run_downstream`` function is heavily driven by configuration
parameters within ``adata``. It decides whether to plot or not based on
the ``adata.uns["config"]["to_plot"]`` settings and whether to generate
DEG and marker gene CSV files based on the ``adata.uns["config"]["to_create"]``
settings. The verbosity and logging behavior are also adjustable via
function parameters, allowing for flexible output depending on the
user's needs.
'''


def run_downstream(
            adata: ad.AnnData,
            ref_dict: dict = {},
            part: str = "/downstream/",
            rerun_DEG: bool = False,
            deg_key: str = "rank_genes_groups",
            with_figure_explaination: bool = True,
            disable_sc_verbosity: bool = True,
            layer: str | None = "log2norm_counts",
        ) -> ad.AnnData:
    """Run the DEG calculations and plot/save all desired visualisations.

    The DEGs are calculated based on the
    adata.uns["config"]["general"]["cluster_algorithm"] classes. The plots are
    shown/saved based on the adata.uns["config"]["to_plot"]. The DEG and stat
    CSVs are created base on adata.uns["config"]["to_create"]

    Args:
        adata (anndata.AnnData): Adata object post clustering.
        ref_dict (dict, optional): Reference dictionary for plotting and DEG
            calculation. Defaults to {}.
        part (str, optional): Path to save the output.
            Defaults to "/downstream/".
        rerun_DEG (bool, optional): Flag to rerun DEG calculations.
            Defaults to False.
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        with_figure_explaination (bool, optional): Include figure explanations
            in logs. Defaults to True.
        disable_sc_verbosity (bool, optional): Disable scanpy verbosity.
            Defaults to True.
        layer (str | None, optional): The layer to compute the DEGs for.
            Defaults to "log2norm_counts".

    Returns:
        anndata.AnnData:
            The processed adata object.

    Calls:
        continuos_umap_helper, create_group_stats_df, discrete_umap_helper,
        downstream_preprocessing, get_DEG_gene_csvs, get_DEGs_per_group,
        get_adata_by_mod, get_colormap, get_highly_variable,
        get_print_highlighter, get_save_path, plot_embedding_density,
        plot_mediods_heatmap, plot_per_group_DEG_dotplot_n_gene_dendrogram,
        plot_per_group_DEG_umaps, plot_per_group_stacked_violins,
        plot_ref_dotplots, plot_ref_stacked_violins, plot_umap,
        save_genesets_to_csv, setup_image_folder

    TODO:
        split run_downstream into run and analysis

    Tags:
        DEG, annotation, config, pipeline, visualization
    """
    if disable_sc_verbosity:
        old_sc_verbosity = copy(sc.settings.verbosity)
        sc.settings.verbosity = 0
    # #############################################################################################
    # TODO: maybe use show_all everywhere to determin if images are saved and/or showed
    # TODO: unify order of use_xxx and config["..."] / config["..."] and use_xxx
    # TODO: split all separate plottings to functions, to be able to call them also from outside this function
    # #############################################################################################
    # Get the printing highlighters
    highlighers = get_print_highlighter(120, 3)
    # #############################################################################################
    # Fix the backslashes for the part, needed for correct folder saving
    if part[0] != "/":
        part = "/" + part
    if part[-1] != "/":
        part = part + "/"
    # #############################################################################################
    # Setup
    # Get the config object of the adata
    config = adata.uns["config"]
    # #############################################################################################
    # Setting log1p normalized counts as X
    if layer not in adata.layers.keys():
        logger.warning(f"Layer {layer} not found in adata.layers, trying log2norm_counts instead...")
        # TODO: Maybe add a idiotproove sanity check here ?!?
        # if adata.X.min() >= 0 and adata.X.max() > 100:
    elif "log2norm_counts" in adata.layers:
        adata.X = adata.layers["log2norm_counts"].copy()
        logger.warning("Setting the X to log2norm_counts counts.")
    else:
        logger.warning(
            "You have no log2norm_counts in the layers, It's your responsibility "
            "not to ensure the X is log2 normalized")
    # #############################################################################################
    # Create a new folder for the images
    setup_image_folder(config, part)
    # #########################################
    # get variable for both modalitys to check this only once
    # #########################################################
    # Get downstream preprocessed data
    # ############
    # get the higly variable genes ranked
    # TODO: calculate this only if necessary and generate variable for that
    #       may still needs adjustments
    if (
            config["to_plot"]["down"]["cluster_violins"]["visualize"]
            or config["to_plot"]["down"]["cell_type_score_umap"]["visualize"]
            or config["to_create"]["down"]["DEG_csv"]["create"]):
        calc_deg = True
    else:
        calc_deg = False
    # Check if the deg_key is already there and the rerun flag is not set
    if deg_key in adata.uns.keys() and not rerun_DEG:
        calc_deg = False
    # ############
    adata = downstream_preprocessing(
        adata, ref_dict=ref_dict,
        calc_deg=calc_deg,
        deg_key=deg_key, layer=None)  # We processed the layer already
    # Get a dataframe for each of the modalities, with DEGs calculated only for the modality
    start = time()
    # #############################################################################################
    # #############################################################################################
    # #############################################################################################
    # create the donwstream plots
    # #############################################################################################
    # #############################################################################################
    # #############################################################################################
    # #######################################################################
    # #######################################################################
    # Create the UMAPS
    # #######################################################################
    # #######################################################################
    if config["to_plot"]["down"]["umap"]:
        this_part = part + "UMAP/"
        # Create a new folder for the images
        this_fig_path = setup_image_folder(config, this_part, return_path=True)
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info("This set of figures visualizes Features in the UMAP Space")
            logger.info(f'The figures are saved in {this_fig_path}')
            logger.info(highlighers[0])
        # ########################################
        # Visualize the leiden and Classification clustering
        # ########################################
        if config["to_plot"]["down"]["visualize_leiden_umap"]:
            neighbor_key = config["general"]["cluster_algorithm"]
            save_path = get_save_path(
                f'_{neighbor_key}_clustering.pdf', config, this_part)

            if with_figure_explaination:
                logger.info(highlighers[1])
                logger.info(f'This Figure shows the unsupervised {neighbor_key}'
                            " Clustering, ")
                logger.info(f'it is saved as: umap{save_path}')
                logger.info(highlighers[1])

            with rc_context({'figure.figsize': (10, 10)}):
                # plot_umap(
                sc.pl.umap(
                    adata, color=neighbor_key,
                    legend_loc='on data', save=save_path, cmap=None,
                    **{k: v for k, v in config["pl"]["umap"].items() if k not in ["legend_loc", "cmap"]})
        # ########################################
        # Visualize the Density of the Samples
        # ########################################
        if config["to_plot"]["cl"]["visualize_embedding_density_umap"]:
            sample_keys = config["general"]["sample_n_condition_keys"]
            # ##################
            # If there is one key as a string, convert it to list
            if isinstance(sample_keys, str):
                sample_keys = [sample_keys]
            # ##################
            for sample_key in sample_keys:
                this_part = part + "UMAP/density_plots/"
                save_path = get_save_path(
                    '.pdf', config, this_part)

                if with_figure_explaination:
                    logger.info(highlighers[1])
                    logger.info("This Figure shows Density of cells for each Sample or Condition")
                    logger.info(f'it is saved as: umap_density_{sample_key}_{save_path}')
                    logger.info(highlighers[1])

                plot_embedding_density(
                        adata, groupby=sample_key, save_path=save_path)
        # ########################################
        # Visualize QC params
        # ########################################
        if config["to_plot"]["down"]["visualize_qc_umap"]:
            this_part = "/QC/UMAP/QC_plots/"
            save_path = get_save_path(
                "_qc.pdf", config, this_part)

            if with_figure_explaination:
                logger.info(highlighers[1])
                logger.info("This Figure shows Quality Control for Cells")
                logger.info(f'it is saved as: umap_{save_path}')
                logger.info(highlighers[1])

            # plot_umap(
            sc.pl.umap(
                adata, color=config["general"]["keys_for_qc"]["obs"],
                save=save_path, cmap=get_colormap(cmap=config["pl"]["umap"]["cmap"], fade_alpha=True),
                **{k: v for k, v in config["pl"]["umap"].items() if k not in ["cmap"]})
        # ########################################
        # Visualize the gradients over the cells by umap projection
        # ########################################
        if "rna" in config["to_plot"]["down"]["gradients_umap"]["include_DEG_variables"]:
            this_part = part + "UMAP/DEG_plots/"
            plot_per_group_DEG_umaps(
                adata, top_n_genes=config["to_plot"]["down"]["gradients_umap"]["num_DEGs"],
                part=this_part, deg_key=deg_key,
                with_figure_explaination=True,
                highlighers=highlighers[1],
                **config["to_plot"]["down"]["gradients_umap"]["plot_per_group_DEG_umaps"])
        # ################################
        # All Cell type marker discrete/gradients
        this_part = part + "UMAP/marker_plots/"
        if config["to_plot"]["down"]["gradients_umap"]["include_marker"] == "all":
            markers = np.intersect1d(
                        config["to_plot"]["down"]["gradients_umap"]["include_mod_variables"]["rna"],
                        adata.var_names)
            if len(markers) > 0:
                if config["to_plot"]["down"]["gradients_umap"]["discretize"]:
                    discrete_umap_helper(
                            adata, keys=markers,
                            part=this_part, with_figure_explaination=with_figure_explaination,
                            name="all_markers", highlighers=highlighers[2], config=config)
                if config["to_plot"]["down"]["gradients_umap"]["continuos"]:
                    continuos_umap_helper(
                            adata, keys=markers,
                            part=this_part, with_figure_explaination=with_figure_explaination,
                            name="all_markers", highlighers=highlighers[2], config=config)
        # ################################
        # Cell type marker discrete/gradients subsetted
        elif config["to_plot"]["down"]["gradients_umap"]["include_marker"] == "subset":
            markers = np.intersect1d(config["to_plot"]["down"]["marker_subset"], adata.var_names)
            if len(markers) > 0:
                if config["to_plot"]["down"]["gradients_umap"]["discretize"]:
                    discrete_umap_helper(
                            adata, keys=markers,
                            part=this_part, with_figure_explaination=with_figure_explaination,
                            name="subsetted_markers", highlighers=highlighers[2], config=config)
                if config["to_plot"]["down"]["gradients_umap"]["continuos"]:
                    continuos_umap_helper(
                            adata, keys=markers,
                            part=this_part, with_figure_explaination=with_figure_explaination,
                            name="subsetted_markers", highlighers=highlighers[2], config=config)
        # ################################
        # Cell type marker discrete/gradients per celltype from ref_dict
        elif config["to_plot"]["down"]["gradients_umap"]["include_marker"] == "ref_dict":
            for key, values in ref_dict.items():
                markers = np.intersect1d(
                        values[:config["to_plot"]["down"]["gradients_umap"]["num_ref_genes"]],
                        adata.var_names)
                if len(markers) > 0:
                    if config["to_plot"]["down"]["gradients_umap"]["discretize"]:
                        discrete_umap_helper(
                                adata, keys=markers,
                                part=this_part, with_figure_explaination=with_figure_explaination,
                                name=f'{key}_markers', highlighers=highlighers[2], config=config)
                    if config["to_plot"]["down"]["gradients_umap"]["continuos"]:
                        continuos_umap_helper(
                                adata, keys=markers,
                                part=this_part, with_figure_explaination=with_figure_explaination,
                                name=f'{key}_markers', highlighers=highlighers[2], config=config)
        # ################################
        # Highly variable genese discrete/gradients
        this_part = part + "UMAP/HVG_plots/"
        if config["to_plot"]["down"]["gradients_umap"]["include_highly_variable"]:
            num_high = config["to_plot"]["down"]["gradients_umap"]["num_highly_variable"]
            markers = np.intersect1d(get_highly_variable(
                    adata, return_adata=False, return_genes=True, sort_by_variance=True
                    )[:num_high], adata.var_names)
            if len(markers) > 0:
                if config["to_plot"]["down"]["gradients_umap"]["discretize"]:
                    discrete_umap_helper(
                            adata, keys=markers,
                            part=this_part, with_figure_explaination=with_figure_explaination,
                            name=f'top{num_high}_highly_variable', highlighers=highlighers[2], config=config)
                if config["to_plot"]["down"]["gradients_umap"]["continuos"]:
                    continuos_umap_helper(
                            adata, keys=markers,
                            part=this_part, with_figure_explaination=with_figure_explaination,
                            name=f'top{num_high}_highly_variable', highlighers=highlighers[2], config=config)
            # general_markers.extend()
    # #######################################################################
    # #######################################################################
    # Create dotplots
    # #######################################################################
    # #######################################################################
    if config["to_plot"]["down"]["cluster_dotplot"]["visualize"]:
        this_part = part + "Dotplots/"
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info("This Shows Dotplots for the DEGs and/or Markers per Cluster.")
            logger.info(f'The figures are saved in {this_part}')
            logger.info(highlighers[0])
        # ################################
        # DEG Dotplots
        for top_n in config["to_plot"]["down"]["cluster_dotplot"]["n_DEGs"]:
            this_part = part + "Dotplots/DEG/"
            if with_figure_explaination:
                logger.info(highlighers[1])
                logger.info("This Shows Dotplots for the DEGs per Cluster.")
                logger.info("NOTE: If DEGs are overlapping, they are shown only once and ")
                logger.info("      if the number of genes per Cluster is not always the same and you want it ")
                logger.info("      (or vice versa), please ask your Collaboration partner of choice")
                logger.info(f'The figures are saved in {this_part}')
                logger.info(highlighers[1])
            plot_per_group_DEG_dotplot_n_gene_dendrogram(
                    adata, config, img_suffix="",
                    part=this_part, top_n_genes=top_n,
                    with_figure_explaination=with_figure_explaination,
                    **(config["to_plot"]["down"]["cluster_dotplot"]
                        ["plot_per_group_DEG_dotplot_n_gene_dendrogram"]))
        # ################################
        # Marker Dotplots
        # Check if any key is used
        marker_vis_keys = [
                'marker_mod', 'marker_mod_per_cluster',
                'marker_subs_mod', 'marker_subs_mod_per_cluster']
        if max([len(config["to_plot"]["down"]["cluster_dotplot"][k]) for k in marker_vis_keys]) != 0:
            this_part = part + "Dotplots/markers/"
            if with_figure_explaination:
                logger.info(highlighers[1])
                logger.info("This Shows Dotplots for the markers per Cluster.")
                logger.info("NOTE: It is possible to subset these and you want it")
                logger.info("      please ask your Collaboration partner of choice.")
                logger.info(f'The figures are saved in {this_part}')
                logger.info(highlighers[1])

            plot_ref_dotplots(
                    adata, part=this_part,
                    with_figure_explaination=with_figure_explaination)
    # #######################################################################
    # #######################################################################
    # Create violinplots
    # #######################################################################
    # #######################################################################
    if config["to_plot"]["down"]["cluster_violins"]["visualize"]:
        this_part = part + "Violinplots/"
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info("This Shows Stacked Violinplots for the DEGs and/or Markers per Cluster.")
            logger.info(f'The figures are saved in {this_part}')
            logger.info(highlighers[0])
        # ################################
        # DEG Violinplots
        for top_n in config["to_plot"]["down"]["cluster_violins"]["n_DEGs"]:
            this_part = part + "Violinplots/DEG/"
            if with_figure_explaination:
                logger.info(highlighers[1])
                logger.info("This Shows Stacked Violinplots for the DEGs per Cluster.")
                logger.info("NOTE: If DEGs are overlapping, they are shown only once and ")
                logger.info("      if the number of genes per Cluster is not always the same and you want it ")
                logger.info("      (or vice versa), please ask your Collaboration partner of choice")
                logger.info(f'The figures are saved in {this_part}')
                logger.info(highlighers[1])
            plot_per_group_stacked_violins(
                    adata, config=None, highly_variable=False,
                    top_n_genes=20, part=this_part,
                    deg_key=deg_key,
                    with_figure_explaination=True,
                    **config["to_plot"]["down"]["cluster_violins"][
                            "plot_per_group_stacked_violins"])
        # ################################
        # Marker Violinplots
        # Check if any key is used
        marker_vis_keys = []
        if "marker_mod" in config["to_plot"]["down"]["cluster_violins"].keys():
            marker_vis_keys.append('marker_mod')
        if "marker_subs_mod" in config["to_plot"]["down"]["cluster_violins"].keys():
            marker_vis_keys.append('marker_subs_mod')

        if sum([len(config["to_plot"]["down"]["cluster_violins"][k]) for k in marker_vis_keys]) != 0:
            this_part = part + "Violinplots/markers/"
            if with_figure_explaination:
                logger.info(highlighers[1])
                logger.info("This Shows Stacked Violinplots for the markers per Cluster.")
                logger.info("NOTE: It is possible to subset these and you want it")
                logger.info("      please ask your Collaboration partner of choice.")
                logger.info(f'The figures are saved in {this_part}')
                logger.info(highlighers[1])

            plot_ref_stacked_violins(
                    adata, part=this_part,
                    with_figure_explaination=with_figure_explaination)
    # ##############################################################
    # #######################################################################
    # #######################################################################
    # Create DEG plots
    # #######################################################################
    # #######################################################################
    # create DEG plots per cluster
    if config["to_plot"]["down"]["rank_genes_groups"]["visualize"]:
        this_part = part + "rank_genes_groups/"
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info("This Shows the DEGs per Cluster.")
            logger.info(f'The figures are saved in {this_part}')
            logger.info(highlighers[0])
        if "rna" in config["to_plot"]["down"]["rank_genes_groups"]["marker_mod"]:
            save_path = get_save_path("_rna_cluster_DEG.pdf", config, this_part)
            sc.pl.rank_genes_groups(adata, save=save_path,
                                    **{k: v for k, v in config["pl"]["rank_genes_groups"].items()})
    # #######################################################################
    # #######################################################################
    # Create the Heatmaps
    # #######################################################################
    # #######################################################################
    # create marker gene heatmap plots
    # logger.debug("11", adata.uns["rank_genes_groups"].keys())
    if config["to_plot"]["down"]["heatmaps"]["visualize"]:
        logger.info("Visualizing: heatmaps")
        this_part = part + "heatmaps/"
        # #####################################
        # Marker genes
        for mod in config["to_plot"]["down"]["heatmaps"]["marker_mod"]:
            view = get_adata_by_mod(adata=adata, mod=mod)  # , adata_r=adata_r, adata_p=adata_p)
            geneset = view.uns["ref_dict_upd"]
            save_path = get_save_path(
                    f'_{mod}_"{config["to_plot"]["down"]["heatmaps"]["title"]["marker"]}.pdf',
                    config, this_part)
            plot_mediods_heatmap(
                    view, ref=geneset, part=part,
                    **config["to_plot"]["down"]["heatmaps"]["medioid_params"])
        if len(config["to_plot"]["down"]["marker_subset"]):
            for mod in config["to_plot"]["down"]["heatmaps"]["marker_subs_mod"]:
                view = get_adata_by_mod(adata=adata, mod=mod)  # , adata_r=adata_r, adata_p=adata_p)
                geneset = config["to_plot"]["down"]["marker_subset"]
                save_path = get_save_path(
                        f'_{mod}_{config["to_plot"]["down"]["heatmaps"]["title"]}.pdf',
                        config, part)
                plot_mediods_heatmap(
                        view, ref=geneset, part=part,
                        **config["to_plot"]["down"]["heatmaps"]["medioid_params"])
        else:
            logger.info(
                'You did not provide markers in config["to_plot"]["down"]'
                '["marker_subset"], skipping selected marker heatmaps!')
        # #####################################
        # DEGs
        for mod in config["to_plot"]["down"]["heatmaps"]["DEG_mod"]:
            view = get_adata_by_mod(adata=adata, mod=mod)  # , adata_r=adata_r, adata_p=adata_p)
            DEG_params = config["to_plot"]["down"]["heatmaps"]["DEG_params"]
            geneset = get_DEGs_per_group(
                view, n_genes=DEG_params["top_n_genes"],
                perc=DEG_params["perc"],
                p_val_cutoff=DEG_params["p_val_cutoff"],
                lfc=DEG_params["lfc"], direction=DEG_params["direction"])
            if not geneset:
                continue
            save_path = get_save_path(
                    f'_{mod}_{config["to_plot"]["down"]["heatmaps"]["title"]["deg"]}.pdf',
                    config, part)
            plot_mediods_heatmap(
                    view, ref=geneset, part=part,
                    **config["to_plot"]["down"]["heatmaps"]["medioid_params"])
    # #######################################################################
    # #######################################################################
    # Save the DEGs, markers and Cluster Stats
    # #######################################################################
    # #######################################################################
    # Combined and separate Cluster marker gene csvs
    # logger.debug("9", adata.uns["rank_genes_groups"].keys())
    if config["to_create"]["down"]["DEG_csv"]["create"]:
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info(f"Saving Cluster stats at: {config['general']['save_path']}DEGs/")
            logger.info(highlighers[0])
        get_DEG_gene_csvs(adata)
    # ##############################################################
    if config["to_create"]["down"]["marker_csv"]["create"]:
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info(f"Saving Marker gensets at: {config['general']['save_path']}markers/")
            logger.info(highlighers[0])
        # save_genesets_to_csv(adata, config=None, uns_key="genesets", genesets=None)
        save_genesets_to_csv(adata)
    # ##############################################################
    # Create Cluster stats
    # logger.debug("12", adata.uns["rank_genes_groups"].keys())
    # TODO: Implement/finish the stats!
    if config["to_create"]["down"]["cluster_stats_csv"]["create"] and False:
        # TODO: fix the cluster stats!
        if with_figure_explaination:
            logger.info(highlighers[0])
            logger.info(f"Saving Cluster stats at: {config['general']['save_path']}rna_cluster_stats.csv")
            logger.info(highlighers[0])
        if "rna" in config["to_create"]["down"]["cluster_stats_csv"]["marker_mod"]:
            if config["general"]["save_cluster_stats"]:
                create_group_stats_df(
                        adata, save_path=f"{config['general']['save_path']}rna_cluster_stats.csv")
    # #############################################################################################
    logger.info("".join([
        "#" * 80,
        f"\nFinished Downstream Visualisations after {time() - start} seconds\n",
        "#" * 80
    ]))
    if disable_sc_verbosity:
        sc.settings.verbosity = old_sc_verbosity
    return adata


# ###########################################################################################################
# Genset Enrichment
def get_ref_db(
            db: str = "Combined",
            databases: list[str] | None = None,
            ref_dict: dict[str, list[str]] | None = None,
            all_celltypes_to_use: list[str] | None = None,
            organism: str = "human"
        ) -> pd.DataFrame:
    """
    Retrieves marker data from various databases for specific cell types in a
    specified organism.

    This function interfaces with multiple databases (e.g., PanglaoDB,
    CellTypist, CellMarker, SCType) to retrieve markers (genes) associated with
    specific cell types. Depending on the specified database, it filters and
    processes the data to return a formatted DataFrame. It supports
    customization through several parameters, including organism type and cell
    types of interest.

    NOTE:
        - organism="mouse" only:
          The database of Decouplr is "well" curated
          and the mouse genes only consist of capital letters,
          we convert them to first capital and rest lower.
        - FILTER CELLS example for plasma cells:
          cell_types = sc_utils.search_database(get_ref_db(), "plasma")
          # CHECK the each of the celltypes in  cell_types, all of them will be
          # considered for gene extraction
          subsetted_df = combined_database[combined_database["cell_type"].isin(cell_types)]
          sc_utils.get_element_and_counts_with_positional_sum(subsetted_df)[:n_genes]
          # more examples at the doku of ``search_database``)
        - We cleaned some of the databases, e.G having non Capitalized Genes for
          the human ['PanglaoDB', 'CellMarker', 'SCType'], and cleaned spaces
          in the SCType

    Args:
        db (str, optional): The name of the database to retrieve data from.
            Default is "PanglaoDB". Defaults to "Combined".
        databases (list[str] | None, optional): A list of database names to use
            when ``db`` is set to "Combined". Default is None. Defaults to None.
        ref_dict (dict[str, optional): A dictionary containing custom markers to
            include. Default is None.
        all_celltypes_to_use (list[str] | None, optional): A list of cell types
            to filter the results by. Default is None. Defaults to None.
        organism (str, optional): The organism for which to retrieve marker
            data. Must be "human" or "mouse". Default is "human". Defaults to
            "human".

    Returns:
        pandas.DataFrame:
            A DataFrame containing the marker data filtered and
            formatted based on the specified parameters.

    Raises:
        ValueError: If the specified organism is not "human" or "mouse".

    Calls:
        df_split_col, get_ref_db

    Called By:
        get_ref_db

    TODO:
        - Replace hardcoded CSV reads with ``dc.get_resource`` where applicable.
        - Refactor db-specific logic into modular handlers.
    """
    # ################################################################################
    # Extract the paths form the local databases
    path_to_data = ALL_PATHS["PATH_TO_DATABASE"]
    PATH_TO_CELL_TYPE_GENESETS = GENESET_FILENAMES["PATH_TO_CELL_TYPE_GENESETS"]
    # ################################################################################
    # Check if organism exists in mapping
    if organism not in PATH_TO_CELL_TYPE_GENESETS:
        raise ValueError(
            f'Organism "{organism}" not found, please use one of: '
            f'{list(PATH_TO_CELL_TYPE_GENESETS.keys())}')
    # ################################################################################
    # Initial variable settings and checks
    source = "cell_type"
    target = "genesymbol"
    # ################################################################################
    if db == "Combined":
        # ##########################################
        # Handling 'Combined' database by merging results from multiple databases
        if not databases:
            databases = ["PanglaoDB", "CellTypist", "CellMarker", "SCType"]
        markers = get_ref_db(db=databases[0],
                             all_celltypes_to_use=all_celltypes_to_use,
                             organism=organism)
        for database in databases[1:]:
            markers = pd.concat([markers, get_ref_db(
                db=database, all_celltypes_to_use=all_celltypes_to_use,
                organism=organism)])
        if ref_dict:
            markers = pd.concat([markers, get_ref_db(
                db="Custom", all_celltypes_to_use=all_celltypes_to_use,
                ref_dict=ref_dict, organism=organism)])
    # ################################################################################
    elif db == "PanglaoDB":
        # ##########################################
        # Handling 'PanglaoDB' database by filtering based on organism and canonical markers
        if "PanglaoDB" not in PATH_TO_CELL_TYPE_GENESETS[organism]:
            raise ValueError(f'No PanglaoDB available for organism "{organism}"')
        PanglaoDB = pd.read_csv(
            path_to_data + PATH_TO_CELL_TYPE_GENESETS[organism]["PanglaoDB"]
        ).drop_duplicates()
        if "bool" in PanglaoDB[organism].dtype.name:
            reference_true = True
        else:
            reference_true = "True"
        PanglaoDB = PanglaoDB[PanglaoDB[organism] == reference_true]
        if "bool" in PanglaoDB["canonical_marker"].dtype.name:
            reference_true = True
        else:
            reference_true = "True"
        PanglaoDB = PanglaoDB[PanglaoDB["canonical_marker"] == reference_true]
        PanglaoDB = PanglaoDB[[
            "genesymbol", "cell_type", "germ_layer", "human_specificity",
            "organ", "ubiquitiousness"]]
        PanglaoDB = PanglaoDB.rename(columns={
            "organ": "tissue", f"{organism}_specificity": "specificity",
            "ubiquitiousness": "presence",
            "cell_type": source, "genesymbol": target})
        markers = PanglaoDB
        markers["db"] = "PanglaoDB"

    elif db == "CellTypist":
        # ##########################################
        # Handling 'CellTypist' database by filtering based on marker type and organism
        if organism == "mouse":
            logger.warning("CellTypist doesn't contain any Mouse Genesets!")
            return pd.DataFrame()
        if "CellTypist" not in PATH_TO_CELL_TYPE_GENESETS[organism]:
            raise ValueError(f'No CellTypist available for organism "{organism}"')
        markers = pd.read_csv(
            path_to_data + PATH_TO_CELL_TYPE_GENESETS[organism]["CellTypist"]
        ).drop_duplicates()
        markers = markers[markers['marker_type'] == 'curated_marker']
        # Mabye the database changes again and uses cell_type, just for the future.
        if True:
            markers = markers[['cell_subtype', 'genesymbol']]
        else:
            markers = markers[['cell_type', 'genesymbol']]
        markers.columns = [source, target]
        markers["db"] = "CellTypist"

    elif db == "CellMarker":
        # ##########################################
        # Handling 'CellMarker' database by filtering based on species and experimental validation
        if "CellMarker" not in PATH_TO_CELL_TYPE_GENESETS[organism]:
            raise ValueError(f'No CellMarker available for organism "{organism}"')
        Cell_marker_All = pd.read_csv(
            path_to_data + PATH_TO_CELL_TYPE_GENESETS[organism]["CellMarker"]
        ).drop_duplicates()
        Cell_marker_All = Cell_marker_All[Cell_marker_All["species"].str.lower() == organism]
        Cell_marker_All = Cell_marker_All[Cell_marker_All["marker_source"] == "Experiment"]
        Cell_marker_All = Cell_marker_All[[
            "tissue_class", "tissue_type", "cancer_type", "cell_type",
            "cell_name", "marker", "Symbol", "Genetype"]]
        Cell_marker_All = Cell_marker_All.rename(
            columns={
                "tissue_class": "tissue", "tissue_type": "tissue_specific",
                "cell_type": "condition",
                "cell_name": source, "Genetype": "gene_function",
                "Symbol": target, "marker": "other_marker_name"})
        markers = Cell_marker_All
        markers["db"] = "CellMarker"

    elif db == "SCType":
        # ##########################################
        # Handling 'SCType' database by splitting columns and renaming them
        if organism == "mouse":
            logger.warning("SCType doesn't contain any Mouse Genesets!")
            return pd.DataFrame()
        if "SCType" not in PATH_TO_CELL_TYPE_GENESETS[organism]:
            raise ValueError(f'No SCType available for organism "{organism}"')
        sc_type = pd.read_csv(
            path_to_data + PATH_TO_CELL_TYPE_GENESETS[organism]["SCType"]
        ).drop_duplicates()
        sc_type = sc_type[["tissueType", "cellName", "geneSymbolmore1", "geneSymbolmore2"]]
        sc_type = sc_type.rename(columns={
            "tissueType": "tissue", "cellName": source,
            "geneSymbolmore1": target, "geneSymbolmore2": "genesymbol_down"})
        sc_type = df_split_col(sc_type, key="genesymbol", sep=",")
        markers = sc_type
        markers["db"] = "SCType"
    else:
        logger.info("No Valid database! Try another one.")
        return
    # #################################################################################################
    # Preselect cell types, to include in the Classification.
    # NOTE: Check the databases for cells and use them! and automatically
    #       appends the marker in the ref_dict if passed
    if all_celltypes_to_use is not None:
        markers = markers[markers["cell_type"].isin(all_celltypes_to_use)]
    # #################################################################################################
    # Clean up and remove duplicates or empty values
    markers = markers[~markers.duplicated(['cell_type', 'genesymbol'])]
    markers = markers.dropna(subset=['cell_type', "genesymbol"])

    if organism == "mouse":
        markers["genesymbol"] = [
            vv[0].upper() + vv[1:].lower() for vv in markers["genesymbol"]]

    return markers


def get_receptor_ligand_db(
            organism: str = "human",
            adata: ad.AnnData | None = None,
            subset_var_key: str | None = None
        ) -> pd.DataFrame:
    """Load receptor–ligand interactions from OmniPath.

    Args:
        organism (str, optional): Organism to use. Only ``"human"`` is
            supported. Defaults to ``"human"``.
        adata (ad.AnnData, optional): AnnData object for subsetting
            interactions to genes present in ``adata.var_names``.
            Defaults to ``None``.
        subset_var_key (str, optional): .var key to further restrict
            to a subset of genes (e.g. ``"highly_variable"``).
            Defaults to ``None``.

    Returns:
        pd.DataFrame:
            Filtered receptor–ligand interactions.

    Raises:
        ValueError:
            If the organism is not available in the mapping.
        FileNotFoundError:
            If the OmniPath database file is missing.
    """
    PATH_TO_RECEPTOR_LIGAND_GENESETS = GENESET_FILENAMES[
        "PATH_TO_RECEPTOR_LIGAND_GENESETS"]
    if organism not in PATH_TO_RECEPTOR_LIGAND_GENESETS:
        raise ValueError(
            f'Organism "{organism}" not supported. '
            f'Available: {list(PATH_TO_RECEPTOR_LIGAND_GENESETS.keys())}')
    if "OmniPath" not in PATH_TO_RECEPTOR_LIGAND_GENESETS[organism]:
        raise ValueError(
            f'No OmniPath database defined for organism "{organism}".')

    path_to_data = ALL_PATHS["PATH_TO_DATABASE"]
    path_to_file = (
        path_to_data +
        PATH_TO_RECEPTOR_LIGAND_GENESETS[organism]["OmniPath"])

    if not os.path.exists(path_to_file):
        raise FileNotFoundError(
            f"Database file not found: {path_to_file}")

    # Load OmniPath
    df = pd.read_csv(path_to_file)

    # Subset to AnnData var_names
    if adata is not None:
        df = df[
            df["source_genesymbol"].isin(adata.var_names) &
            df["target_genesymbol"].isin(adata.var_names)]

        # Further restrict to subset_var_key
        if subset_var_key is not None and subset_var_key in adata.var:
            hv_genes = adata[:, adata.var[subset_var_key]].var_names
            df = df[
                df["source_genesymbol"].isin(hv_genes) &
                df["target_genesymbol"].isin(hv_genes)]

    return df


def run_decoupler(  # noqa: C901
            adata: ad.AnnData,
            ref: dict[str, list[str]] | pd.DataFrame,
            run_type: str = "consensus",
            weight: str | None = None,
            methods: list[str] | None = None,
            verbose: bool = False,
            min_n: int = 5,
            layer: str = "log2norm_counts",
            use_raw: bool = False,
        ) -> None:
    """
    Runs the Decoupler tool to analyze and visualize cell type-specific marker
    expression.

    The function applies various decoupling methods on single-cell data provided
    in ``adata``, using predefined ref/markers and the specified run type. It
    allows for customization options like adjusting for specific layers.

    Args:
        adata (anndata.AnnData): Adata object.
        ref (dict[str] | pandas.DataFrame): Marker genes to use in the
            decoupling analysis.
        run_type (str, optional): The type of decoupling analysis to run.
            Defaults to "consensus".
        visualize (bool, optional): Whether to generate visualizations. Defaults
            to False.
        weight (str | None, optional): Weight column name in the ref DataFrame.
            Defaults to None.
        methods (list[str] | None, optional): List of decoupling methods to use.
            Defaults to None.
        databases (str, optional): Path to databases used in the analysis.
            Defaults to None.
        verbose (bool, optional): Whether to output verbose information.
            Defaults to False.
        min_n (int, optional): Minimum number of cells required for the
            analysis. Defaults to 5.
        layer (str, optional): Layer in ``adata`` to use instead of the default
            expression matrix. Defaults to "log2norm_counts".
        use_raw (bool, optional): Whether to use the raw adata.
            Defaults to False.

    Returns:
        None:
            The function updates ``adata`` in-place with decoupling results and
            possibly visualizations.

    Raises:
        NotImplementedError: If an unsupported ``run_type`` is provided.

    Calls:
        run_viper

    TODO:
        - Write detailed documentation for this function.
        - Introduce function argument for raw or not.
        - Fix the seeding to ensure reproducibility.
        - Test the functionality extensively.
        - Implement classification of cells based on mean with shared names if
          means are close. and "Major_celltype_per_cluster".
        - Fix random seeding for reproducibility.
        - Support classification with mean proximity across cell types.
        - Get Celltype stats: "Celltypes_per_cluster" and
          "Major_celltype_per_cluster" statistics.

    Tags:
        annotation, config
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*IProgress not found.*")
        import decoupler as dc

    source = "cell_type"
    target = "genesymbol"
    weight = "weight"
    # #########################################################
    # Get the config object of the adata or set a default seed
    if "config" in adata.uns.keys():
        config = adata.uns["config"]
    else:
        config = {"general": {"seed": 42}}
    # #########################################################
    # Convert ref to DataFrame if it's a dictionary
    if isinstance(ref, dict):
        ref = pd.melt(pd.DataFrame.from_dict(ref, orient="index").T)
        ref.columns = ['cell_type', 'genesymbol']
        ref[weight] = 1
    # #########################################################
    # Delete duplicate genes for cell types
    ref = ref[~ref.duplicated(['cell_type', 'genesymbol'])]
    # #########################################################
    # If layer is specified, switch to that layer in adata
    prev_layer = None
    if layer is not None:
        if layer in adata.layers.keys():
            prev_layer = adata.X
            adata.X = adata.layers[layer]
        else:
            logger.info("Using the default X matrix, as the provided layer is not available.")
    # #########################################################
    # Run decoupler analysis based on the selected run_type
    if run_type == "decouple":
        dc.decouple(mat=adata, net=ref, weight=weight, source=source, target=target,
                    min_n=min_n, verbose=verbose, use_raw=use_raw, methods=methods)
        obsm_key = 'consensus_estimate'
    elif run_type == "ora":
        dc.run_ora(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                   verbose=verbose, use_raw=use_raw)
        obsm_key = 'ora_estimate'
    elif run_type == "consensus":
        dc.run_consensus(mat=adata, net=ref, weight=weight, source=source, target=target,
                         min_n=min_n, verbose=verbose, use_raw=use_raw)
        obsm_key = 'consensus_estimate'
    elif run_type == "aucell":
        dc.run_aucell(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                      verbose=verbose, use_raw=use_raw)
        obsm_key = 'aucell_estimate'
    elif run_type == "gsea":
        dc.run_gsea(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                    verbose=verbose, use_raw=use_raw)
        obsm_key = 'gsea_estimate'
    elif run_type == "gsva":
        dc.run_gsva(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                    verbose=verbose, use_raw=use_raw)
        obsm_key = 'gsva_estimate'
    elif run_type == "wmean":
        dc.run_wmean(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                     verbose=verbose, use_raw=use_raw)
        obsm_key = 'wmean_estimate'
    elif run_type == "wsum":
        dc.run_wsum(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                    verbose=verbose, use_raw=use_raw, batch_size=1000, times=1000)
        obsm_key = 'wmean_estimate'
    elif run_type == "ulm":
        dc.run_ulm(mat=adata, net=ref, source=source, target=target, weight=weight,
                   batch_size=10000, min_n=min_n, verbose=verbose, use_raw=use_raw)
        obsm_key = 'ulm_estimate'
    elif run_type == "mlm":
        dc.run_mlm(mat=adata, net=ref, source=source, target=target, min_n=min_n,
                   verbose=verbose, use_raw=use_raw)
        obsm_key = 'mlm_estimate'
    elif run_type == "mdt":
        dc.run_mdt(mat=adata, net=ref, source=source, target=target, weight=weight,
                   trees=10, min_leaf=5, n_jobs=-1, min_n=min_n, seed=config["general"]["seed"],
                   verbose=verbose, use_raw=use_raw)
        obsm_key = 'mdt_estimate'
    elif run_type == "udt":
        dc.run_udt(mat=adata, net=ref, source=source, target=target, weight=weight,
                   min_leaf=5, min_n=min_n, seed=config["general"]["seed"], verbose=verbose, use_raw=use_raw)
        obsm_key = 'udt_estimate'
    elif run_type == "viper":
        dc.run_viper(mat=adata, net=ref, source=source, target=target, weight=weight,
                     pleiotropy=True, reg_sign=0.05, n_targets=10, penalty=20, batch_size=10000,
                     min_n=min_n, verbose=verbose, use_raw=use_raw)
        obsm_key = 'viper_estimate'
    else:
        logger.info(f'{run_type} is not a valid decoupler run type! Try another one.')
        return
    obsm_key = obsm_key + ""  # To silence the obsm_key flake8, might be using it.
    # #########################################################
    # Reset adata layer to previous layer if changed
    if prev_layer is not None:
        adata.X = prev_layer
    # #########################################################
    # Add results to ``adata.obs`` if specified
    # if add_to_obs:
    #     acts = dc.get_acts


# ###########################################################################################################
# Saving and Loading
def load_h5ad(path_to_h5ad: str) -> ad.AnnData:
    """
    Loads an adata object from an H5AD file and populates its ``.uns`` attribute
    with experiment configuration data from a corresponding pickle file.

    NOTE:
        The function assumes that the pickle file containing experiment
        configuration data has the same base name as the H5AD file, but with the
        appropriate compression extension (e.g. ``.pickle.xz``, ``.pickle.gz``, or
        ``.pickle``).

    Args:
        path_to_h5ad (str): Path to the H5AD file to be loaded.

    Returns:
        anndata.AnnData:
            The loaded adata object with experiment configuration
            data added to ``adata.uns["config"]``.

    Raises:
        FileNotFoundError: If the corresponding pickle file is not found in the
            same directory as the H5AD file.

    Called By:
        load_spatial_prox

    TODO:
        Consider adding error handling for cases where the H5AD file cannot be
        read or the pickle file is corrupted.

    Tags:
        config, io, pipeline
    """
    # #########################################################
    # Load the adata object from the H5AD file
    adata = sc.read_h5ad(path_to_h5ad)
    # #########################################################
    # Determine the path to the corresponding pickle file
    # Check if the path does not end with ".h5ad" and adjust accordingly
    if path_to_h5ad[-5:] != ".h5ad":
        path_to_uns = copy(path_to_h5ad)
        path_to_h5ad = path_to_h5ad + ".h5ad"
    else:
        # wouldn't work if you have a .h5ad folder
        # path_to_uns = sub("\.h5ad", "", path_to_h5ad)
        path_to_uns = path_to_h5ad[:-5]

    # Print the path for debugging purposes
    # print(path_to_uns)
    # #########################################################
    # Load the experiment configuration data from the pickle file
    # and populate the ``uns`` attribute of the adata object
    if os.path.exists(path_to_uns + ".pickle.xz"):
        with lzma.open(path_to_uns + ".pickle.xz", "rb") as f:
            uns = pickle.load(f)
    elif os.path.exists(path_to_uns + ".pickle.gz"):
        with gzip.open(path_to_uns + ".pickle.gz", "rb") as f:
            uns = pickle.load(f)
    elif os.path.exists(path_to_uns + ".pickle"):
        with open(path_to_uns + ".pickle", "rb") as f:
            uns = pickle.load(f)
    else:
        # #####################################################
        # Handle the case where the pickle file is not found
        logger.warning(
            "The Pickled adata.uns file is not in the same directory as the .h5ad. "
            "We could not load it. Please ensure the pickled file is in the same directory.")
        # Create an empty ``uns`` dictionary as a fallback
        uns = {}
    # print(uns.keys())
    # #########################################################
    # Populate the adata object's ``uns`` attribute with loaded data
    for key in uns.keys():
        adata.uns[key] = uns[key]
    # #########################################################
    # If the adata.X is None, create a copy from the first available layer
    if adata.X is None:
        candidate_layers: Sequence[str] = [
            "log2norm_counts",
            "log2_counts",
            "counts"]

        selected_layer = None
        for layer_name in candidate_layers:
            if layer_name in addata.layers and adata.layers[layer_name] is not None:
                selected_layer = layer_name
                break

        if selected_layer is None:
            raise ValueError(
                "adata.X is None and none of the required layers are available: "
                f"{candidate_layers}")

        logger.warning(f"Setting adata.X from layer '{selected_layer}'.")
        adata.X = adata.layers[selected_layer].copy()
    # #########################################################
    # Return the populated adata object
    return adata


def save_h5ad(
            adata: ad.AnnData,
            path_to_h5ad: str,
            compression: str = "gzip",
            save_umap: bool = True,
            force_remove: bool = True
        ) -> None:
    """Saves an ``adata`` object with experiment_dict.

    NOTE:
        - If interrupted, ``adata.uns["config"]`` might not be restored properly.
        - JSON is avoided due to HDF5 limitations with None values and
          lists/arrays.

    Args:
        adata (anndata.AnnData): Adata object with experiment_dict in
            ``adata.uns["config"]``.
        path_to_h5ad (str): Path to save the adata object.
        compression (str, optional): The compression to use for h5ad.
            Defaults to "gzip".
        save_umap (bool, optional): Whether to save the UMAP object.
            Defaults to True.
        force_remove (bool, optional): If True, deletes any existing file before
            saving. Defaults to True.

    Returns:
        None:
            This function performs an in-place save operation and returns
            nothing.

    Raises:
        ValueError: If there is an issue with saving the adata object.
        KeyboardInterrupt: If the operation is interrupted by the user.

    Calls:
        convert_mixed_types

    Called By:
        Spatial_prox.save

    TODO:
        - Re-evaluate HDF5 version compatibility with dict keys set to None or
          list types.
        - Add checks if the pickle automated pickle file is there, to also delete
          it.

    Tags:
        config, io, pipeline
    """
    # ########################################################################
    # Remove the UMAP object if present and not needed
    umap_key = adata.uns["config"]["general"]["save_umap_object"]["key"]
    if not save_umap and umap_key in adata.uns.keys():
        del adata.uns[umap_key]
    # ########################################################################
    # Adjust the path for saving the uns and ensure correct file extension
    if path_to_h5ad[-5:] != ".h5ad":
        path_to_uns = copy(path_to_h5ad)
        path_to_h5ad = path_to_h5ad + ".h5ad"
    else:
        # wouldn't work if you have a .h5ad folder
        # path_to_uns = sub("\.h5ad", "", path_to_h5ad)
        path_to_uns = path_to_h5ad[:-5]

    # Save the uns depending on the compression type
    if compression == "lzma":
        with lzma.open(path_to_uns + ".pickle.xz", "wb") as f:
            pickle.dump(adata.uns, f)
    elif compression == "gzip":
        with gzip.open(path_to_uns + ".pickle.gz", "wb") as f:
            pickle.dump(adata.uns, f)
    else:
        with open(path_to_uns + ".pickle", "wb") as f:
            pickle.dump(adata.uns, f)
    # ########################################################################
    # Remove specific keys from adata.uns before saving
    keys_to_remove = np.intersect1d(["config", "stats"], list(adata.uns.keys()))
    save_uns = {}
    for key in keys_to_remove:
        save_uns[key] = adata.uns[key]
        del adata.uns[key]
    # ###################
    # Save the adata object to an H5AD file, with error handling
    try:
        if os.path.exists(path_to_h5ad) and force_remove:
            os.system(f'rm {path_to_h5ad}')
        adata.write_h5ad(path_to_h5ad, compression=compression)
    except ValueError:
        logger.warning(
            "IMPORTANT!!! The Adata may be saved incorrectly. We replaced columns with"
            " mixtures of dtypes to have a single one!\n"
            "Resolve it yourself and save it again for default behaviour!")
        adata.obs = convert_mixed_types(adata.obs, fix_multitype=True)
        adata.write_h5ad(path_to_h5ad, compression=compression)
    except KeyboardInterrupt:
        pass
    # ####################
    # Reset the uns to the original state
    for key in save_uns.keys():
        adata.uns[key] = save_uns[key]


# ###########################################################################################################
# Old sc_automatic and sc_advanced
# ################################################################
# Automatic parameter estimation
def set_neighbors_n_pcs(
            adata: ad.AnnData,
            ratio_key: str = "variance_ratio",
            variance_sum: float | None = None,
            use_automatic_pca_variance_sum: str | None = None,
            evaluate_at_pcs: int | None = None,
            overwrite_config: bool = True,
            to_plot: bool = True,
        ) -> int | None:
    """
    Automatically estimates the number of principal components (PCs) to use
    for the neighborhood graph based on explained variance.

    This function selects the number of PCs by applying either a cumulative sum
    threshold ("cumsum") or a cumulative difference heuristic ("cumsum_diffs")
    on PCA variance values. The strategy and threshold can be configured via
    ``adata.uns["config"]``.

    NOTE:
        If your data is as noisy as scRNA-seq (and you don't have computation
        power), it may make sense to use this. But if you want to do a proper
        analysis, use the "cumsum" method.

    This function overwrites the
    ``adata.uns["config"]["pp"]["neighbors"]["n_pcs"]`` parameter based on the
    setting in ``adata.uns["config"]["general"]["use_automatic_pca_variance_sum"]``
    which can be either "cumsum" or "cumsum_diffs".

    For many samples, genes may be all quite variable. Consider using the
    parameter setting
    >>> ``adata.uns["config"]["general"]["use_automatic_pca_variance_sum"] = "cumsum_diffs"``
    >>> ``adata.uns["config"]["general"]["pca_variance_sum"] = 0.95``
    (or 0.99 if the number of PCs would be below 10).

    Args:
        adata (anndata.AnnData): Adata object with PCA results stored in
            ``adata.uns["pca"]`` and config under ``adata.uns["config"]``.
        ratio_key (str, optional): Key for explained variance ratio array in
            ``adata.uns["pca"]``. Defaults to "variance_ratio".
        variance_sum (float | None, optional): Desired cumulative variance
            threshold. If None, uses value from
            ``config["general"]["pca_variance_sum"]``. Defaults to None.
        use_automatic_pca_variance_sum (str | None, optional): Method to compute
            PC count is "cumsum" or "cumsum_diffs". If None, uses
            ``config["general"]["use_automatic_pca_variance_sum"]``. Defaults
            to None.
        evaluate_at_pcs (int | None, optional): If set, logs cumulative variance
            at the specified PC index. Defaults to None.
        overwrite_config (bool, optional): Whether to write result into
            ``config["pp"]["neighbors"]["n_pcs"]``. If False, returns the PC
            count. Defaults to True.
        to_plot (bool, optional): If True, plots PCA variance and chosen
            threshold. Useful for debugging. Defaults to True.

    Returns:
        int | None:
            None or int: Updates
            ``adata.uns["config"]["pp"]["neighbors"]["n_pcs"]`` in place by
            default. If ``overwrite_config`` is False, returns the
            selected number of PCs as an integer.

    Raises:
        KeyError: If required keys are missing in ``adata.uns["pca"]`` or
            ``adata.uns["config"]``.

    Calls:
        StatKeeper.get

    Called By:
        cluster_SC_scanpy_like, collect_n_pcs_for_variance_sums

    Tags:
        clustering, config, visualization
    """
    # #########################################################
    # Retrieve configuration settings from adata object
    config = adata.uns["config"]

    if use_automatic_pca_variance_sum is None:
        use_automatic_pca_variance_sum = config["general"].get("use_automatic_pca_variance_sum", "cumsum")

    if variance_sum is None:
        variance_sum = config["general"]["pca_variance_sum"]
    # #########################################################
    # Check if automatic PCA variance sum calculation is enabled
    if use_automatic_pca_variance_sum in ["cumsum", "cumsum_diffs"]:
        # ##########################################
        # If the "cumsum" method is chosen, calculate the number of PCs based on cumulative explained variance
        if use_automatic_pca_variance_sum == "cumsum":
            var_sum = sum(adata.uns["pca"][ratio_key])
            # ##########################################
            # Rescale PCA variance if needed and adjust the number of PCs to use
            if var_sum < 0.99:
                logger.warning(
                    "Rescaling the PCA Variance. If you want better results, increase the n_comps argument of PCA.")
                cumsum_ = np.cumsum(adata.uns["pca"][ratio_key] / var_sum)
            else:
                cumsum_ = np.cumsum(adata.uns["pca"][ratio_key])
        # ##########################################
        # If the "cumsum_diffs" method is chosen, calculate the number of PCs based on cumulative differences
        elif use_automatic_pca_variance_sum == "cumsum_diffs":
            ddd = np.abs(np.ediff1d(adata.uns["pca"]["variance"][adata.uns["pca"]["variance"] > 1]))
            cumsum_ = np.cumsum(ddd / sum(ddd))

        if evaluate_at_pcs is not None:
            logger.info(f'For {evaluate_at_pcs} PCs the variance ratio is {cumsum_[evaluate_at_pcs]:.4f}')
        n_pcs_to_use = int(np.argmax(cumsum_ > variance_sum))
        # #########################################################
        # Optional: Visualize the PCA variance and chosen number of PCs
        if to_plot:
            tmp_pca = adata.uns["pca"][ratio_key].copy()
            tmp_pca = tmp_pca[tmp_pca > 0]
            plt.scatter(np.arange(tmp_pca.shape[0]), tmp_pca)
            ddd = np.abs(np.ediff1d(adata.uns["pca"]["variance"]))
            bla = np.argmax(np.cumsum(ddd / sum(ddd)) > 0.95)
            # print(bla)
            plt.axvline(bla)
            plt.axvline(n_pcs_to_use)
            plt.yscale("log")
        # ##########################################
        # Update configuration with the calculated number of PCs and log a warning or return
        if overwrite_config:
            logger.warning(
                f'Overwriting config["pp"]["neighbors"]["n_pcs"] = {n_pcs_to_use} due to automatic n pcs '
                f'calculation based on {variance_sum} variance')
            adata.uns["config"]["pp"]["neighbors"]["n_pcs"] = n_pcs_to_use
        else:
            return n_pcs_to_use


def collect_n_pcs_for_variance_sums(
            adata: ad.AnnData,
            pca_variance_sums: list[float] | tuple[float, ...] = [0.90, 0.925, 0.95, 0.975, 0.985, 0.99],
            to_plot: bool = False
        ) -> list[int] | tuple[int, ...]:
    """
    Determine the number of principal components (PCs) for given variance thresholds.

    For each variance sum in ``pca_variance_sums``, this function updates the PCA
    configuration in ``adata.uns`` and applies ``sc_advanced.set_neighbors_n_pcs``.
    This helper is assumed to update
    ``adata.uns["config"]["pp"]["neighbors"]["n_pcs"]`` to reflect the number of
    PCs required to reach the specified variance.

    Args:
        adata (anndata.AnnData): Adata object with expected structure in
            ``adata.uns["config"]``.
        pca_variance_sums (list[float] | tuple[float, optional): List or tuple
            of cumulative variance thresholds for PCA. Defaults to [0.90, 0.925,
            0.95, 0.975, 0.985, 0.99].
        to_plot (bool, optional): Whether to plot the variance explained during
            PCA selection. Passed to ``set_neighbors_n_pcs``. Defaults to False.

    Returns:
        list[int] | tuple[int, ...]:
            list[int] or tuple[int, ...]: Number of PCs selected for each
                threshold. Returned in the same order as ``pca_variance_sums``.

    Raises:
        KeyError: If required config keys are missing from ``adata.uns``.
        AttributeError: If ``sc_advanced.set_neighbors_n_pcs`` fails to update
            ``adata.uns["config"]["pp"]["neighbors"]["n_pcs"]``.

    Calls:
        set_neighbors_n_pcs

    Called By:
        search_for_neighbor_params

    Tags:
        calculation, config
    """
    # #########################################################
    # Initialize the container for storing number of PCs
    n_pcs_to_use = []
    # #########################################################
    # Iterate through the provided variance thresholds
    for val in pca_variance_sums:
        res_pcs = set_neighbors_n_pcs(
            adata,
            variance_sum=val,
            overwrite_config=False,
            to_plot=to_plot)
        n_pcs_to_use.append(res_pcs)
    # #########################################################
    # the percentages might return the same number of pcs, therefore we uniqueify
    n_pcs_to_use = np.unique(n_pcs_to_use).tolist()
    if 0 in n_pcs_to_use:
        n_pcs_to_use.remove(0)
    # #########################################################
    return n_pcs_to_use


# ################################################################
# Semi Automatic parameter estimation
def search_for_leiden_resolution(
            adata: ad.AnnData,
            resolutions: list[float] | None = None,
            n_eval: int = 30,
            keys: list[str] = [],
            jaccard_paired_keys: list[list[str, str]] = [],
            n_cluster_range: list[int] | None = None,
            return_clusterings: bool = True,
            dpi: int = 60,
            figsize:  Sequence[float | int] = [6, 5],
            plot_densitys: bool = True,
            umap_kwargs: dict = {},
            density_kwargs: dict = {},
            jaccard_kwargs: dict = {}
        ) -> None:
    """Visualize different Leiden clusterings optionally with marker genes.

    This function evaluates different Leiden clustering resolutions on an adata
    object and optionally visualizes the clustering results with specified
    marker genes using UMAP.

    NOTE:
        - The ``n_eval`` parameter is not used if ``resolutions`` is provided.
        - Ensure that the adata object is properly pre-processed before passing
          it to this function.
        - If you break the execution of this function, it will overwrite the
          config! Better to save the object before.
        -

    Args:
        adata (anndata.AnnData): Adata object containing the data to cluster.
        resolutions (list[float] | None, optional): List of resolutions to
            evaluate for Leiden clustering. If not provided, a default range of
            30 resolutions from 0.01 to 1.5 will be used. Defaults to None.
        n_eval (int, optional): Number of resolutions to sample if ``resolutions``
            is not provided. Defaults to 30.
        Defaults to 30.
        keys (list[str], optional): List of marker genes to plot on the UMAP. If
            provided, these ref/markers will be highlighted in the UMAP
            visualization. Defaults to [].
        jaccard_paired_keys (list[list[str, str]]):
            List of paired observation keys from ``adata.obs`` to compare.
            Each list must contain two valid column names in ``adata.obs``.
            The function computes Jaccard similarity between the categories
            defined by those paired keys.
        n_cluster_range (list[int] | None, optional): [min, max] allowed number
            of clusters. Defaults to None.
        return_clusterings (bool, optional): If True, return a dict of
            clustering Series per resolution. Defaults to True.
        dpi (int, optional): DPI parameter for the plots, lots of plots might
            overflow the memory. Defaults to 60.
        figsize (Sequence[float | int], optional): figsize parameter for the plots, lots of
            plots might overflow the memory. Defaults to [6.
        plot_densitys (bool, optional): If True, the categorical columns will be
            plotted as density plot. Defaults to True.
        umap_kwargs (dict): Extra keyword arguments passed to plot_umap.
            Defaults to {}.
        density_kwargs (dict): Extra keyword arguments passed to
            plot_embedding_density. Defaults to {}.
        jaccard_kwargs (dict): Extra keyword arguments passed to
            plot_jaccard_heatmap_cluster_comparison. Defaults to {}.

    Returns:
        None:
            The function modifies the ``adata`` object in-place and produces
            UMAP visualizations.

    Calls:
        cluster_SC_scanpy_like, plot_umap_cat_splitting

    Tags:
        clustering, config, visualization
    """
    # #########################################################
    # Setup the resolution list to evaluate
    # If resolutions are not provided, generate a list of 30 resolutions from 0.01 to 1.5
    if resolutions is None:
        resolutions = [round(x, 3) for x in np.linspace(0.01, 1.5, n_eval)]

    logger.info("Search space:")
    logger.info(f"  resolutions: {resolutions}")
    # #########################################################
    config_copy = copy(adata.uns["config"])
    clustering_dict = {} if return_clusterings else None
    # #########################################################
    # Iterate through each resolution and apply Leiden clustering
    for res in resolutions:
        # ##########################################
        # Print the resolution being evaluated and update the resolution in the adata object
        logger.info("#" * 120)
        logger.info(f'resolution: {res}')
        adata.uns["config"]["tl"]["leiden"]["resolution"] = res
        # ##########################################
        # Perform clustering using the provided settings; run PCA, neighbors, and UMAP as required
        adata = cluster_SC_scanpy_like(
            adata, run_pca=False, run_neighbors=False, run_clustering=True, run_umap=False)
        # #########################################################
        # Evaluate number of clusters and optionally skip

        n_clusters = adata.obs["leiden"].nunique()
        print(n_clusters)
        if n_cluster_range is not None:
            if not (n_cluster_range[0] <= n_clusters <= n_cluster_range[1]):
                logger.info(f"Skipping resolution {res}: n_clusters = {n_clusters} not in range {n_cluster_range}")
                continue
        # #########################################################
        # Save clustering if requested
        if return_clusterings:
            clustering_dict[res] = adata.obs[
                adata.uns["config"]["general"]["cluster_algorithm"]].copy()
        # #########################################################
        # Visualize the UMAP with the clustering results
        plot_umap_cat_splitting(
            adata, keys, plot_densitys, figsize, dpi,
            umap_kwargs=umap_kwargs, density_kwargs=density_kwargs)
        # #########################################################
        # Visualize the jaccard overlap
        # ##################
        # Error handling and plotting
        if not isinstance(jaccard_paired_keys, (list, tuple)):
            raise TypeError("jaccard_paired_keys must be a list of pairs")
        for pair in jaccard_paired_keys:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(
                    "Each pair in jaccard_paired_keysmust "
                    "have exactly 2 elements")
            k1, k2 = pair
            if k1 not in adata.obs or k2 not in adata.obs:
                raise KeyError(f"Both {k1} and {k2} must exist in adata.obs")
            ##################
            # Plot the jaccard overlapping score
            plot_jaccard_heatmap_cluster_comparison(
                adata.obs[k1],
                adata.obs[k2],
                **jaccard_kwargs)
    # #########################################################
    # Reset to previous config
    adata.uns["config"] = config_copy

    if return_clusterings:
        return clustering_dict


def search_for_neighbor_params(
            adata: ad.AnnData,
            n_pcs: list[int] | None = None,
            n_neighborss: list[int] | None = None,
            min_dists: list[float] | None = None,
            keys: list[str] = [],
            jaccard_paired_keys: list[list[str, str]] = [],
            clean_keys: bool = False,
            return_umaps: bool = True,
            dpi: int = 60,
            figsize: Sequence[float | int] = [6, 5],
            plot_densitys: bool = True,
            umap_kwargs: dict = {},
            density_kwargs: dict = {},
            jaccard_kwargs: dict = {}
        ) -> dict | None:
    """Visualize different UMAPs optimized with marker genes.

    This function evaluates different UMAP parameters (``min_dist`` and
    ``n_neighbors``) to visualize their effects on the clustering of an adata
    object. It allows users to plot UMAPs with different configurations while
    optionally highlighting specific marker genes.

    NOTE:
        - Ensure that the adata object is properly pre-processed before passing
          it to this function.
        - If you break the execution of this function, it can possibly overwrite
          or delete the config! Better to copy the object before.

    Args:
        adata (anndata.AnnData): Adata object.
        n_pcs (list[int] | None, optional): A list of ``n_pcs`` values to evaluate
            for the Neighborhood. Defaults to None.
        n_neighborss (list[int] | None, optional): A list of ``n_neighbors``
            values to evaluate for UMAP. Defaults to a list generated by
            linspace between 5 and 30 with 6 steps. Defaults to None.
        min_dists (list[float] | None, optional): A list of ``min_dist`` values to
            evaluate for UMAP. 25, 0.5, 0.75]. Defaults to None.
        keys (list[str], optional): A list of marker genes to plot on the UMAP.
            Defaults to [].
        jaccard_paired_keys (list[list[str, str]]):
            List of paired observation keys from ``adata.obs`` to compare.
            Each list must contain two valid column names in ``adata.obs``.
            The function computes Jaccard similarity between the categories
            defined by those paired keys.
        clean_keys (bool, optional): If True, filters the marker genes to those
            present in the adata object. Defaults to False.
        dpi (int, optional): DPI parameter for the plots, lots of plots might
            overflow the memory. Defaults to 60.
        figsize (list[int] | tuple[int], optional): figsize parameter for the
            plots, lots of plots might overflow the memory. Defaults to [6.
        plot_densitys (bool, optional): If True, the categorical columns will be
            plotted as density plot. Defaults to True.
        umap_kwargs (dict): Extra keyword arguments passed to plot_umap.
            Defaults to {}.
        density_kwargs (dict): Extra keyword arguments passed to
            plot_embedding_density. Defaults to {}.
        jaccard_kwargs (dict): Extra keyword arguments passed to
            plot_jaccard_heatmap_cluster_comparison. Defaults to {}.

    Returns:
        dict | None:
            dict or None: Nested dictionary of UMAP coordinates if
            ``return_umaps=True``, else None.

    Calls:
        cluster_SC_scanpy_like, collect_n_pcs_for_variance_sums,
        plot_umap_cat_splitting

    TODO:
        Consider adding functionality to save UMAP plots automatically.

    Tags:
        clustering, config, visualization
    """
    # #########################################################
    # Setup default parameters if none are provided
    if n_pcs is None:
        n_pcs = collect_n_pcs_for_variance_sums(adata)

    if min_dists is None:
        min_dists = [0.25, 0.5, 0.75]

    if n_neighborss is None:
        n_neighborss = [5, 10, 25, 50, 75]  # [round(x) for x in np.linspace(5, 75, 6)]

    logger.info("Search space:")
    logger.info(f"  n_pcs: {n_pcs}")
    logger.info(f"  min_dists: {min_dists}")
    logger.info(f"  n_neighborss: {n_neighborss}")
    # #########################################################
    # Setup the variable to save the resuts
    if return_umaps:
        umap_dict = {}
    # #########################################################
    # Save original config, in the end the last parameters are in the
    # object and this doesn't help so we will reset them.
    config_copy = copy(adata.uns["config"])
    # #########################################################
    # Update marker genes if ``clean_keys`` is True
    if clean_keys:
        keys = np.intersect1d(keys, adata.var_names)
    # #########################################################
    # Iterate through each combination of ``min_dist`` and ``n_neighbors``
    for pcs in n_pcs:
        adata.uns["config"]["pp"]["neighbors"]["n_pcs"] = pcs
        if return_umaps:
            umap_dict[pcs] = {}
        logger.info("#" * 120)
        logger.info(f'n_pcs: {pcs}')
        for n_neighbors in n_neighborss:
            logger.info("#" * 100)
            logger.info(f'n_pcs: {pcs}')
            logger.info(f'n_neighbors: {n_neighbors}')
            adata.uns["config"]["pp"]["neighbors"]["n_neighbors"] = n_neighbors
            adata = cluster_SC_scanpy_like(
                adata, run_pca=False, run_neighbors=True, run_clustering=False, run_umap=False)
            for min_dist in min_dists:
                # ##########################################
                # Print current ``min_dist`` being evaluated
                logger.info("#" * 80)
                logger.info(f'n_pcs: {pcs}, n_neighbors: {n_neighbors}, min_dist: {min_dist}')
                logger.info(f'[{pcs}, {n_neighbors}, {min_dist}]')
                # Update UMAP ``min_dist`` parameter in adata object
                adata.uns["config"]["tl"]["umap"]["min_dist"] = min_dist
                adata = cluster_SC_scanpy_like(
                    adata, run_pca=False, run_neighbors=False, run_clustering=False, run_umap=True)
                # ##########################################
                if return_umaps:
                    if n_neighbors not in umap_dict[pcs]:
                        umap_dict[pcs][n_neighbors] = {}
                    umap_dict[pcs][n_neighbors][min_dist] = adata.obsm["X_umap"].copy()
                # ##########################################
                # Visualize the UMAP with the clustering results
                plot_umap_cat_splitting(
                    adata, keys, plot_densitys, figsize, dpi,
                    umap_kwargs=umap_kwargs, density_kwargs=density_kwargs)
                # ##########################################
                # Visualize the jaccard overlap
                # ##################
                # Error handling and plotting
                if not isinstance(jaccard_paired_keys, (list, tuple)):
                    raise TypeError("jaccard_paired_keys must be a list of pairs")
                for pair in jaccard_paired_keys:
                    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                        raise ValueError(
                            "Each pair in jaccard_paired_keysmust "
                            "have exactly 2 elements")
                    k1, k2 = pair
                    if k1 not in adata.obs or k2 not in adata.obs:
                        raise KeyError(f"Both {k1} and {k2} must exist in adata.obs")
                    ##################
                    # Plot the jaccard overlapping score
                    plot_jaccard_heatmap_cluster_comparison(
                        adata.obs[k1],
                        adata.obs[k2],
                        **jaccard_kwargs)
    # #########################################################
    # Reset to previous config
    adata.uns["config"] = config_copy
    # #########################################################
    # Return the results
    if return_umaps:
        return umap_dict


# ################################################################
# Histogram equalization
def hist_equalize_data(
            adata: ad.AnnData,
            bins: int = 50,
            mask: np.ndarray | None = None,
            use_cell_only: bool = True
        ) -> ad.AnnData:
    """DO NOT USE THIS FUNCTION!!! It is (was) under development.

    Runs histogram equalization for each cell in the ``adata`` object, rescaling
    the cells to have an evenly spaced distribution instead of a
    negative-binomial distribution.

    This function is designed to adjust the data within each cell, or optionally
    across all cells, to achieve a more uniform distribution. This may enhance
    clustering results by summarizing similar expressed values into one value,
    but it is not suitable for differential expression analysis (DEG).

    Args:
        adata (anndata.AnnData): Adata object.
        bins (int, optional): The number of bins to use for spacing. More bins
            result in stronger discretization. Defaults to 50.
        mask (numpy.ndarray | None, optional): A mask indicating which elements
            to include. Defaults to None.
        use_cell_only (bool, optional): Whether to treat each cell separately
            during equalization. Defaults to True.

    Returns:
        anndata.AnnData:
            The ``adata`` object with histogram equalized data.

    Calls:
        StatKeeper.get

    Called By:
        Run_all_prep_steps_clustering

    TODO:
        - SETUP mask as the 0, to make sure they are not changed.
        - Process to fit CPU/GPU.
        - This should actually be run on a whole dataframe; run and test the
          clustering should be much better, because similar expressed values
          will be summarized into one value. This should be much better than
          the harmonization techniques available. BUT: Never use this for
          DEGs. It would only make sense if we would have a ground truth
          to tune the ``bins`` parameter.

    Tags:
        normalization, scaling
    """
    # #########################################################
    # Import the necessary module
    from skimage.exposure import equalize_hist
    # #########################################################
    # Initial setup and mask processing
    if mask is None:
        mask = np.where(adata.X.get().copy().toarray() > 0, 1, 0)
        entrys = sum(mask)
    # #########################################################
    # Histogram equalization depending on whether ``use_cell_only`` is set
    if use_cell_only:
        logger.debug(adata.X.shape)
        X_temp = np.zeros_like(adata.X.get().copy().toarray())
        for i in range(X_temp.shape[0]):
            x = adata.X.get().copy().toarray()[i, :]
            x = equalize_hist(x, bins)
            x_min = np.min(np.where(x > 0, x, 0))
            x_max = np.max(x)
            logger.debug(x_min, x_max)
            x = (x - x_min) / (x_max - x_min)
            X_temp[i, :] = x

        adata.X = X_temp
    else:
        # TODO: run and test this
        entrys = sum(np.where(adata.X > 0, 1, 0))
        x = equalize_hist(
                adata.X.get().copy().toarray(),
                entrys // adata.uns["config"]["scToolkit"]["hist_eq_divisor"])
        adata.X = x.squeeze()
    # #########################################################
    # Apply mask and normalize the data
    x = adata.X
    x = np.multiply(x, mask)
    x_sum = x.sum()
    x = x / x_sum * 1e4
    # #########################################################
    return adata
