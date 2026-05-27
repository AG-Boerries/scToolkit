'''These are the lowlevel functions

'''

# TODO: implement the scaling according to the median maximum value, not 10
# TODO: Check if sys.modules are only the loaded modules or if these are all
# TODO: []"pp"]["neighbors"]["n_pcs"] is somehow overwritten to numpy check origin

from . import (
    # Standard Library Imports
    os, compile, copy, Path, Counter, combinations, defaultdict, json_load,
    sub, re, warnings, normaltest, gzip,  # nb,
    # Typing
    Any, Sequence, Literal,
    # Joblib for parallel processing
    delayed, Parallel, parallel_config, ThreadPoolExecutor, as_completed,
    # Pandas and NumPy for data manipulation and analysis
    np, pd, pd_unique,
    # Scipy for scientific and technical computing
    csr_matrix, csc_matrix, issparse,
    stats_mode, map_coordinates, linkage, leaves_list,
    # Matplotlib for plotting and visualization
    plt,  # rcParams,
    # Scanpy and Anndata for single-cell data analysis
    sc, ad,
    # Sklearn
    adjusted_rand_score, MinMaxScaler, StandardScaler, gaussian_kde,
    distinctipy,
    # ##########################
    # scToolkit specifics
    # Custom modules from scToolkit
    get_logger, score_genes_efficient, replace_special_chars, ALL_PATHS,
    GENESET_FILENAMES,
    inplace_max_scale_csr, set_sparse_subset_to_zero, _compute_ratio_csr,
    get_colors_wrapped, bilinear_interpolate_numpy, get_fft_grid, save_dataframe,
    determine_clip_threshold, validate_groupby_column)

logger = get_logger(name="sc_utils")


# ###################################################################################################
# Setup functions
def initialize_cuda(
            devices: list[int] | None = None,
            managed_memory: bool = True,
            pool_allocator: bool = True
        ) -> None:
    """Initialize the GPU memory management system for CUDA-enabled devices.

    This function sets up the GPU environment by initializing memory management
    and allocators for CUDA-enabled devices. It allows for configuring memory
    management with options for managed memory and pool allocation, which can
    impact performance and memory usage. The function is designed to be flexible
    for different GPU setups, including those managed by SLURM.

    Args:
        devices (list[int] | None, optional): List of GPU device IDs to
            initialize. If None, all available GPUs will be used. Defaults to
            None.
        managed_memory (bool, optional): Whether to enable managed memory.
            Managed memory allows memory oversubscription at the cost of
            potential performance hits. Defaults to True.
        pool_allocator (bool, optional): Whether to use a memory pool allocator
            for performance optimization. Defaults to True.

    Returns:
        None:
            This function modifies global CUDA memory allocator state.

    Raises:
        ImportError: If the required packages are not installed.

    Called By:
        setup_adata

    Tags:
        utils
    """
    # #########################################################
    # Import necessary libraries for CUDA memory management
    from rmm import reinitialize  # noqa: E501
    from rmm.allocators.cupy import rmm_cupy_allocator  # noqa: E501
    from cupy.cuda import set_allocator  # noqa: E501
    from numba import cuda  # noqa: E501
    # #########################################################
    # Determine devices to use for initialization
    if devices is None:
        # ##########################################
        # Use all available GPU devices if none are specified
        devices = [c.id for c in cuda.gpus]
    # #########################################################
    # Reinitialize GPU memory management with the specified options
    reinitialize(
        managed_memory=managed_memory,
        pool_allocator=pool_allocator,
        devices=devices,
        initial_pool_size=2**30)
    # Set the memory allocator for CuPy to the RMM allocator
    set_allocator(rmm_cupy_allocator)


def setup_image_folder(
            config: dict[str, dict[str, str | bool]],
            part: str,
            return_path: bool = False
        ) -> str | None:
    """
    Creates a directory for saving images, with optional overwrite and path return.

    This function generates a directory at the path specified by the
    configuration and part name. If the directory already exists and the
    'overwrite' option is enabled in the config, the existing directory is
    deleted and a new one is created. Optionally, the function can return the
    path to the created directory.

    Args:
        config (dict[str): A configuration dictionary containing general
            settings for the experiment. The dictionary should include a
            'general' key with sub-keys 'save_path' and 'overwrite'.

                - 'save_path' (str): The base path where the directory should be
                  created.
                - 'overwrite' (bool): A flag indicating whether to overwrite the
                  directory if it exists.

        part (str): The name of the directory to be created within the save
            path.
        return_path (bool, optional): If True, the function returns the path to
            the created directory. Default is False. Defaults to False.

    Returns:
        str | None:
            Path to the image directory if ``return_path`` is True, else
            None.

    Raises:
        OSError: If there is an error in creating the directory or removing the
            existing one.

    Called By:
        all_qc_preprocessing, run_downstream

    Tags:
        config, io
    """
    # #########################################################
    # Construct the full path to the image folder based on config and part
    path_to_images = f"{config['general']['save_path']}figures{part}"
    # #########################################################
    # Check if the directory exists and if overwrite is enabled
    if os.path.exists(path_to_images) and config["general"]["overwrite"]:
        # ##########################################
        # If the directory exists and overwrite is True, remove the directory
        os.system(f"rm -r {path_to_images}")
    # #########################################################
    # Optionally return the path to the created directory
    if return_path:
        return path_to_images


# ###################################################################################################
# Automatic error/warning/raising handlin
def check_key_conflicts(
            adata: ad.AnnData,
            key: str,
            obsm_keys: list[str] | tuple[str, ...] | None = None,
            varm_keys: list[str] | tuple[str, ...] | None = None,
            var_keys_only: bool = False,
            extra_msg: str = "Please ensure this is intentional to avoid ambiguity."
        ) -> str | None:
    """Check if a key is present in multiple adata locations.

    Args:
        adata (anndata.AnnData): Adata object.
        key (str): The key to check.
        obsm_keys (list[str] | tuple[str, optional): Keys to check inside
            adata.obsm.
        varm_keys (list[str] | tuple[str, optional): Keys to check inside
            adata.varm.
        var_keys_only (bool, optional): If True, only consider var and varm.
            Defaults to False.
        extra_msg (str, optional): Additional text appended to conflict message.
            Defaults to "Please ensure this is intentional to avoid ambiguity.".

    Returns:
        str | None:
            str or None: Conflict message if found in multiple locations, else None.

    Called By:
        handle_key_conflicts

    Tags:
        obs, utils, var
    """
    presence = {
        "obs": False,
        "obs_names": False,
        "obsm": False,
        "var": False,
        "var_names": False,
        "varm": False}

    if not var_keys_only:
        # Check obs columns
        presence["obs"] = key in adata.obs.columns
        # Check var_names
        presence["var_names"] = key in adata.obs_names

        if obsm_keys is not None:
            for obsm_key in obsm_keys:
                if obsm_key in adata.obsm and isinstance(adata.obsm[obsm_key], pd.DataFrame):
                    if key in adata.obsm[obsm_key].columns:
                        presence["obsm"] = True
                        break
    else:
        # Check var columns
        presence["var"] = key in adata.var.columns
        # Check obs_names
        presence["obs_names"] = key in adata.var_names

        # Check varm keys
        if varm_keys is not None:
            for varm_key in varm_keys:
                if varm_key in adata.varm:
                    varm_entry = adata.varm[varm_key]
                    if isinstance(varm_entry, pd.DataFrame) and key in varm_entry.columns:
                        presence["varm"] = True
                        break
                    elif isinstance(varm_entry, np.ndarray):
                        # Can't resolve column names for ndarray varm
                        pass

    # Count how many places found
    found_count = sum(presence.values())
    if found_count > 1:
        msg = (
            f'Key "{key}" found in multiple locations: '
            f'{", ".join([loc for loc, present in presence.items() if present])}. '
            f'{extra_msg}')
        return msg
    return None


def handle_key_conflicts(
            adata: ad.AnnData,
            keys: list[str],
            obsm_keys: list[str] | tuple[str, ...] | None = None,
            varm_keys: list[str] | tuple[str, ...] | None = None,
            var_keys_only: bool = False,
            mode: str = "warn",
            log_level: str = "warning",
            **kwargs
        ) -> None:
    """Handle key conflicts in adata attributes using various reporting modes.

    Checks for naming conflicts between ``keys`` and existing adata fields
    (``.var``, ``.varm``, ``.obsm``) and handles them based on the specified ``mode``.

    Args:
        adata (anndata.AnnData): Adata object containing the target keys.
        keys (list[str]): List of keys to check for conflicts.
        obsm_keys (list[str] | tuple[str): Keys to check in ``.obsm``. Ignored if
            None.
        varm_keys (list[str] | tuple[str): Keys to check in ``.varm``. Ignored if
            None.
        var_keys_only (bool, optional): If True, only ``.var`` and ``.varm`` are
            checked. Defaults to False.
        mode (str, optional): One of "warn", "raise", "log", or "print".
            Determines how to handle conflicts. Defaults to "warn".
        log_level (str, optional): Logging level used if mode is "log".
            Defaults to "warning".
        **kwargs: Additional keyword arguments passed to ``check_key_conflicts``.

    Returns:
        None:
            This function is used for its side effects (warnings, exceptions,
            logs, etc.).

    Raises:
        ValueError: If ``mode`` is "raise" and a conflict is found.
        ValueError: If ``mode`` is "log" and no logger is provided in ``kwargs``.
        ValueError: If ``log_level`` is not a valid method of the given logger.
        ValueError: If ``mode`` is not one of the supported values.

    Calls:
        check_key_conflicts

    Called By:
        get_adata_sub_keys

    Tags:
        obs, utils, var
    """
    for key in keys:
        conflict_msg = check_key_conflicts(
            adata,
            key,
            obsm_keys=obsm_keys,
            varm_keys=varm_keys,
            var_keys_only=var_keys_only,
            **kwargs)
        if conflict_msg:
            if mode == "raise":
                raise ValueError(conflict_msg)
            elif mode == "warn":
                logger.warning(conflict_msg)
            elif mode == "log":
                if logger is None:
                    raise ValueError('Logger instance must be provided when mode is "log".')
                log_func = getattr(logger, log_level, None)
                if not callable(log_func):
                    raise ValueError(f'Invalid log level "{log_level}" for the provided logger.')
                log_func(conflict_msg)
            elif mode == "print":
                print(conflict_msg)  # THIS SHOULD BE A PRINT, DON'T CHANGE!
            else:
                raise ValueError(f'Invalid mode "{mode}". Choose "warn", "raise", "log", or "print".')


# ###################################################################################################
# Multimodal data
def get_adata_by_mod(
            adata: ad.AnnData,
            mod: str,
            adata_r: ad.AnnData | None = None,
            adata_p: ad.AnnData | None = None
        ) -> ad.AnnData:
    """Select the appropriate modality from the provided adata objects.

    This function is useful in situations where you need to handle different
    modalities of single-cell data (such as RNA and protein data) within a loop.
    Instead of writing separate loops for each modality, this function allows
    you to specify the desired modality and retrieve the corresponding adata
    object.

    NOTE:
        A dictionary of adata objects can do the same, this is old code.

    Args:
        adata (anndata.AnnData): Adata object, typically representing a combined
            or primary modality.
        mod (str): The modality identifier. Acceptable values are 'combined',
            'rna', or 'prot'.
        adata_r (anndata.AnnData | None, optional): The adata object for gene
            expression (RNA). Defaults to None.
        adata_p (anndata.AnnData | None, optional): The adata object for protein
            expression (Citeseq). Defaults to None.

    Returns:
        anndata.AnnData:
            adata object corresponding to the specified modality.

    Raises:
        ValueError: If ``mod`` is not one of 'combined', 'rna', or 'prot'.
        ValueError: If the requested modality is not available (None).

    Called By:
        run_downstream

    Tags:
        utils
    """
    # #########################################################
    # Check if both adata_r and adata_p are None
    if adata_r is None and adata_p is None:
        # ##########################################
        # If both are None, return the main adata object
        # since it's the only modality available
        return adata
    # #########################################################
    # Handle the 'combined' modality case
    if mod == "combined" and adata is not None:
        return adata
    # #########################################################
    # Handle the 'rna' modality case
    if mod == "rna" and adata_r is not None:
        return adata_r
    # #########################################################
    # Handle the 'prot' modality case
    if mod == "prot" and adata_p is not None:
        return adata_p


# ###########################################################################################################
# Filtering
def filter_cells_multicall(
            adata: ad.AnnData,
            min_counts: int | None = None,
            min_genes: int | None = None,
            max_counts: int | None = None,
            max_genes: int | None = None,
            inplace: bool = True,
        ) -> ad.AnnData | None:
    """
    Filters cells in an adata object based on multiple criteria, using
    scanpy's filtering functions.

    NOTE:
        This is deprecated, due to slow speed of the scanpy functions, use the
        automated all_filters_tagging.

    This function serves as a wrapper around ``scanpy.pp.filter_cells``, allowing
    for multiple filtering criteria (e.g., ``min_counts``, ``min_genes``,
    ``max_counts``, ``max_genes``) to be applied in a single call. The filtering can
    be performed either in-place or by returning a new adata object.

    Args:
        adata (anndata.AnnData): Adata object containing the single-cell data to
            be filtered.
        min_counts (int | None, optional): Minimum number of counts required for
            a cell to be retained. Defaults to None.
        min_genes (int | None, optional): Minimum number of genes required for a
            cell to be retained. Defaults to None.
        max_counts (int | None, optional): Maximum number of counts allowed for
            a cell to be retained. Defaults to None.
        max_genes (int | None, optional): Maximum number of genes allowed for a
            cell to be retained. Defaults to None.
        inplace (bool, optional): If True, performs filtering in-place,
            modifying the original adata object. If False, returns a new adata
            object with the filtering applied. Defaults to True.

    Returns:
        anndata.AnnData or None: Returns the filtered adata object if ``inplace``
            is False or ``copy`` is True. Otherwise, returns None.

    Calls:
        get_shape_diff

    Called By:
        all_filters

    Tags:
        QC, obs, stats
    """
    if not inplace:
        adata = adata.copy()

    for key, value in {
            "min_counts": min_counts,
            "min_genes": min_genes,
            "max_counts": max_counts,
            "max_genes": max_genes}.items():
        if value is not None:
            shape_before = adata.shape
            sc.pp.filter_cells(adata, **{key: value}, inplace=True)
            get_shape_diff(adata, f"pp_filter_cells_{key}", shape_before, adata.shape)

    return None if inplace else adata


def filter_genes_multicall(
            adata: ad.AnnData,
            min_counts: int | None = None,
            min_cells: int | None = None,
            max_counts: int | None = None,
            max_cells: int | None = None,
            inplace: bool = True,
            copy: bool = False
        ) -> ad.AnnData | None:
    """Apply multiple gene filtering criteria to an adata object.

    This function acts as a wrapper for ``scanpy.pp.filter_genes`` to support
    multiple filtering criteria in a single call. It filters genes in the adata
    object based on minimum and maximum counts or cells. The filtering is done
    in-place by default but can be configured to return a copy.

    NOTE:
        This is deprecated, due to slow speed of the scanpy functions, use the
        automated all_filters_tagging.

    Args:
        adata (anndata.AnnData): Adata object of shape ``n_obs`` x ``n_vars``.
            Rows correspond to cells and columns to genes.
        min_counts (int | None, optional): Minimum number of counts required for
            a gene to be retained. Defaults to None.
        min_cells (int | None, optional): Minimum number of cells required for a
            gene to be retained. Defaults to None.
        max_counts (int | None, optional): Maximum number of counts allowed for
            a gene to be retained. Defaults to None.
        max_cells (int | None, optional): Maximum number of cells allowed for a
            gene to be retained. Defaults to None.
        inplace (bool, optional): Whether to perform the filtering in-place or
            return a new adata object. Defaults to True.
        copy (bool, optional): If True, return a copy of the adata object with
            filtered genes, otherwise return None. Defaults to False.

    Returns:
        anndata.AnnData | None:
            anndata.AnnData or None: If ``copy=True``, returns a new adata
            object with the filtered genes. Otherwise, it modifies the input
            adata object and returns None.

    Raises:
        ValueError: If ``inplace=False`` and ``copy=False`` are both set, as at
            least one of these must be True to retain the filtered data.

    Calls:
        get_shape_diff

    Called By:
        all_filters

    Tags:
        QC, stats, var
    """
    if not inplace:
        adata = adata.copy()

    for key, value in {
            "min_counts": min_counts,
            "min_cells": min_cells,
            "max_counts": max_counts,
            "max_cells": max_cells}.items():
        if value is not None:
            shape_before = adata.shape
            sc.pp.filter_genes(adata, **{key: value}, inplace=True)
            get_shape_diff(adata, f"pp_filter_genes_{key}", shape_before, adata.shape)

    return None if inplace else adata


def get_shape_diff(
            adata: ad.AnnData,
            key: str,
            shape_before: list[int] | tuple[int, int],
            shape_after: list[int] | tuple[int, int]
        ) -> None:
    """
    Calculates the difference in shape (number of cells and genes) before and
    after a filtering operation and updates the corresponding statistics in
    the adata object.

    This function updates or creates an entry in the ``adata.uns["stats"]``
    dictionary with the difference in the number of cells and genes before and
    after a filtering operation. If the ``key`` already exists in
    ``adata.uns["stats"]``, the difference is added to the existing values.
    Otherwise, a new entry is created with the calculated difference.

    Args:
        adata (anndata.AnnData): Adata object.
        key (str): The key in ``adata.uns["stats"]`` that should be updated or
            created.
        shape_before (list[int] | tuple[int): A list or tuple containing the
            shape (n_cells, n_genes) before the filtering operation.
        shape_after (list[int] | tuple[int): A list or tuple containing the
            shape (n_cells, n_genes) after the filtering operation.

    Returns:
        None

    Called By:
        Run_all_prep_steps_clustering, all_filters, filter_cells_multicall,
        filter_genes_multicall, filter_genes_n_families,
        filter_genes_n_families_regex

    Tags:
        stats
    """
    # #########################################################
    # Check if the key already exists in adata.uns["stats"]
    if key in adata.uns["stats"].keys():
        # ##########################################
        # Update existing key with the difference in shape
        adata.uns["stats"][key] = [
            v1 + v2 for v1, v2 in zip(
                adata.uns["stats"][key],
                [shape_before[0] - shape_after[0], shape_before[1] - shape_after[1]])]
        # ##########################################
    else:
        # ##########################################
        # Create a new key with the difference in shape
        adata.uns["stats"][key] = [
            shape_before[0] - shape_after[0],
            shape_before[1] - shape_after[1]]


def parse_geneset_to_mask(
            geneset: str,
            var_names: pd.Index
        ) -> np.ndarray:
    """
    Parses a geneset string containing explicit gene blocks and regex and
    returns a combined boolean mask.

    The geneset string can contain explicit gene blocks in the form
    ``|^GENE$|``, which are used for exact matches. The remaining part of the
    string is treated as a regex pattern.

    Args:
        geneset (str): Geneset string containing explicit gene blocks and/or
            regex.
        var_names (pandas.Index): Gene names to match against (e.g.,
            adata.var_names).

    Returns:
        numpy.ndarray:
            Combined boolean mask indicating matching genes.

    Raises:
        re.error: If the regular expression in the geneset is invalid.

    Called By:
        filter_genes_n_families_regex, flag_gene_family

    TODO:
        Add handling for edge cases like malformed gene blocks or duplicate
        delimiters.

    Tags:
        annotation, var
    """
    # Add delimiters and replace single pipes with double pipes
    geneset_double = "|" + geneset.replace("|", "||") + "|"

    # Find explicit gene blocks
    genes = re.findall(r'\|\^([A-Za-z0-9_-§]+)\$\|', geneset_double)

    # Remove gene blocks from regex string
    pure_regex = re.sub(r'\|\^([A-Za-z0-9_-§]+)\$\|', '', geneset_double).replace("||", "|")[1:-1]

    # Create isin mask
    isin_mask = var_names.isin(genes)

    # Create regex mask only if pure regex is non-empty
    if pure_regex.strip():
        regex_mask = var_names.str.contains(pure_regex)
    else:
        regex_mask = np.zeros(len(var_names), dtype=bool)

    # Return combined mask
    return isin_mask | regex_mask


def unmark_highly_variable_regex(
            adata: ad.AnnData,
            regex: str
        ) -> ad.AnnData:
    """
    Unmarks genes as highly variable in an adata object based on a given regex
    pattern.

    This function searches for gene names in the adata object (``adata``) that
    match a specified regular expression pattern (``regex``). If such genes are
    found and they are marked as highly variable, the function will unmark them
    and update the stats in the ``adata.uns`` dictionary accordingly.

    Args:
        adata (anndata.AnnData): Adata object containing gene expression data,
            where highly variable genes are marked in the ``var`` DataFrame.
        regex (str): A regular expression pattern used to identify gene names
            that should be unmarked as highly variable. Default is an empty
            string.

    Returns:
        anndata.AnnData:
            The modified adata object with updated highly variable
            gene status.

    Raises:
        KeyError: If the ``highly_variable`` column is not found in ``adata.var``.

    Tags:
        stats, var
    """
    # #########################################################
    # Identify genes that match the provided regex pattern
    genes_to_exclude = adata.var_names.str.contains(regex)
    # logger.debug(f"{genes_to_exclude.sum()} Genes will be excluded")
    # #########################################################
    # Check if "highly_variable_genes_excluded" key exists in adata.uns["stats"]
    if "highly_variable_genes_excluded" not in adata.uns["stats"].keys():
        adata.uns["stats"]["highly_variable_genes_excluded"] = []
    # #########################################################
    # Process and unmark genes that are currently marked as highly variable
    if genes_to_exclude.sum() >= 1:
        gene_names = adata.var_names[genes_to_exclude]
        # ##########################################
        # Filter to only include genes that are currently marked as highly variable
        gene_names = [x for x in gene_names if x in adata.var_names[adata.var["highly_variable"]]]
        # Update the stats in adata.uns
        adata.uns["stats"]["highly_variable_genes_excluded"].append(gene_names)
        # ##########################################
        # Unmark the identified genes as highly variable
        for g in gene_names:
            adata.var.highly_variable.at[g] = False
        # ##########################################
        # logger.debug(adata.var["highly_variable"].sum())
    return adata


def flag_gene_family(
            adata: ad.AnnData,
            prot: bool = False,
            regex: bool = True
        ) -> None:
    """
    Flags genes in the ``adata`` object as belonging to the gene identifiers
    specified in the configuration file (``config["qc"]["gene_identifiers"]``).

    This function marks genes in the adata object (``adata``) as part of specific
    gene families or identifiers, depending on whether Citeseq data is included
    and whether regular expressions (regex) are used. The function checks for
    empty gene identifiers and removes them from the analysis to ensure that
    only relevant gene sets are flagged.

    Args:
        adata (anndata.AnnData): Adata object.
        prot (bool, optional): Indicates whether Citeseq data is included.
            Defaults to False.
        regex (bool, optional): Determines whether the identifiers in
            ``config["qc"]["gene_identifiers"]`` should be treated as regular
            expressions. If False, the identifiers are considered to be specific
            gene sets with gene symbols. Defaults to True.

    Returns:
        None

    Raises:
        Warning: If any gene identifiers are empty, a warning is logged, and the
            empty identifiers are removed from the analysis.

    Calls:
        parse_geneset_to_mask

    Called By:
        all_qc_preprocessing

    Tags:
        annotation, config, var
    """
    # #########################################################
    # Initial setup: prepare to process gene identifiers and handle potential empty identifiers
    identifiers_to_remove = []
    if not prot:
        # ##########################################
        # Loop through each identifier in the configuration and check if genes match the identifier
        for identifier, geneset in adata.uns["config"]["qc"]["gene_identifiers"].items():
            if regex:
                if isinstance(geneset, list):
                    is_identifier = adata.var_names.isin(geneset)
                else:
                    # OLD and slow:
                    # Check if genes match the regex pattern in the identifier
                    # is_identifier = adata.var_names.str.contains(geneset)
                    # NEW and fast:
                    is_identifier = parse_geneset_to_mask(geneset, adata.var_names)
            else:
                # Check if genes match the gene set provided by the identifier
                is_identifier = adata.var_names.isin(geneset)
            # ##########################################
            # Check if the identifier is empty and add it to the removal list if it is
            if sum(is_identifier) == 0:
                identifiers_to_remove.append(identifier)
            else:
                # Flag genes as part of the identifier set in the adata object
                adata.var[identifier] = is_identifier
    # #########################################################
    # Clean up: remove any empty identifiers from the configuration and log a warning
    if len(identifiers_to_remove) != 0:
        logger.warning(
            f'Gene set identifiers {" ".join(identifiers_to_remove)} '
            'is/are empty, deleting it from the analysis')
    for identifier in identifiers_to_remove:
        # Remove the empty identifier from the configuration to avoid errors in downstream analysis
        del adata.uns["config"]["qc"]["gene_identifiers"][identifier]
        adata.uns["config"]["pp"]["calculate_qc_metrics"]["qc_vars"].remove(identifier)


def filter_genes_n_families_regex(
            adata: ad.AnnData,
            config: dict | None = None
        ) -> ad.AnnData:
    """
    Filters out genes from an adata object based on a regular expression provided
    in the configuration. Optionally, if ``config["general"]["only_mask_genes"]``
    is True, genes are not removed but only flagged.

    This function is designed to remove specific gene families from the adata
    object based on a regular expression pattern. The configuration for
    filtering can either be passed directly or fetched from the adata object's
    ``uns`` attribute.

    Args:
        adata (anndata.AnnData): Adata object containing gene expression data to
            be filtered.
        config (dict | None, optional): A dictionary containing the key
            ``exclude_gene_family_regex`` which is a list containing plain gene
            names and/or regex patterns. If not provided, it will use the
            ``config`` key from the ``adata.uns`` attribute. Default is None.

    Returns:
        anndata.AnnData:
            A filtered or flagged adata object depending on configuration.

    Calls:
        get_shape_diff, parse_geneset_to_mask

    Called By:
        Run_all_prep_steps_clustering

    TODO:
        Filter genes and ``regex``, to speed up the process!

    .. code-block:: python

        def is_regex(s):
            regex_chars = [
                '.', '*', '+', '?', '^', '$',
                '(', ')', '[', ']', '{', '}', '|']
            return any(char in s for char in regex_chars)

        regex_ = "$|^".join([
            x for x in adata.uns["config"]["exclude_gene_family_regex"].split("$|^")
            if is_regex(x)
        ])

        non_regex_genes = [
            x for x in adata.uns["config"]["exclude_gene_family_regex"].split("$|^")
            if not is_regex(x)]

    Tags:
        QC, config, stats, var
    """
    # #########################################################
    # Check if the configuration is provided, otherwise fetch it from the adata object
    if config is None:
        config = adata.uns["config"]
    # #########################################################
    # Build combined mask from list of plain genes and regexes
    is_identifier = np.zeros(len(adata.var_names), dtype=bool)
    for pattern in config["exclude_gene_family_regex"]:
        if isinstance(pattern, list):
            mask = adata.var_names.isin(pattern)
        else:
            mask = parse_geneset_to_mask(pattern, adata.var_names)
        is_identifier |= mask

    genes_to_exclude = adata.var_names[is_identifier].tolist()
    # #########################################################
    # If there are any genes to exclude, proceed with filtering
    if len(genes_to_exclude) > 0:
        if config.get("general", {}).get("only_mask_genes", False):
            # Just flag genes instead of removing them
            adata.var["gene_family_filtered"] = is_identifier
        else:
            # Remove genes
            shape_before = adata.shape
            adata = adata[:, ~is_identifier].copy()
            get_shape_diff(adata, "filter_gene_fams", shape_before, adata.shape)
            if "stats" in adata.uns.keys():
                if "pre_clustering_gene_exclustion" not in adata.uns["stats"].keys():
                    adata.uns["stats"]["pre_clustering_gene_exclustion"] = genes_to_exclude
                else:
                    adata.uns["stats"]["pre_clustering_gene_exclustion"].extend(genes_to_exclude)
    # #########################################################
    return adata


def filter_genes_n_families(
            adata: ad.AnnData,
            config: dict | None = None
        ) -> ad.AnnData:
    """
    Filters out specified gene families and individual genes from an adata object.

    This function modifies the input adata object by removing gene families and
    specific genes as defined in a configuration dictionary. It ensures that the
    resulting adata object only contains the desired genes for further analysis.

    Args:
        adata (anndata.AnnData): Adata object. The data
            is filtered based on the configuration.
        config (dict | None, optional): A dictionary specifying the gene
            families and individual genes to exclude. If not provided, the
            function uses the configuration stored in ``adata.uns["config"]``.
            Defaults to None.

    Returns:
        anndata.AnnData:
            The filtered adata object with specified genes and gene
            families removed.

    Raises:
        NotImplementedError: Raised if the function attempts to filter
            individual genes using an incomplete or untested implementation.

    Calls:
        get_shape_diff

    Called By:
        Run_all_prep_steps_clustering

    Tags:
        QC, config, var
    """
    # #########################################################
    # Assign the configuration if it is not provided
    if config is None:
        config = adata.uns["config"]
    # #########################################################
    # Filter out specified gene families
    for g in config["exclude_gene_family"]:
        # ##########################################
        # Check if the gene family exists in the dataset
        if sum(adata.var_names.str.startswith(g)) >= 1:
            logger.debug(f"exclude gene family: {g}")
            shape_before = adata.shape
            # Filter out genes belonging to the specified gene family
            adata = adata[:, adata.var_names[~adata.var_names.str.startswith(g)]].copy()
            get_shape_diff(adata, f"filter_gene_fam_{g}", shape_before, adata.shape)
    # #########################################################
    # Filter out specified individual genes
    shape_before = adata.shape
    for g in config["exclude_genes"]:
        # ##########################################
        # Check if the individual gene exists in the dataset
        if sum(adata.var_names.str.startswith(g)) == 1:
            logger.debug(f"exclude gene: {g}")
            logger.debug(adata.shape)
            # The filtering process for individual genes is not fully implemented
            adata = adata[:, adata.var_names[~adata.var_names.str.startswith(g)]].copy()
            logger.debug(adata.shape)
        raise NotImplementedError
    # #########################################################
    # Log the shape change if any genes were filtered
    if shape_before != adata.shape:
        get_shape_diff(adata, "filter_genes", shape_before, adata.shape)

    # Return the filtered adata object
    return adata


# ###########################################################################################################
# Random Genes and Generators
def get_random_generator(
            adata: ad.AnnData,
            seed: int | None = None
        ) -> np.random.Generator:
    """Returns a random number generator based on the provided seed value.

    NOTE:
        If seed is None, it will use the adata.uns["config"]["general"]["seed"].
        If seed is -1, the function will not fix the seed and will generate a
        new random seed. If seed is a positive natural number, it will use it as
        a seed.

    Args:
        adata (anndata.AnnData): Adata object,
            expected to contain a seed value in
            adata.uns["config"]["general"]["seed"].
        seed (int | None, optional): Seed value for the random number generator.
            Special cases:
            - If None, uses adata.uns["config"]["general"]["seed"].
            - If -1, does not fix the seed.
            - If a positive integer, uses it as the seed. Defaults to None.

    Returns:
        numpy.random.Generator:
            A random number generator initialized based on
            the provided or inferred seed.

    Called By:
        get_proper_random_ref, get_random_genes, get_random_geneset_reference,
        get_ref_gensests, subset_adata_random

    Tags:
        config, utils
    """
    # #########################################################
    # Determine the random number generator based on the seed
    if seed == -1:
        # ##########################################
        # No fixed seed: initialize with a new random seed
        rnd_gen = np.random.default_rng()
        # ##########################################
    elif seed is None:
        # ##########################################
        # Use the seed from adata's configuration
        rnd_gen = np.random.default_rng(seed=adata.uns["config"]["general"]["seed"])
    else:
        # ##########################################
        # Use the provided seed value
        rnd_gen = np.random.default_rng(seed=seed)
    # #########################################################
    return rnd_gen


def get_random_genes(
            adata: ad.AnnData,
            n_genes: int = 100,
            highly_variable: bool = True,
            exclude_highly_variable: bool = False,
            seed: int = None,
            rnd_gen: np.random.Generator | int | None = None
        ) -> list[str]:
    """
    Get random genes from an adata object, with options for selecting highly
    variable genes.

    This function selects a specified number of genes from an adata object, with
    options to filter based on whether the genes are highly variable or not. It
    can use a seed for reproducibility.

    NOTE:
        This function is deterministic; if run with the same seed (except for -1),
        it will return the same genes. See the doc of get_random_generator for details.

    Args:
        adata (anndata.AnnData): Adata object.
        n_genes (int, optional): Number of genes to return. Defaults to 100.
        highly_variable (bool, optional): If True, sample from highly variable
            genes only. Defaults to True.
        exclude_highly_variable (bool, optional): If True, sample from
            non-highly variable genes only. Defaults to False.
        seed (int, optional): Seed for random number generator. Defaults to None.
        rnd_gen (numpy.random.Generator | int | None, optional): Predefined
            random generator. If provided, the seed option will be ignored.
            Defaults to None.

    Returns:
        list[str]:
            A list of selected random genes from the adata object.

    Calls:
        get_highly_variable, get_random_generator

    Called By:
        get_random_geneset_reference

    Tags:
        utils, var
    """
    # #########################################################
    # Create the random generator based on the provided seed or use the provided random generator
    if rnd_gen is None:
        # ##########################################
        # If rnd_gen is not provided, create it using the provided or default seed
        rnd_gen = get_random_generator(adata, seed=seed)
    # #########################################################
    # Select the pool of genes based on highly variable and exclusion flags
    if highly_variable:
        # ##########################################
        # If highly_variable is True, get highly variable genes from the adata object
        genes = get_highly_variable(adata, return_adata=False, return_genes=True)
    elif exclude_highly_variable:
        # ##########################################
        # If exclude_highly_variable is True, exclude highly variable genes from the selection
        genes = np.setxor1d(
            adata.var_names,
            get_highly_variable(adata, return_adata=False, return_genes=True))
    else:
        # ##########################################
        # If no specific filtering, use all available genes
        genes = adata.var_names
    # #########################################################
    # Return a random selection of the specified number of genes
    return rnd_gen.choice(genes, n_genes)


def get_random_geneset_reference(
            adata: ad.AnnData,
            genesets: dict,
            n_genesets: int | None = None,
            use_only_geneset: bool = False,
            highly_variable: bool = True,
            exclude_highly_variable: bool = False,
            dist_sample: str = "uniform",
            seed: int = -1,
            inplace: bool = True,
            rnd_key: str = "rnd",
            rnd_gen: np.random.Generator | None = None
        ) -> dict | None:
    """Creates random genesets based on a reference geneset list.

    This function generates random genesets by sampling genes from the provided
    adata object. The size of each geneset is determined based on the
    distribution specified, and the genes can be sampled from the entire set or
    restricted to highly variable genes. The function can update the original
    genesets in place or return the new random genesets.

    NOTE:
        - If ``highly_variable``, only sample from the highly variable genes.
        - If ``use_only_geneset``, only use the genes from the provided genesets.
        - If ``seed`` is None, it will use
          ``adata.uns["config"]["general"]["seed"]``.
        - If ``seed`` is -1, don't fix the seed at all.

    Args:
        adata (anndata.AnnData): Adata object from which to get the genes.
        genesets (dict): Dictionary of marker genes, with geneset names as keys
            and gene lists as values.
        n_genesets (int | None, optional): Number of random genesets to
            generate. Defaults to None.
        use_only_geneset (bool, optional): If True, only draw from the genes of
            the provided genesets. Defaults to False.
        highly_variable (bool, optional): If True, sample only from highly
            variable genes. Defaults to True.
        exclude_highly_variable (bool, optional): If True, sample only from
            non-highly variable genes. Defaults to False.
        dist_sample (str, optional): Distribution from which to sample the
            geneset sizes. Options are ["uniform", "normal", "same"]. Defaults
            to "uniform".
        seed (int, optional): Seed for random number generation. Defaults to -1.
        inplace (bool, optional): If True, update the original genesets with the
            random genesets. Defaults to True.
        rnd_key (str, optional): The prefix for the random geneset names.
            Defaults to "rnd".
        rnd_gen (numpy.random.Generator | None, optional): Predefined random
            number generator. Defaults to None.

    Returns:
        dict | None:
            If ``inplace`` is False, returns a dictionary of new random
            genesets. If ``inplace`` is True, returns the updated genesets.

    Calls:
        get_all_markers_from_ref_dict, get_random_generator,
        get_random_genes

    Called By:
        get_ref_gensests

    TODO:
        Maybe add an option for non-inplace operation, returning the new
        genesets separately.

    Tags:
        utils, var
    """
    # #########################################################
    # Initialize the random generator
    if rnd_gen is None:
        rnd_gen = get_random_generator(adata, seed=seed)
    # #########################################################
    # Extract necessary information from the genesets
    # Get the lengths of the genesets
    geneset_lengths = [len(x) for x in genesets.values()]
    # ##########################################
    # Get all marker genes from the genesets
    marker_genes = get_all_markers_from_ref_dict(genesets, unique=True)
    # ##########################################
    # Set the number of genesets to generate if not provided
    if n_genesets is None:
        n_genesets = len(genesets)
    # #########################################################
    # Determine how to sample the geneset sizes
    min_ = min(geneset_lengths)
    # ##########################################
    # Sample geneset sizes based on the specified distribution
    if dist_sample == "uniform":
        max_ = max(geneset_lengths)
        rnd_geneset_lengths = rnd_gen.integers(min_, max_, n_genesets)
    elif dist_sample == "normal":
        mean_, std_ = np.mean(geneset_lengths), np.std(geneset_lengths)
        rnd_geneset_lengths = rnd_gen.normal(loc=mean_, scale=std_, size=n_genesets).astype(int)
        rnd_geneset_lengths = np.where(rnd_geneset_lengths < 2, min_, rnd_geneset_lengths)
    elif dist_sample == "same":
        rnd_geneset_lengths = rnd_gen.choice(geneset_lengths, n_genesets)
    else:
        rnd_geneset_lengths = geneset_lengths
    # #########################################################
    # Generate the random genesets
    rnd_genesets = {}
    for i, l in enumerate(rnd_geneset_lengths):
        if use_only_geneset:
            rnd_genesets[f'{rnd_key}_{i}'] = rnd_gen.choice(marker_genes, l)
        else:
            rnd_genesets[f'{rnd_key}_{i}'] = get_random_genes(
                adata, n_genes=l, highly_variable=highly_variable,
                exclude_highly_variable=exclude_highly_variable, rnd_gen=rnd_gen)
    # #########################################################
    # Update the original genesets if inplace, otherwise return the new genesets
    if inplace:
        # ##########################################
        # Check for overlapping geneset names and adjust if necessary
        intersecting_keys = np.intersect1d(list(rnd_genesets.keys()), list(genesets.keys()))
        while len(intersecting_keys) != 0:
            rnd_genesets = {f'{k}_{i}': rnd_genesets[k] for i, k in enumerate(intersecting_keys)}
            intersecting_keys = np.intersect1d(list(rnd_genesets.keys()), list(genesets.keys()))
        genesets.update(rnd_genesets)
    else:
        return rnd_genesets


# ###########################################################################################################
# CMO/Hash
def calc_hash_features(
            adata: ad.AnnData,
            hashes_keys: list[str],
            separation_key: str = "condition",
            obsm_key: str = "hashes",
            logs: list[str] = ["log10", "log1p"],
            norms: list[str] = ["norm", "lognorm", "norm_separately"],
            include_raw: bool = False,
            raw_rescale: bool = False
        ) -> None:
    """
    Compute log transforms and normalizations for hash keys in adata
    and store in ``obsm``.

    NOTE:
        THIS IS UNFINISHED, DON'T USE IT!
        Uses log1p z-score based outlier detection for clipping during
        group-wise rescaling.

    Args:
        adata (anndata.AnnData): Adata object with hash keys in ``obs``.
        hashes_keys (list[str]): List of columns in ``obs`` to process.
        separation_key (str, optional): Grouping column in ``obs``.
            Defaults to "condition".
        obsm_key (str, optional): Key to store output DataFrame in ``obsm``.
            Defaults to "hashes".
        logs (list[str], optional): Log transforms to apply.
            Defaults to ["log10".
        norms (list[str], optional): Normalizations to apply on each log.
            Defaults to ["norm".
        include_raw (bool, optional): Whether to include raw values.
            Defaults to False.
        raw_rescale (bool, optional): Whether to rescale raw and log values per
            group. Defaults to False.

    Returns:
        None:
            Modifies ``adata`` in-place by adding a DataFrame to ``adata.obsm``.

    Calls:
        determine_clip_threshold
    """
    hash_features: dict[str, np.ndarray] = {}
    obs_df = adata.obs

    for hash_key in hashes_keys:
        raw_vals = obs_df[hash_key].values.copy()

        # Optionally include raw values
        if include_raw:
            if raw_rescale:
                rescaled_vals = np.zeros_like(raw_vals)
                for group in obs_df[separation_key].unique():
                    mask = obs_df[separation_key] == group
                    group_vals = raw_vals[mask]

                    clip_thresh = determine_clip_threshold(group_vals)
                    group_vals = np.clip(group_vals, None, clip_thresh)

                    group_min, group_max = group_vals.min(), group_vals.max()
                    if group_max > group_min:
                        scaled = (group_vals - group_min) / (group_max - group_min)
                    else:
                        scaled = np.zeros_like(group_vals)

                    rescaled_vals[mask] = scaled

                hash_features[f"{hash_key}_raw_norm"] = rescaled_vals
            else:
                hash_features[f"{hash_key}_raw"] = raw_vals

        # Global normalization on raw
        if "norm" in norms:
            norm_vals = raw_vals / raw_vals.sum()
            norm_vals = (norm_vals - norm_vals.min()) / (norm_vals.max() - norm_vals.min())
            hash_features[f"{hash_key}_norm"] = norm_vals

        # Iterate logs
        for log_method in logs:
            if log_method == "log10":
                log_vals = np.log10(raw_vals)
            elif log_method == "log1p":
                log_vals = np.log1p(raw_vals)
            else:
                raise ValueError(f"Unsupported log method: {log_method}")

            log_vals[~np.isfinite(log_vals)] = 0
            key_base = f"{hash_key}_{log_method}"
            hash_features[key_base] = log_vals

            # Rescale logs if requested
            if raw_rescale:
                rescaled_vals = np.zeros_like(log_vals)
                for group in obs_df[separation_key].unique():
                    mask = obs_df[separation_key] == group
                    group_vals = log_vals[mask]

                    clip_thresh = determine_clip_threshold(group_vals)
                    group_vals = np.clip(group_vals, None, clip_thresh)

                    group_min, group_max = group_vals.min(), group_vals.max()
                    if group_max > group_min:
                        scaled = (group_vals - group_min) / (group_max - group_min)
                    else:
                        scaled = np.zeros_like(group_vals)

                    rescaled_vals[mask] = scaled

                hash_features[f"{key_base}_norm"] = rescaled_vals

            if "lognorm" in norms:
                lognorm_vals = log_vals / log_vals.sum()
                lognorm_vals = (lognorm_vals - lognorm_vals.min()) / (lognorm_vals.max() - lognorm_vals.min())
                hash_features[f"{key_base}_lognorm"] = lognorm_vals

            if "norm_separately" in norms:
                norm_sep_vals = np.full(adata.n_obs, -100000.0, dtype=float)
                for group in obs_df[separation_key].unique():
                    mask = obs_df[separation_key] == group
                    group_vals = log_vals[mask]
                    group_min, group_max = group_vals.min(), group_vals.max()
                    if group_max > group_min:
                        scaled = (group_vals - group_min) / (group_max - group_min)
                    else:
                        scaled = np.zeros_like(group_vals)
                    norm_sep_vals[mask] = scaled

                hash_features[f"{key_base}_norm_separately"] = norm_sep_vals

    feature_df = pd.DataFrame(hash_features, index=adata.obs.index)
    adata.obsm[obsm_key] = feature_df


def calc_clr(
            adata: ad.AnnData,
            columns: str | list[str] = "H-2Kb",
            suffix: str = "_clr",
            inplace: bool = True,
            subset_by: str | None = None
        ) -> ad.AnnData | dict[str, np.ndarray] | None:
    """
    Computes the centered log-ratio (clr) transformation for specified
    column(s) in the adata object's obs, optionally based on subsets of
    the data.

    NOTE:
        - Ensure that no gene has only 0 counts!
        - Ensure that the column names specified in ``columns`` exist in
          ``adata.obs``.

    Args:
        adata (anndata.AnnData): Adata object.
        columns (str | list[str], optional): The column(s) in the adata object's
            obs to transform. Default is 'H-2Kb'. Defaults to "H-2Kb".
        suffix (str, optional): The suffix to add to the new column name(s).
            Default is '_clr'. Defaults to "_clr".
        inplace (bool, optional): Whether to modify the adata object in place or
            return the transformed data. Default is True. Defaults to True.
        subset_by (str | None, optional): The column to use for subsetting the
            data before calculating the clr. If None, calculates clr without
            subsetting. Default is None. Defaults to None.

    Returns:
        anndata.AnnData | dict[str, numpy.ndarray] | None:
            If inplace is True, modifies the adata object
            in place and returns None. If inplace is False, returns a dictionary
            with column names as keys and numpy arrays of the clr-transformed
            data as values.

    .. doctest::

        >>> adata = AnnData(pd.DataFrame(
        >>>     {'H-2Kb': [1, 2, 3, 4, 5], 'H-2Db': [2, 3, 4, 5, 6],
        >>>     'sample': ['A', 'A', 'B', 'B', 'B']}))
        >>> calc_clr(adata, columns=['H-2Kb', 'H-2Db'])
        >>> print(adata.obs)
        Index  H-2Kb  H-2Db sample  H-2Kb_clr  H-2Db_clr
        0      1      2      A  -0.434294  -0.263034
        1      2      3      A   0.000000   0.263034
        2      3      4      B  -0.602060  -0.342423
        3      4      5      B  -0.124939  -0.060206
        4      5      6      B   0.150515   0.207579

    Raises:
        ValueError: If any of the specified columns are not in the adata
            object's obs.
        ValueError: If subset_by is specified but is not in the adata object's
            obs.
    """
    # #########################################################
    # Initial setup and validation of input parameters
    if isinstance(columns, str):
        columns = [columns]
    # ##########################################
    # Check for missing columns in the data
    missing_columns = [col for col in columns if col not in adata.obs.keys()]
    if missing_columns:
        raise ValueError(f"The following columns are not in adata.obs.keys(): {missing_columns}")
    # ##########################################
    # Check if the subset_by column exists if specified
    if subset_by and subset_by not in adata.obs.keys():
        raise ValueError(f"The subset_by column '{subset_by}' is not in adata.obs.keys()")
    # #########################################################
    # Initialize container for transformed data if not inplace
    transformed_data = {}
    # #########################################################
    # Perform clr transformation based on whether subsetting is used
    if subset_by:
        # ##########################################
        # Handle subsetting if subset_by is specified
        subsets = adata.obs[subset_by].unique()
        for subset in subsets:
            subset_mask = adata.obs[subset_by] == subset
            for column in columns:
                geometric_mean = np.exp(np.mean(np.log1p(adata.obs.loc[subset_mask, column])))
                clr_values = np.log1p(adata.obs.loc[subset_mask, column] / geometric_mean)

                if inplace:
                    adata.obs.loc[subset_mask, f'{column}{suffix}'] = clr_values
                else:
                    if column not in transformed_data:
                        transformed_data[column] = np.zeros_like(adata.obs[column])
                    transformed_data[column][subset_mask] = clr_values
    else:
        # ##########################################
        # Handle clr transformation without subsetting
        for column in columns:
            geometric_mean = np.exp(np.mean(np.log1p(adata.obs[column])))
            clr_values = np.log1p(adata.obs[column] / geometric_mean)

            if inplace:
                adata.obs[f'{column}{suffix}'] = clr_values
            else:
                transformed_data[column] = clr_values
    # #########################################################
    # Return the result if not inplace
    if not inplace:
        return transformed_data


def run_hashsolo_and_add_results(
            adata: ad.AnnData,
            this_hashes: list[str],
            split_hashes: list[list[str]] | None = None,
            demultiplexing_key: str = "demultiplexing_hash_solo",
            per_sample_key: str = "hashsolo_per_sample",
            condition_obs_key: str = "condition",
            hash_solo_classification_key: str = "Classification",
            obsm_key: str | None = None,
            overwrite: bool = False,
            regex_remove: str | None = None,
            categorize_cols: bool = True,
            **kwargs,
        ) -> None:
    """Run HashSolo demultiplexing on adata object and store results.

    Args:
        adata (anndata.AnnData): Adata object.
        this_hashes (list[str]): List of all hashes to run on full and
            per-condition data.
        split_hashes (list[list[str]] | None, optional): Nested list specifying
            additional hash groups (e.g., [['CMO301', 'CMO302'], ['CMO303',
            'CMO304']]). If None, no split hash groups are processed. Defaults
            to None.
        demultiplexing_key (str, optional): Key in obsm to store results.
            Defaults to "demultiplexing_hash_solo".
        per_sample_key (str, optional): Key for per-condition assignments.
            Defaults to "hashsolo_per_sample".
        condition_obs_key (str, optional): Key in obs specifying condition.
            Defaults to "condition".
        hash_solo_classification_key (str, optional): HashSolo classification
            key (used for sample and overall). Defaults to "Classification".
        obsm_key (str | None, optional): If set, hash features are read from
            adata.obsm[obsm_key]. Defaults to None.
        overwrite (bool, optional): If True, overwrite existing
            demultiplexing_key in obsm. Defaults to False.
        regex_remove (str | None, optional): Regex pattern to remove from hash
            names in output columns. Defaults to None.
        categorize_cols (bool, optional): Categorize the output columns.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to ``hashsolo``, with keys
            already used in this function (like 'inplace') automatically
            removed.

    Returns:
        None

    Tags:
        annotation, obs
    """
    # ############################################
    # Error handling
    if demultiplexing_key not in adata.obsm:
        adata.obsm[demultiplexing_key] = pd.DataFrame(index=adata.obs_names)
    elif overwrite:
        logger.info(f'Overwriting adata.obsm[{demultiplexing_key}]')
        adata.obsm[demultiplexing_key] = pd.DataFrame(index=adata.obs_names)
    else:
        raise ValueError(
            f"The key '{demultiplexing_key}' already exists in adata.obsm. "
            "Please choose a new key or remove the existing entry first.")

    if obsm_key:
        if obsm_key not in adata.obsm:
            raise ValueError(f"obsm_key '{obsm_key}' not found in adata.obsm.")
        missing = [h for h in this_hashes if h not in adata.obsm[obsm_key].columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in adata.obsm['{obsm_key}'].")
    else:
        missing = [h for h in this_hashes if h not in adata.obs.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in adata.obs.")

    if condition_obs_key not in adata.obs.columns:
        raise ValueError(f"condition_obs_key '{condition_obs_key}' not found in adata.obs.")
    # ############################################
    # Extract minimal data for hashsolo
    if obsm_key:
        hash_adata = ad.AnnData(obs=adata.obsm[obsm_key][this_hashes])
    else:
        hash_adata = ad.AnnData(obs=adata.obs[this_hashes])
    hash_adata.obs[condition_obs_key] = adata.obs[condition_obs_key]
    # ############################################
    # Per-condition assignments
    for condition in hash_adata.obs[condition_obs_key].unique():
        temp_data = sc.external.pp.hashsolo(
            hash_adata[hash_adata.obs[condition_obs_key] == condition, :],
            this_hashes,
            inplace=False,
            **kwargs)
        adata.obsm[demultiplexing_key].loc[temp_data.obs_names, per_sample_key] = (
            temp_data.obs[hash_solo_classification_key])
    # ############################################
    # All hashes on full data
    temp_data = sc.external.pp.hashsolo(hash_adata, this_hashes, inplace=False, **kwargs)
    adata.obsm[demultiplexing_key]["hashsolo"] = temp_data.obs[hash_solo_classification_key]
    # ############################################
    # Split hashes (low/high groups)
    if split_hashes:
        for i, hash_group in enumerate(split_hashes):
            groupby = f"hashsolo_group_{i+1}"
            temp_data = sc.external.pp.hashsolo(hash_adata, hash_group, inplace=False, **kwargs)
            adata.obsm[demultiplexing_key][groupby] = temp_data.obs[hash_solo_classification_key]
    # ############################################
    # Remove the generics like log or whatever
    if regex_remove:
        for col in adata.obsm[demultiplexing_key].columns:
            unique_vals = adata.obsm[demultiplexing_key][col].unique()
            mapping = {v: re.sub(regex_remove, "", v) for v in unique_vals if isinstance(v, str)}
            adata.obsm[demultiplexing_key][col] = adata.obsm[demultiplexing_key][col].replace(mapping)
    if categorize_cols:
        adata.obsm[demultiplexing_key] = adata.obsm[demultiplexing_key].astype("category")


def fix_hash_out_of_distribution(
            adata: ad.AnnData,
            this_hashes: list[str],
            obsm_key: str | None = None,
            groupby: str | None = None,
            to_add: list = ["raw", "log", "log_adj", "log_adj_unlog"],
            delete_from_obs: bool = True,
            target_df: pd.DataFrame | None = None,
        ) -> None:
    """Fix hash out-of-distribution effects by mode-based log adjustment.

    NOTE:
        This function expects raw antibody data!

    For each hash in ``this_hashes``, this function:
        - Creates log1p-transformed versions of the raw counts.
        - Computes mode differences to the highest mode hash.
        - Shifts log values so all modes match the max.
        - Clips adjusted logs to a global outlier threshold.
        - Exponentiates back to obtain adjusted unlogged counts.

    Args:
        adata (anndata.AnnData): Adata object.
        this_hashes (list[str]): List of hash columns in adata.obs to adjust.
        obsm_key (str | None, optional): If set, each of the groups will be
            rescaled separatly. Defaults to None.
        groupby (str | None, optional): Optional key to group data by, must be a
            column in adata.obs None will use the
            config["general"]["cluster_algorithm"]. Defaults to None.
        to_add (list, optional): Which columns to add to the target DataFrame.
            Options are "raw", "log", "log_adj", "log_adj_unlog". Defaults to
            ["raw".
        delete_from_obs (bool, optional): Whether to delete original
            ``this_hashes`` columns from adata.obs after processing. Defaults to
            True.

    Returns:
        None

    Calls:
        determine_clip_threshold

    Tags:
        normalization, obs
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    # #########################################################
    # Compute log1p
    log_data = np.log1p(adata.obs[this_hashes])
    # #########################################################
    # If mutliple datasets are provided normalize each hash first
    if groupby is not None:
        log_data[groupby] = adata.obs[groupby].copy()
        for h in this_hashes:
            # this_log_data = log_data[h][adata.obs[groupby] == k].copy()
            this_log_data = log_data[[h, groupby]].copy()
            meds = this_log_data.groupby(
                groupby, observed=False
                ).agg(lambda x: x.mode().mean())
            max_mode = meds.max()
            meds_add = max_mode - meds
            for k in adata.obs[groupby].unique():
                subsetter_ = adata.obs[groupby] == k
                log_data.loc[subsetter_, h] = (
                   log_data.loc[subsetter_, h] + meds_add.T[k].item())
        del log_data[groupby]
    # #########################################################
    # Compute medians and adjustments
    # meds = pd.Series(np.median(log_data.values, axis=0), index=this_hashes)
    # Compute mode and adjustments
    meds = log_data.mode().T[0]
    meds.index = this_hashes
    max_mode = meds.max()
    meds_add = max_mode - meds
    # #########################################################
    # Global outlier clip threshold
    global_outlier_threshold = determine_clip_threshold(log_data.values.flatten())
    # #########################################################
    # Prepare target dataframe
    if obsm_key is not None:
        if obsm_key not in adata.obsm:
            adata.obsm[obsm_key] = pd.DataFrame(index=adata.obs_names)
        target_df = adata.obsm[obsm_key]
    else:
        target_df = adata.obs
    # #########################################################
    # Adjust, clip, and unlog
    if groupby is not None:
        for h in this_hashes:
            log_vals = log_data[h].copy()
            adj_vals = log_vals + meds_add[h]
            adj_vals_clipped = np.clip(adj_vals, None, global_outlier_threshold)
            unlog_vals = np.expm1(adj_vals_clipped).astype(int)

            # Add to target_df if requested
            if "log" in to_add:
                target_df[f"{h}_log_group_adjusted"] = log_vals
            if "log_adj" in to_add:
                target_df[f"{h}_log_mode_group_adjusted"] = adj_vals_clipped
            if "log_adj_unlog" in to_add:
                target_df[f"{h}_log_mode_group_adjusted_unlog"] = unlog_vals
        # And reset to save the raw as raw
        log_data = np.log1p(adata.obs[this_hashes])

    for h in this_hashes:

        log_vals = log_data[h].copy()
        adj_vals = log_vals + meds_add[h]
        adj_vals_clipped = np.clip(adj_vals, None, global_outlier_threshold)
        unlog_vals = np.expm1(adj_vals_clipped).astype(int)

        # Add to target_df if requested
        if "raw" in to_add:
            target_df[h] = adata.obs[h]
        if "log" in to_add:
            target_df[f"{h}_log"] = log_vals
        if "log_adj" in to_add:
            target_df[f"{h}_log_mode_adjusted"] = adj_vals_clipped
        if "log_adj_unlog" in to_add:
            target_df[f"{h}_log_mode_adjusted_unlog"] = unlog_vals
        if groupby is not None:
            target_df[f"{h}_group_adjusted"] = log_data[h]

    # #########################################################
    # Optionally delete original raw hash columns
    if delete_from_obs:
        adata.obs.drop(columns=this_hashes, inplace=True)


# ###########################################################################################################
# Helpers
def get_save_path(
            save_path: str,
            config: dict,
            part: str = "/downstream/",
            full_path: bool = False
        ) -> str | bool:
    """
    Constructs and returns the path for saving images based on the provided
    configuration.

    This function checks whether the saving of figures is enabled in the
    provided ``config``. If saving is enabled, it constructs the directory path
    where figures should be saved. Depending on the ``full_path`` flag, it either
    returns the base path or the full path including the image directory.

    NOTE:
        This function modifies the Scanpy configuration to set the directory
        for saving figures.

    Args:
        save_path (str): The base path to be used for saving figures.
        config (dict): A configuration dictionary containing the general
            settings for saving figures.
        part (str, optional): A string to be appended to the base save path.
            Defaults to "/downstream/".
        full_path (bool, optional): Flag indicating whether to return the full
            path including the figure directory. Defaults to False.

    Returns:
        str | bool:
            The constructed path if saving is enabled in the config,
            otherwise False.

    Called By:
        continuos_umap_helper, discrete_umap_helper, plot_mediods_heatmap,
        plot_per_group_DEG_dotplot_n_gene_dendrogram,
        plot_per_group_stacked_violins, plot_qc_scatter,
        plot_qc_scatter_combinations, plot_qc_violins, plot_ref_dotplots,
        plot_ref_stacked_violins, run_downstream

    TODO:
        Implement error handling for invalid paths and configuration entries.

    Tags:
        config, io
    """
    # #########################################################
    # Check if saving figures is enabled in the config
    if config["general"]["save_figures"]:
        # ##########################################
        # Construct the path to the image directory
        path_to_images = f"{config['general']['save_path']}figures{part}"
        # Set the Scanpy configuration to use the constructed directory
        sc._settings.ScanpyConfig.figdir = Path(path_to_images)
        # ##########################################
        # Return the full path or just the base path based on the full_path flag
        if not full_path:
            return save_path
        else:
            return f"{path_to_images}{save_path}"
        # ##########################################
    else:
        # Return False if saving figures is not enabled
        return False


def get_genes_sum_sorted(
            adata: ad.AnnData,
            descending: bool = True,
            return_counts: bool = False
        ) -> np.ndarray | dict[str, float]:
    """
    Returns the variable names (genes) sorted by the sum of their values across
    all cells.

    This function takes an adata object, computes the sum of each gene's
    expression values across all cells, and returns the gene names sorted by
    these sums. The sorting order can be ascending or descending based on the
    input arguments. Additionally, it can return a dictionary of the gene names
    and their corresponding sums if specified.

    Args:
        adata (anndata.AnnData): Adata object.
        descending (bool, optional): If True, sorts the genes in descending
            order based on their sums. Default is False (ascending order).
            Defaults to True.
        return_counts (bool, optional): If True, returns a dictionary where keys
            are sorted gene names and values are their corresponding sums.
            Default is False. Defaults to False.

    Returns:
        numpy.ndarray | dict[str, float]:
            np.ndarray or dict: If ``return_counts`` is False, returns an array
            of gene names sorted by their summed expression values across all
            cells. If ``return_counts`` is True, returns a dictionary of sorted
            gene names and their corresponding sums.

    TODO:
        Consider adding functionality to handle more complex sorting criteria,
        such as sorting by specific subsets of cells or additional metadata.

    Tags:
        calculation, var
    """
    # #########################################################
    # Calculate the sum of each gene's expression values across all cells
    # and sort the genes based on these sums.
    # ##########################################
    # Compute the sum across all cells for each gene
    gene_sums = adata.X.sum(0) if isinstance(adata.X, np.ndarray) else adata.X.sum(0).A1
    # ##########################################
    # Determine the sorting order based on the ``descending`` argument
    if descending:
        sorted_gene_indices = np.array(np.argsort(-gene_sums)).flatten()
    else:
        sorted_gene_indices = np.array(np.argsort(gene_sums)).flatten()

    sorted_gene_names = adata.var_names[sorted_gene_indices]
    # ##########################################
    # Return the sorted gene names, and optionally, a dictionary of the sorted sums
    if return_counts:
        sorted_gene_sums = gene_sums[sorted_gene_indices]
        gene_counts_dict = dict(zip(sorted_gene_names, sorted_gene_sums))
        return gene_counts_dict
    else:
        return sorted_gene_names


def convert_mixed_types(
            df: pd.DataFrame,
            fix_multitype: bool = False,
            categorical_threshold: int = 10,
            float_dtype: str = "float64",
            int_dtype: str = "Int64",
            unknown_str: str = "unknown",
            just_report: bool = False
        ) -> pd.DataFrame:
    """Separate mixed-type columns and apply consistent typing rules.

    For columns that contain multiple data types, this function creates one new
    column for each data type, except for string columns containing only the
    value 'unknown'. The original column is then dropped from the DataFrame.

    NOTE:
        - For columns with mixed types:
            - If fix_multitype is True, separate into multiple columns.
            - If fix_multitype is False, convert to 'object' type and fill NaN
              with 'unknown'.
        - For columns with consistent types:
            - Convert to 'Int64' for integers with NaN preserved.
            - Convert to 'float64' for floats.
            - Convert to 'category' if unique string values are below
              categorical_threshold.
            - Convert to 'object' and fill NaN with 'unknown' for other types.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be
            processed.
        fix_multitype (bool, optional): If True, separates mixed types into
            multiple columns. If False, converts the column to 'object' type and
            fills NaN values with 'unknown'. Default is False. Defaults to
            False.
        categorical_threshold (int, optional): The threshold for unique string
            values below which a column is converted to 'category' type. Default
            is 10. Defaults to 10.
        float_dtype (str, optional): The dtype to which float values should be
            converted (e.g., 'float64'). Default is 'float64'. Defaults to
            "float64".
        int_dtype (str, optional): The dtype to which integer values should be
            converted (e.g., 'Int64'). Default is 'Int64'. Defaults to "Int64".
        unknown_str (str, optional): The string to use for unknown values,
            typically 'unknown'. Default is 'unknown'. Defaults to "unknown".

    Returns:
        pandas.DataFrame:
            The modified DataFrame with separate columns for each data type.

.. doctest::

    >>> data = {
    >>>     'cell_id': ['cell1', 'cell2', 'cell3'],
    >>>     'cluster': [1, 2, 'unknown'],
    >>>     'cluster_2': [1, 2, np.NaN],
    >>>     'quality': ['good', 'poor', np.NaN],
    >>>     'bla': [.1, .2, np.NaN],
    >>> }
    >>> obs_df = pd.DataFrame(data)
    >>> obs_df.set_index('cell_id', inplace=True)
    >>> obs_df = convert_mixed_types(obs_df, fix_multitype=True)

    Called By:
        save_h5ad

    Tags:
        utils
    """
    # #########################################################
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # ##########################################
        # Identify unique data types present in the column
        unique_types = set(type(x) if pd.notnull(x) else type(np.nan) for x in df[col])
        # print(col, unique_types)
        # print(df[col].dtype.name)
        was_categorical = False
        if str in unique_types:
            was_categorical = True

        if just_report:
            logger.info(col, unique_types)
            logger.info(df[col].dtype.name)
            continue
        # #########################################################
        # Handle columns with mixed data types
        if len(unique_types) > 1:
            if type(np.nan) in unique_types:
                unique_types.remove(type(np.nan))
            # ##########################################
            # If multiple types exist and fix_multitype is True
            if len(unique_types) > 1:
                if fix_multitype:
                    # Separate each dtype into one column per dtype
                    for dtype in unique_types:
                        new_col_name = f"{col}_{dtype.__name__}"
                        # ##########################################
                        # Separate numeric types (int, float) into new columns
                        if pd.api.types.is_numeric_dtype(dtype):
                            if dtype is int:
                                df[new_col_name] = df[col].apply(
                                    lambda x: x if isinstance(x, dtype) or pd.isnull(x) else np.nan
                                ).astype(int_dtype)
                            elif dtype is float:
                                df[new_col_name] = df[col].apply(
                                    lambda x: x if isinstance(x, dtype) or pd.isnull(x) else np.nan
                                ).astype(float_dtype)
                        # ##########################################
                        # Handle non-numeric types, replacing unknowns
                        else:
                            df[new_col_name] = df[col].apply(
                                lambda x: x if isinstance(x, dtype) and not pd.isnull(x) else unknown_str)
                            # If the new column only contains 'unknown', delete it
                            if len(df[new_col_name].unique()) == 1 and df[new_col_name].unique()[0] == unknown_str:
                                del df[new_col_name]
                            # Convert it to categorical if it was
                            elif was_categorical:
                                df[new_col_name] = df[new_col_name].astype("category")
                    # Drop the original mixed-type column
                    df.drop(columns=[col], inplace=True)
                # ##########################################
                # Convert mixed-type column to object and fill NaN with 'unknown'
                else:
                    df[col] = df[col].astype('object').fillna(unknown_str)
            # #########################################################
            # Handle columns with a consistent data type
            else:
                single_type = next(iter(unique_types))
                if single_type is int:
                    df[col] = df[col].fillna(pd.NA).astype(int_dtype)
                elif single_type is float:
                    df[col] = df[col].astype(float_dtype)
                elif single_type is bool:
                    df[col] = df[col].astype(pd.BooleanDtype())  # allows NA values
                elif single_type is str:
                    df[col] = df[col].astype(str).fillna(unknown_str)
                    if df[col].nunique() < categorical_threshold or was_categorical:
                        df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('object').fillna(unknown_str)

    if not just_report:
        return df


def create_obs_range(
            adata: ad.AnnData,
            in_key: str,
            out_key: str,
            ranges: list[float]
        ) -> None:
    """Categorizes continuous values in an adata object into discrete ranges.

    This function takes continuous data from a specified column in the
    ``adata.obs`` dataframe, segments it into discrete ranges (bins), and assigns
    corresponding labels to those bins. The ranges are defined by the user, and
    values that fall outside the specified range are categorized as either
    '-inf' or 'inf' based on their magnitude.

    NOTE:
        This function assumes that the input data (``adata.obs[in_key]``) is
        numeric and can be categorized into continuous bins.

    Args:
        adata (anndata.AnnData): Adata object.
        in_key (str): The key in the ``adata.obs`` dataframe from which values
            will be binned.
        out_key (str): The key in the ``adata.obs`` dataframe where the binned
            values and their corresponding labels will be stored.
        ranges (list[float]): A list of floats that defines the boundaries of
            the bins. These values are used to categorize the continuous data
            into labeled ranges.

    Returns:
        None:
            This function modifies the ``adata.obs`` dataframe in place.

    Raises:
        KeyError: If ``in_key`` is not found in the ``adata.obs`` dataframe.
        ValueError: If the ``ranges`` list is empty or contains non-numeric
            values.

    TODO:
        Add support for custom labels beyond automatic formatting.

    Tags:
        annotation, obs
    """
    # #########################################################
    # Check if all elements in ranges are integers
    all_int = all(isinstance(r, int) for r in ranges)
    # #########################################################
    # Create bins using numpy.inf to represent infinity boundaries
    # Bins are built by adding -inf at the start and inf at the end of the ranges list
    bins = [-np.inf] + ranges + [np.inf]
    # #########################################################
    # Generate labels for the bins
    # Labels take the form of "lower-upper" for each bin, handling infinity explicitly
    # ##########################################
    # Generate labels for the bins, adjusting format if ranges are integers
    # If the upper bound of a bin is infinity, label it with '-inf'
    if all_int:
        labels = [f'{bins[i]}-{bins[i+1]}' if bins[i+1] != np.inf
                  else f'{bins[i]}-inf' for i in range(len(bins) - 1)]
    else:
        labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' if bins[i+1] != np.inf
                  else f'{bins[i]:.2f}-inf' for i in range(len(bins) - 1)]
    # #########################################################
    # Apply the pandas cut function to categorize the continuous data into the bins
    # The resulting labels will be stored in the ``out_key`` column of ``adata.obs``
    adata.obs[out_key] = pd.cut(adata.obs[in_key], bins=bins, labels=labels, right=False)
    # #########################################################
    # Remove empty categories
    adata.obs[out_key] = adata.obs[out_key].cat.remove_unused_categories()


def interpolate_density_to_embedding(
            adata: ad.AnnData,
            density: np.ndarray,
            method: str = "nearest",
            embedding_key: str = "X_umap",
            key: str = "interpolated_density",
            # grid_size: int = 1024,
        ) -> None:
    """
    Interpolate a 2D density grid into the embedding space of an adata object.

    This projects the given ``density`` grid onto the embedding defined in
    ``adata.obsm[embedding_key]``, using either fast nearest-neighbor
    interpolation (``map_coordinates``) or custom bilinear interpolation. The
    result is stored in ``adata.obs[key]``.

    NOTE:
        Internally, ``get_fft_grid`` is used to construct the transformation from
        real-valued embedding coordinates into grid index space. The
        interpolation is then performed in that index space against the provided
        density map.

    Args:
        adata (anndata.AnnData): Adata object with embedding coordinates in
            ``.obsm``.
        density (numpy.ndarray): Either a 2D array representing the density grid
            or a key in ``adata.uns`` from which the density grid will be
            retrieved.
        method (str, optional): Interpolation method to use. One of:
            - "nearest": Uses scipy.ndimage.map_coordinates (fast).
            - "interpolate": Uses custom NumPy bilinear interpolation.
            Defaults to "nearest".
        embedding_key (str, optional): Key in ``adata.obsm`` for embedding
            coordinates. Defaults to "X_umap".
        key (str, optional): Name of the column in ``adata.obs`` where the
            interpolated values will be stored. Defaults to
            "interpolated_density".

    Returns:
        None

    Calls:
        bilinear_interpolate_numpy, get_fft_grid

    Tags:
        obs, visualization
    """
    # If ``density`` is a string, treat it as a key to adata.uns
    if isinstance(density, str):
        if density not in adata.uns:
            raise ValueError(f"Density key '{density}' not found in adata.uns.")
        density = adata.uns[density]

    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm.")

    if density.ndim != 2:
        raise ValueError("Density must be a 2D array.")

    coords = adata.obsm[embedding_key]
    grid = get_fft_grid(coords, grid_size=density.shape[0])

    # Normalize embedding to grid index space (float index coordinates)
    x_scaled = (coords[:, 0] - grid.x_min) * grid.x_scaler_co_to_gi
    y_scaled = (coords[:, 1] - grid.y_min) * grid.y_scaler_co_to_gi

    if method == "nearest":
        grid_coords = np.vstack([y_scaled, x_scaled])
        interpolated = map_coordinates(density, grid_coords, order=1, mode="nearest")

    elif method == "interpolate":
        # Manual bilinear interpolation
        interpolated = bilinear_interpolate_numpy(density, x_scaled, y_scaled)

    else:
        raise ValueError("Method must be one of {'nearest', 'interpolate'}")

    adata.obs[key] = interpolated


def _validate_thresholds(
            plot_thresholds: list[tuple[float | None, float | None]] | None,
            keys: list[str],
        ) -> list[tuple[float | None, float | None]] | None:
    """Validate and normalize per-key thresholds.

    Args:
        plot_thresholds:
            List of (lower, upper) pairs aligned with ``keys``. Each entry may
            be ``None`` to skip. May itself be ``None``.
        keys:
            Keys to be plotted (after removing groupby if needed).

    Returns:
        list[tuple[float | None, float | None]] | None:
            List of (lower, upper) tuples for each key, or ``None`` if no
            thresholds provided.

    Raises:
        ValueError:
            If ``plot_thresholds`` length mismatches ``keys``, or elements are
            not (lower, upper) pairs of numeric-or-None.

    Called By:
        plot_violin

    """
    if plot_thresholds is None:
        return None

    if not isinstance(plot_thresholds, (list, tuple)):
        raise ValueError("plot_thresholds must be a list of (lower, upper) pairs.")

    if len(plot_thresholds) != len(keys):
        raise ValueError(
            f"{len(plot_thresholds)} must equal len(keys) after removing any "
            "groupby key from 'keys'.")

    thresholds_by_key: list[tuple[float | None, float | None]] = []
    for th in plot_thresholds:
        if th is None:
            thresholds_by_key.append((None, None))
            continue
        if not isinstance(th, (list, tuple)) or len(th) != 2:
            raise ValueError(
                "Each plot_thresholds element must be a (lower, upper) pair or None.")
        lower, upper = th
        if ((lower is not None and not np.isscalar(lower)) or
                (upper is not None and not np.isscalar(upper))):
            raise ValueError("Threshold values must be numeric or None.")
        thresholds_by_key.append((lower, upper))

    return thresholds_by_key


def get_thresholds(
            keys: list[str],
            filters: dict[str, int | float] | None,
            adata: ad.AnnData | None = None,
            use_obs: bool = True,
            qc_config_key: str = "keys_for_qc",
        ) -> list[list[int | float | None]]:
    """Build [[min, max], ...] thresholds for each key using a filter dict.

    The function looks for 'min_<key>' and 'max_<key>' entries in ``filters``
    for every ``key`` in ``keys``. Missing entries become ``None``. Values
    must be numeric (int or float, but not bool) and not NaN.

    NOTE:
        This is a helper for the qc violins to plot the already set thresholds.

    Args:
        keys (list[str] | None): Ordered collection of metric names
            (e.g., "n_counts").
        filters (dict[str, int | float] | None): Mapping that may contain
            'min_<key>' and/or 'max_<key>'.
        adata (anndata.AnnData, optional): Adata object with config after qc
            preprocessing.
        use_obs (bool, optional): If True use the obs, else use the var.
        qc_config_key (str, optional): The key of adata.uns["config"]["general"]
            to use (keys_for_qc or keys_for_qc_all). Defaults to "keys_for_qc".

    Returns:
        list[list[int | float | None]]:
            A list of [min_value_or_None, max_value_or_None] pairs, aligned with
            the order of ``keys``.

    Raises:
        TypeError:
            If ``keys`` is not a sequence of strings or ``filters`` is
            not a mapping.
        ValueError:
            If a relevant filter value is non-numeric, is NaN, or if
            a min value is greater than its corresponding max.

    .. code-block:: python

        >>> keys = ["n_cells", "n_unique", "n_counts"]
        >>> filters = {
        ...     "min_n_cells": 2000,
        ...     "min_n_unique": 7,
        ...     "min_n_counts": 2000,
        ...     "max_n_counts": 10000,
        ... }
        >>> build_thresholds(keys, filters)
        [[2000, None], [7, None], [2000, 10000]]

    TODO:
        It would maybe also make sense to accept a dict.

    Called By:
        plot_qc_violins
    """
    # ###############################################################
    # keys and filters or adata handling.
    if keys is None and filters is None and adata is None:
        raise ValueError(
            "Either `keys` and `filters`, or `adata` must be provided.")
    if keys is None:
        if adata is None:
            raise ValueError("`adata` must be provided to infer keys.")
        if use_obs:
            keys = adata.uns["config"]["general"][qc_config_key]["obs"].copy()
        else:
            keys = adata.uns["config"]["general"][qc_config_key]["var"].copy()
    if filters is None and isinstance(adata, ad.AnnData):
        if adata is None:
            raise ValueError("`adata` must be provided to infer keys.")
        if use_obs:
            filters = adata.uns["config"]["pp"]["filter_qc_var_obs"].copy()
        else:
            filters = adata.uns["config"]["pp"]["filter_qc_var_var"].copy()
    # ###############################################################
    # keys and filter checks
    if not isinstance(filters, dict):
        raise TypeError("`filters` must be a dict.")
    if not isinstance(keys, list) or isinstance(keys, (str, bytes)):
        raise TypeError("`keys` must be a list of strings.")
    for idx, key in enumerate(keys):
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"`keys[{idx}]` must be a non-empty string; got {key!r}.")

    def _as_number(value: int | float, name: str) -> int | float:
        """Helper"""
        if isinstance(value, bool):
            raise ValueError(f"{name!r} must be numeric, not bool.")
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value is None:
                raise ValueError(f"{name!r} must not be NaN.")
            return value
        raise ValueError(f"{name!r} must be an int or float.")
    # ###############################################################
    # Create the thresholds
    thresholds: list[list[int | float | None]] = []
    for key in keys:
        min_key = f"min_{key}"
        max_key = f"max_{key}"

        min_val = None
        max_val = None

        if min_key in filters:
            min_val = _as_number(filters[min_key], min_key)
        if max_key in filters:
            max_val = _as_number(filters[max_key], max_key)
        if (
                min_val is not None
                and max_val is not None
                and min_val > max_val):
            raise ValueError(
                f"{min_key} ({min_val}) is greater than {max_key} ({max_val}).")

        thresholds.append([min_val, max_val])

    return thresholds


# ###########################################################################################################
# Easy printings
def show_num_genes_and_count_per_cell(
            adata: ad.AnnData,
            celllist: list[str]
        ) -> None:
    """
    Logs the number of genes and counts for each cell in the provided cell list.

    This function iterates over a list of cells, extracting the number of genes
    and the count per cell from the provided adata object and logs this
    information. It is particularly useful for exploring the gene expression
    data of specific cells within a single-cell RNA sequencing dataset.

    NOTE:
        This function assumes that ``adata`` is an adata object that contains
          ``n_genes`` and ``n_counts`` in its ``.obs`` attribute for each cell.

    Args:
        adata (anndata.AnnData): Adata object.
        celllist (list[str]): A list of cell identifiers for which the number of
            genes and counts will be logged.

    Returns:
        None

    Raises:
        KeyError: If any cell in ``celllist`` is not found in ``adata.obs``.

    Tags:
        obs
    """
    # #########################################################
    # Iterate over the list of cells provided in celllist
    for c in np.intersect1d(adata.obs_names, celllist):
        # ##########################################
        # Log the number of genes and counts for the current cell
        logger.info(
            f'Cell: {c}, n_genes: {adata.obs.n_genes[c]}, n_counts: {adata.obs.n_counts[c]}')


def compare_gene_expression_deltas(
            adata: ad.AnnData,
            gene: str,
            group: str,
            groupby: str,
            condition_obs_key: str = None,
            vs: tuple[str, str] | list[str, str] | None = None,
        ) -> None:
    """Compare gene expression in a group vs. all cells.

    Args:
        adata (anndata.AnnData): Adata object.
        gene (str): Gene name in ``adata.var_names``.
        group (str): Group label in ``adata.obs[groupby]`` to compare.
        groupby (str): Column in ``adata.obs`` defining the grouping.
        condition_obs_key (str, optional): Column in ``adata.obs`` with condition
            labels. Defaults to None.
        vs (tuple[str, optional): Pair of conditions to compare within the
            group.

    Returns:
        None

    Calls:
        validate_groupby_column

    Tags:
        calculation, groupby, obs
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
            adata.obs, groupby, check_categorical=True, groups=[group])

    valid_multi_comparison = condition_obs_key and vs
    # Check if condition_obs_key and vs are properly setup
    if valid_multi_comparison:
        validate_groupby_column(
            adata.obs, condition_obs_key, check_categorical=True, groups=vs)
        # Check if it is realy 2
        if not isinstance(vs, (tuple, list)) or len(vs) != 2:
            raise ValueError(
                "'vs' must be a tuple of exactly two condition labels.")
    # #########################################################
    # Subset to group
    adata_group = adata[adata.obs[groupby] == group]

    if valid_multi_comparison:
        if condition_obs_key not in adata.obs:
            raise KeyError(f"The specified key '{condition_obs_key}' was not found in the adata.obs.keys().")
        adata = adata[adata.obs[condition_obs_key] == vs[1]]
        adata_group = adata_group[adata_group.obs[condition_obs_key] == vs[0]]
    # #########################################################
    # Get the genes data
    x_all = adata[:, gene].X
    x_grp = adata_group[:, gene].X
    # with 0
    x_all = x_all.toarray().flatten() if hasattr(x_all, "toarray") else x_all.flatten()
    x_grp = x_grp.toarray().flatten() if hasattr(x_grp, "toarray") else x_grp.flatten()
    # Without 0
    x_all_nz = x_all[x_all > 0]
    x_grp_nz = x_grp[x_grp > 0]
    # #########################################################
    # If one of the groups is empty
    if x_all_nz.size == 0:
        if valid_multi_comparison:
            logger.info(f"Non-zero check: condition '{vs[0]}' has no non-zero {gene}")
        else:
            logger.info(f"Non-zero check: rest (excluding group '{group}') has no non-zero {gene}")

    if x_grp_nz.size == 0:
        if valid_multi_comparison:
            logger.info(f"Non-zero check: condition '{vs[1]}' has no non-zero {gene}")
        else:
            logger.info(f"Non-zero check: group '{group}' has no non-zero {gene}")
    # #########################################################
    # print results
    label = f"{group} vs rest" if not (valid_multi_comparison) else f"{vs[0]} vs {vs[1]}"
    logger.info(f"Δ{gene} (mean, all values) [{label}]:      {x_grp.mean() - x_all.mean():.4f}")
    logger.info(f"Δ{gene} (median, all values) [{label}]:    {np.median(x_grp) - np.median(x_all):.4f}")

    if x_all_nz.size > 0 and x_grp_nz.size > 0:
        logger.info(f"Δ{gene} (mean, non-zero only) [{label}]:   {x_grp_nz.mean() - x_all_nz.mean():.4f}")
        logger.info(f"Δ{gene} (median, non-zero only) [{label}]: {np.median(x_grp_nz) - np.median(x_all_nz):.4f}")
    else:
        logger.info(f"Δ{gene} (mean, non-zero only) [{label}]:   N/A (empty)")
        logger.info(f"Δ{gene} (median, non-zero only) [{label}]: N/A (empty)")
    return x_all, x_grp, x_all_nz, x_grp_nz


# ###########################################################################################################
# Efficiency and GPU Handling
def convert_to_mem_efficient(
            data: csr_matrix | pd.Series | np.ndarray,
            general_dtype: str = "int"
        ) -> csr_matrix | pd.Series | np.ndarray:
    """
    Converts the input data to the most memory-efficient dtype based on the
    specified general dtype.

    This function is designed to handle data types such as scipy sparse
    matrices, pandas series, and numpy arrays. The conversion process aims to
    minimize memory usage while ensuring that the data remains within the valid
    range of the target dtype.

    NOTE:
        Only valid for scipy sparse, pandas series, and numpy arrays

    Args:
        data (csr_matrix | pd.Series | np.ndarray): The input data to be
            converted. It can be a scipy sparse matrix, pandas series, or numpy
            array.
        general_dtype (str, optional): Specifies the general dtype for
            conversion. Default is "int". Can also be set to "float". Defaults
            to "int".

    Returns:
        csr_matrix | pandas.Series | numpy.ndarray:
            The converted data with the most memory-efficient dtype.
            If the conversion is not possible or the data type is
            not supported, the original data is returned.

    Raises:
        TypeError: If the input data type is not one of the supported types.

    Called By:
        all_qc_preprocessing

    TODO:
        General dtype automatic? Or is this too slow?

    Tags:
        sparse, utils
    """
    # #########################################################
    # Check if the input data is a supported type
    if not (isinstance(data, csr_matrix)
            or isinstance(data, pd.Series)
            or isinstance(data, np.ndarray)):
        logger.warning("Data type not implemented, Object remains unchanged!")
        return data
    # #########################################################
    # If general_dtype is 'int', determine the smallest suitable integer dtype
    if general_dtype == "int":
        dmin = data.min()  # Calculate the minimum value in the data
        dmax = data.sum()  # Calculate the maximum value in the data

        for dt in ["u2", "i2", "u4", "i4", "u8", "i8"]:
            # ##########################################
            # Check if data fits within the current dtype range
            if (dmin >= np.iinfo(np.dtype(dt)).min and dmax <= np.iinfo(np.dtype(dt)).max):
                return data.astype(np.dtype(dt))
    # #########################################################
    # If general_dtype is 'float', return a warning as conversion to float is not handled
    elif general_dtype == "float":
        logger.warning("Conversion to float not possible!")
        return data
    # #########################################################
    # If no suitable conversion was performed, return the original data
    return data


def get_desired_cpu_gpu_object(
            adata: ad.AnnData
        ) -> ad.AnnData:
    """
    DEPRECATED: Return the CPU or GPU variant of the adata object based on config.

    NOTE:
        Deprecated. GPU path is currently inactive and ``cunnData`` is no longer
        used. Decision is based on ``adata.uns["config"]["general"]["use_GPU"]``.

    Args:
        adata (anndata.AnnData): Adata object with configuration stored in
            ``adata.uns["config"]``.

    Returns:
        anndata.AnnData:
            The original or converted object depending on ``use_GPU`` flag.

    Raises:
        NotImplementedError: If the input type is not supported or not
            adata-compatible.

    TODO:
        Re-enable GPU functionality using updated ``cunnData`` if available.
        Add support for MuData or explicitly reject with informative error.

    Tags:
        config, stats
    """
    # Returns a adata object for cpu/gpu based on
    # Get the config object of the adata
    config = adata.uns["config"]
    # #################
    # Check if stats dict is in uns and append it if not
    if "stats" not in adata.uns.keys():
        adata.uns["stats"] = {}
    # #################
    # Check datatype and return the desired one
    if isinstance(adata, ad.AnnData):
        if not config["general"]["use_GPU"]:
            return adata
        else:
            return adata
            # THE cunnData is deprecated and the function
            # cdata = cunnData(adata=adata)
            # return cdata
    elif "to_AnnData" in dir(adata):
        # if cudata is not installed we cannot check it here,
        # but we can check if the to_AnnData function is there.
        if not config["general"]["use_GPU"]:
            return adata.to_AnnData()
        else:
            return adata
    # elif isinstance(adata, mu.MuData):
    #   raise NotImplementedError("The input Format is MuData, which cannot be"
    #                             "handled right now, please consider using adata!")
    else:
        raise NotImplementedError("This data type is not supported, please consider using adata!")


# ###########################################################################################################
# analysis Functions and wrapers for scanpy
def get_all_layers(
            adata: ad.AnnData,
            layers: list[str] = ["counts", "norm_counts", "log2norm_counts", "scaled"]
        ) -> None:
    """Creates the specified layers within an adata object.

    This function generates various layers in the given adata object based on
    the input list ``layers``. The layers represent different forms of the data,
    such as raw counts, normalized counts, log-transformed normalized counts,
    and scaled data. This is particularly useful in single-cell RNA-seq analysis
    workflows where different forms of the data are required for different
    stages of the analysis pipeline.

    NOTE:
        This is only valid if raw counts are provided in the adata.X!
        If you don't have the "norm_count" in layers, itskip this step,
        so the log2norm_counts are actually not from normalized.

    Args:
        adata (anndata.AnnData): Adata object containing the adata (``adata.X``)
            to be transformed.
        layers (list[str], optional): A list of layers to create. The default
            layers are ["counts", "norm_counts", "log2norm_counts", "scaled"].
            Defaults to ["counts".

    Returns:
        None:
            The function modifies the input adata object in-place, adding the
            specified layers to ``adata.layers``.

    Raises:
        ValueError: If any of the specified layers cannot be created due to
            missing raw counts or other prerequisites.

    Tags:
        normalization, scaling
    """
    # #########################################################
    # Creating the 'counts' layer if specified in the layers list
    if "counts" in layers:
        adata.layers["counts"] = adata.X.copy()
        # Logging for debugging purposes could be added here (currently commented out)
        # logger.debug(adata.layers["counts"].sum(1).mean())
    # #########################################################
    # Creating the 'log2_counts' layer if specified in the layers list
    if "log2_counts" in layers:
        if "counts" in adata.layers:
            adata.layers["log2_counts"] = adata.layers["counts"].copy()
            adata.layers["log2_counts"].data = np.log1p(
                adata.layers["log2_counts"].data)
        elif hasattr(adata, "X"):
            adata.layers["log2_counts"] = adata.X.copy()
            adata.layers["log2_counts"] = np.log1p(adata.layers["log2_counts"])
        else:
            raise KeyError("Neither 'counts' layer nor 'X' matrix found in adata")
    # #########################################################
    # Creating the 'norm_counts' layer if specified in the layers list
    if "norm_counts" in layers:
        sc.pp.normalize_total(adata)  # Normalizes the adata, total expression per cell to 1e4
        # logger.debug(adata.X.sum(1).mean())
        adata.layers["norm_counts"] = adata.X.copy()
        # logger.debug(adata.layers["norm"].sum(1).mean())
    # #########################################################
    # Creating the 'log2norm_counts' layer if specified in the layers list
    if "log2norm_counts" in layers:
        sc.pp.log1p(adata)  # Logarithmize the adata with a pseudocount of 1
        # logger.debug(adata.X.sum(1).mean())
        # ##########################################
        # Remove the unused adata.uns["log1p"] entry to clean up the adata object
        if "log1p" in adata.uns.keys():
            del adata.uns["log1p"]
        # ##########################################
        adata.layers["log2norm_counts"] = adata.X.copy()
        # logger.debug(adata.layers["lognorm"].sum(1).mean())
    # #########################################################
    # Creating the 'scaled' layer if specified in the layers list
    if "scaled" in layers:
        sc.pp.scale(adata, zero_center=True, max_value=10)  # Scales data to unit variance and zero mean
        # logger.debug(adata.X.sum(1).mean())
        adata.layers["scaled"] = adata.X.copy()
        # logger.debug(adata.layers["scaled"].sum(1).mean())
    # #########################################################
    # Resetting adata.X to original counts for consistency
    adata.X = adata.layers["counts"].copy()


def get_n_unique(
            adata: ad.AnnData,
            key_added: str = "n_unique",
            calc_score: bool = False
        ) -> None:
    """
    Calculate the number of unique non-zero count values from a sparse AnnData
    matrix and optionally compute a unique score.

    NOTE:
        This function converts the counts to integers, so it should be run on
        raw count data. If the adata.X was a csc_matrix afterwards it is a
        preferable csr_matrix

    The unique score is under development and may not be interpretable at the
    moment.

    Args:
        adata (anndata.AnnData): Adata object.
        key_added (str, optional): Key under which the unique counts will be
            added to ``adata.obs`` and ``adata.var``. Default is "n_unique".
            Defaults to "n_unique".
        calc_score (bool, optional): Whether to calculate a score based on the
            rarity of unique values. Default is False. Defaults to False.

    Returns:
        None:
            The function modifies the ``adata`` object in place by adding unique
            counts to ``adata.obs`` and ``adata.var``. If ``calc_score`` is
            True, it also adds a score based on the unique values' rarity.

    Raises:
        None

    Called By:
        all_qc_preprocessing, check_scoring_validity

    TODO:
        - Implement a better scoring algorithm.
        - Change this function to be run in the filtering step.

    Tags:
        calculation, obs, sparse, var
    """
    # #########################################################
    # Check if adata.X is a CSC matrix, convert to CSR if necessary
    # This conversion ensures compatibility with downstream processing
    if isinstance(adata.X, csc_matrix):
        adata.X = adata.X.tocsr()
    # #########################################################
    # Calculate the number of unique values for each row (cell) in adata.X
    # Iterate over rows, extract data, and compute unique values
    uniques_per_row = []
    for row_index in range(adata.X.shape[0]):
        start = adata.X.indptr[row_index]
        end = adata.X.indptr[row_index + 1]
        data = adata.X.data[start:end]
        uniques_per_row.append(np.unique(data))
    # #########################################################
    # Calculate the number of unique values for each column (gene) in adata.X
    # Convert the matrix to CSC format and transpose for faster column processing
    uniques_per_col = []
    data = adata.X.tocsc().T
    for col_index in range(data.shape[0]):
        start = data.indptr[col_index]
        end = data.indptr[col_index + 1]
        uniques_per_col.append(np.unique(data.data[start:end].astype(int)))
    # #########################################################
    # Save the number of unique values to adata.obs and adata.var
    adata.obs[key_added] = [len(unique) for unique in uniques_per_row]
    adata.var[key_added] = [len(unique) for unique in uniques_per_col]

    if calc_score:
        # #########################################################
        # Calculate scores based on the rarity of unique values
        # Flatten the unique values to compute occurrence counts
        uniques_col = [unique for col in uniques_per_col for unique in col]
        uniques_row = [unique for row in uniques_per_row for unique in row]
        # #########################################################
        # Count the occurrences of each unique value
        col_counter = Counter(np.array(uniques_col, dtype=int)).most_common()
        row_counter = Counter(np.array(uniques_row, dtype=int)).most_common()
        # #########################################################
        # Get the counts for the most common element and subtract from each value to compute scores
        most_common_element_count_col = np.log(col_counter[0][1] - 1)
        most_common_element_count_row = np.log(col_counter[0][1] - 1)

        count_to_scores_col = {
            k: np.abs(np.log(v) - most_common_element_count_col) for k, v in col_counter}
        count_to_scores_row = {
            k: np.abs(np.log(v) - most_common_element_count_row) for k, v in row_counter}
        # #########################################################
        # Save the calculated scores to adata.obs and adata.var
        adata.obs[f"{key_added}_score"] = [
            sum([count_to_scores_row[unique] for unique in unique_row]) for unique_row in uniques_per_row]
        adata.var[f"{key_added}_score"] = [
            sum([count_to_scores_col[unique] for unique in unique_col]) for unique_col in uniques_per_col]

        # Normalize the scores by dividing by the number of unique values + 1 to avoid division by zero
        adata.obs[f"{key_added}_score"] = adata.obs[f"{key_added}_score"] / (adata.obs["n_unique"] + 1)
        adata.var[f"{key_added}_score"] = adata.var[f"{key_added}_score"] / (adata.var["n_unique"] + 1)


def mark_highly_variable_genes(
            adata: ad.AnnData,
            layer: str = "norm_counts"
        ) -> None:
    """
    Wrapper for ``scanpy.pp.highly_variable_genes`` with additional gene
    inclusion/exclusion based on config settings.

    This function adjusts the layer used for identifying highly variable genes
    based on the configuration stored in ``adata.uns["config"]``. Depending on the
    selected flavor of preprocessing (e.g., "seurat_v3", "ours"), the function
    may overwrite the specified layer to ensure compatibility. If the flavor is
    "ours", it uses a unique method for identifying highly variable genes.

    NOTE:
        The function will modify the
        ``adata.uns["config"]["pp"]["highly_variable_genes"]["layer"]`` field
        based on the availability of layers and the flavor specified. If the
        necessary layers are not available, it will log warnings and potentially
        raise errors depending on the preprocessing flavor.

    Args:
        adata (anndata.AnnData): Adata object. The function assumes this object
            contains a configuration under ``adata.uns["config"]`` that guides the
            preprocessing steps.
        layer (str, optional): The default layer to use for identifying highly
            variable genes. Defaults to "norm_counts". Defaults to
            "norm_counts".

    Returns:
        None:
            The function modifies the ``adata`` object in place, updating the
            layer information and identifying highly variable genes accordingly.

    Raises:
        NotImplementedError: If a required layer (e.g., "counts", "norm_counts")
            is not provided when using specific flavors like "seurat_v3".
        AttributeError: If the required layer for the "ours" flavor (e.g.,
            "log2norm_counts") is not found in the data layers.

    Calls:
        calc_highly_variable_genes_unique_based

    Called By:
        Run_all_prep_steps_clustering, get_highly_variable

    TODO:
        Think about copying the config dictionary (not all but the
        ``.uns["config"]["pp"]["highly_variable_genes"]``)
        to prevent unintended modifications. However, keep in mind that this
        will then not update the original dictionary in case of further changes.

    Tags:
        calculation, config, normalization, stats, var
    """
    # ################################################################
    # Retrieve and update the configuration for highly variable genes
    config = adata.uns["config"]
    # TODO: Consider copying the relevant config dictionary for safety
    layer = config["pp"]["highly_variable_genes"]["layer"]
    layer_overwrite = None
    # #######################################################
    # Check the preprocessing flavor and adjust the layer accordingly
    if (
        config["pp"]["highly_variable_genes"]["flavor"] == "seurat_v3"
        and layer not in ["norm_counts", "counts"]
    ):
        # ##########################################
        # Handle Seurat V3 requirements for normalized layers
        logger.warning('Overwriting layer to "norm_counts" because seurat_v3 requires that!')
        if "norm_counts" in adata.layers.keys():
            layer_overwrite = "norm_counts"
        elif "counts" in adata.layers.keys():
            layer_overwrite = "counts"
        else:
            logger.warning("Using the X with seurat_v3 at your own risk!")
            # raise NotImplementedError("You must provide a counts or norm_counts layer to run seurat_v3")
    elif config["pp"]["highly_variable_genes"]["flavor"] == "ours":
        # ##########################################
        # Ensure required layers are present for "ours" flavor
        if "log2norm_counts" not in adata.layers.keys():
            raise AttributeError('The adata object lacks a "log2norm_counts" layer, preventing calculation '
                                 'of highly variable genes. Please use the default preprocessing first!')
        layer_overwrite = "log2norm_counts"
    else:
        # ##########################################
        # Use specified layer if available, otherwise set to None
        if layer in adata.layers.keys():
            layer_overwrite = layer
        else:
            layer_overwrite = None
    # #######################################################
    # Update the layer in the configuration based on the overwrite
    adata.uns["config"]["pp"]["highly_variable_genes"]["layer"] = layer_overwrite
    # #######################################################
    # Determine the method for calculating highly variable genes
    if config["pp"]["highly_variable_genes"]["flavor"] == "ours":
        # ##########################################
        # Use a custom method for identifying highly variable genes
        calc_highly_variable_genes_unique_based(adata, inplace=True, key="highly_variable")
    else:
        # ##########################################
        # Perform a sanity check on the number of genes and adjust as needed
        if adata.shape[1] < config["pp"]["highly_variable_genes"]["n_top_genes"]:
            logger.warning("Fewer genes available than aimed for highly variable calculation: "
                           f'{adata.shape[1]} < {config["pp"]["highly_variable_genes"]["n_top_genes"]}')
        sc.pp.highly_variable_genes(adata, **config["pp"]["highly_variable_genes"])
    # #######################################################
    # Optionally include marker genes in the highly variable genes list
    if "stats" not in adata.uns.keys():
        adata.uns["stats"] = {}
    adata.uns["stats"]["highly_variable_genes"] = []
    for m in config["marker_genes_to_include_in_highly_variables"]:
        # TODO: Consider adding only the markers
        adata.uns["stats"]["highly_variable_genes"].append([m, adata.var.highly_variable.at[m]])
        adata.var.highly_variable.at[m] = True
    # #######################################################
    # Clean up by removing outdated "hvg" data if present
    if "hvg" in adata.uns.keys():
        del adata.uns["hvg"]
    # #######################################################
    # NOTE: CPU implementation is not provided in this version
    # Uncomment and implement if necessary
    # raise NotImplementedError("Highly variable gene marking not implemented for CPU yet!")
    # sc.pp.highly_variable_genes(adata, n_top_genes=config["pp"]["highly_variable_genes"]["n_top_genes"],
    #                           flavor=config["pp"]["highly_variable_genes"]["flavor"],
    #                           span=1., layer=layer_overwrite)


def rename_obs_name_suffix(
            adata: ad.AnnData,
            obs_names: list[str],
            suffix: str,
            inplace: bool = True
        ) -> ad.AnnData | None:
    """
    Rename the specified observation names in an adata object by appending
    a suffix to each.

    This function modifies the ``.obs`` attribute of an adata object, renaming the
    specified observation names by appending a given suffix to each name. The
    function is useful when multiple datasets are being concatenated or analyzed
    together, and it is necessary to avoid naming conflicts or simply to add
    context to certain observation names.

    NOTE:
        - This function can modify the input ``adata`` object in place, depending
          on the ``inplace`` parameter.
        - Ensure that the suffix is not already present in the ``obs_names``
          to avoid duplicate suffixes.

    Args:
        adata (anndata.AnnData): Adata object.
        obs_names (list[str]): A list of observation names (column names in
            ``.obs``) to be renamed.
        suffix (str): The suffix to append to each of the specified observation
            names.
        inplace (bool, optional): If True, modifies the ``adata`` object in place
            and returns None. If False, returns a copy of the ``adata`` object
            with the modifications. Default is True. Defaults to True.

    Returns:
        anndata.AnnData | None:
            If ``inplace`` is False, returns a copy of the adata object with
            modified observation names. If ``inplace`` is True, returns None.

    Raises:
        KeyError: If any of the specified ``obs_names`` do not exist in
            ``adata.obs``.

    Tags:
        obs, utils
    """
    # #########################################################
    # Validate that the specified observation names exist in the ``.obs`` attribute
    missing_obs = np.setdiff1d(obs_names, adata.obs_names)
    obs_names = np.intersect1d(adata.obs_names, obs_names)

    if missing_obs.size > 0:
        logger.warning(f"The following observation names are missing from the adata object: {missing_obs.tolist()}")

    if obs_names.size == 0:
        raise KeyError("None of the provided observation names are found in the adata object.")
    # #########################################################
    # Rename the specified observation names by appending the provided suffix
    new_obs = adata.obs.rename(columns={k: k + suffix for k in obs_names})
    # #########################################################
    # Return the modified object or update in place
    if inplace:
        adata.obs = new_obs
        return None
    else:
        # Return a copy of the adata object with modified observation names
        adata_copy = adata.copy()
        adata_copy.obs = new_obs
        return adata_copy


def replace_small_counts(
            df: pd.DataFrame,
            column_name: str,
            inplace: bool = True,
            modified_column_name: str = None
        ) -> pd.DataFrame:
    """
    Replace values in a specified DataFrame column based on their
    frequency count. For string or categorical columns, replace with
    ``too_low_count``, and for integer columns, replace with a specific integer
    value (-1 or ``max_value + 1``) if their counts are lower than or equal to
    the number of unique values in the column.

    This function is designed to handle string, integer, and categorical
    columns. It can modify the DataFrame in place or return a modified column
    depending on the ``inplace`` parameter.

    NOTE:
        This function assumes that the DataFrame's column contains no missing
        values. It also assumes that the column is either of string, integer, or
        categorical type. This is a helper for ``plot_embedding_density``.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be modified.
        inplace (bool, optional): If True, modify the DataFrame in place and add
            a new column. If False, return the modified column only. Defaults to
            True.
        modified_column_name (str, optional): Name for the modified column.
            If None, Defaults to None.

    Returns:
        pandas.DataFrame:
            If ``inplace`` is True, returns the DataFrame with the
            modified column added. If ``inplace`` is False, returns the modified
            column as a pandas Series.

    Raises:
        TypeError: If the column is not of type string, integer, or categorical.

    Tags:
        utils
    """
    # #########################################################
    # Calculate the frequency of each value in the specified column
    value_counts = df[column_name].value_counts()
    # Sort the values by their frequency in ascending order
    sorted_counts = value_counts.sort_values()
    # Create a copy of the column to apply modifications
    modified_column = df[column_name].copy()
    # #########################################################
    # Determine the appropriate replacement value based on the column type
    # Check if the column is of string or categorical type
    if pd.api.types.is_string_dtype(modified_column) or isinstance(modified_column.dtype, pd.CategoricalDtype):
        replacement_value = 'too_low_count'
    # ##########################################
    # Check if the column is of integer type
    elif pd.api.types.is_integer_dtype(modified_column):
        if (modified_column >= 0).all():
            replacement_value = -1
        else:
            replacement_value = modified_column.max() + 1
    # Raise a TypeError if the column is of an unsupported type
    else:
        raise TypeError("Column must be of type string, integer, or categorical.")
    # #########################################################
    # Handle specific cases for categorical columns
    # Add the replacement value to categorical columns if it is not already a category
    if isinstance(modified_column.dtype, pd.CategoricalDtype):
        if replacement_value not in modified_column.cat.categories:
            modified_column = modified_column.cat.add_categories([replacement_value])
    # #########################################################
    # Determine the threshold for small counts and identify values to replace
    # Set the threshold as the number of unique values in the column
    threshold = len(sorted_counts)
    # Initialize variables for tracking cumulative counts and values to replace
    cumulative_count = 0
    to_replace = []
    # Iterate through the sorted values and identify those to replace
    for value, count in sorted_counts.items():
        if cumulative_count >= threshold:
            break
        to_replace.append(value)
        cumulative_count += count
    # Replace identified values in the copied column
    modified_column.loc[modified_column.isin(to_replace)] = replacement_value
    # #########################################################
    # Modify the original DataFrame or return the modified column based on the ``inplace`` parameter
    # Modify the DataFrame in place if specified
    if inplace:
        if modified_column_name is None:
            modified_column_name = 'Modified_' + column_name
        df[modified_column_name] = modified_column
        return df
    # Return only the modified column if not modifying in place
    else:
        return modified_column


def calc_top_n_ratio(adata: ad.AnnData, n_highest: int = 3) -> np.ndarray:
    """Calculate per-cell ratio of top-n counts vs. the rest.

    Args:
        adata (anndata.AnnData): adata object with sparse X (CSR) raw count.
        n_highest (int): Number of top counts to sum. Defaults to 3.

    Returns:
        np.ndarray: Array of ratios, shape (n_cells,).
    """
    if not isinstance(adata.X, csr_matrix):
        raise TypeError("adata.X must be a CSR sparse matrix.")

    X: csr_matrix = adata.X  # type: ignore

    return _compute_ratio_csr(X.data, X.indptr, n_highest)


# ###########################################################################################################
# adata/scanpy modification functions
def adata_hstack(
            rna_ad: ad.AnnData,
            prot_ad: ad.AnnData
        ) -> ad.AnnData:
    """
    DONT USE, was for old buggy adata versions
    Horizontally stacks RNA and protein adata objects while ensuring consistency
    in observation names and managing data from both datasets.

    This function combines RNA and protein data by matching observation names,
    making variable names unique, and then joining the corresponding data. The
    function also handles raw data if present, manages layers, and merges uns
    and obsm annotations.

    NOTE:
        When you use this, you have to be sure that the obs names are sorted
        in the same manner.

    Args:
        rna_ad (anndata.AnnData): Adata object containing RNA data.
        prot_ad (anndata.AnnData): Adata object containing protein data.

    Returns:
        anndata.AnnData:
            A new adata object with RNA and protein data stacked horizontally.

    Raises:
        ValueError: If the observation names are not aligned between the RNA and
            protein data.

    Tags:
        integration
    """
    # #########################################################
    # Identify and filter common observation names between RNA and protein data
    usable_ids = np.intersect1d(prot_ad.obs_names, rna_ad.obs_names)
    prot = prot_ad[usable_ids].copy()
    rna = rna_ad[usable_ids].copy()

    # Ensure variable names are unique to avoid conflicts during merging
    prot.var_names_make_unique()
    rna.var_names_make_unique()

    # Join the RNA and protein data into a single DataFrame
    # TODO: AHHHH adata IS BUGGY for the join!!!
    new_df = rna.to_df().join(prot.to_df())
    # #########################################################
    # Create a new adata object and make variable names unique
    adata = ad.AnnData(new_df)
    adata.var_names_make_unique()
    # #########################################################
    # Handle raw data if both RNA and protein data have raw attributes
    if rna.raw and prot.raw:
        tmp_rna = rna.copy()
        tmp_rna = tmp_rna.raw
        tmp_prot = prot.copy()
        tmp_prot = tmp_prot.raw
        # TODO: AHHHH adata IS BUGGY for the join!!!
        raw_df = tmp_rna[usable_ids].to_df().join(tmp_prot[usable_ids].to_df())
        adata.raw = ad.AnnData(raw_df)
    # #########################################################
    # Copy observation annotations (obs) from RNA and protein data
    for o in rna.obs:
        adata.obs[o] = rna.obs[o]
    for o in prot.obs:
        adata.obs[f"{o}_prot"] = prot.obs[o]

    # Concatenate and merge variable annotations (var) from RNA and protein data
    for o in rna.var:
        if o in prot.var:
            adata.var[o] = np.concatenate((rna.var[o], prot.var[o]), axis=None)
    # #########################################################
    # Merge shared layers from RNA and protein data
    layers_to_use = [x for x in list(rna.layers.keys()) if x in list(prot.layers.keys())]
    for layer in layers_to_use:
        pass
        # Check if the data is sparse
        # if isinstance(rna.layers[layer], csr_matrix):
        #     layer1 = rna.layers[layer].toarray()
        # else:
        #     layer1 = rna.layers[layer]
        # if isinstance(prot.layers[layer], csr_matrix):
        #     layer2 = prot.layers[layer].toarray()
        # else:
        #     layer2 = prot.layers[layer]
        # TODO: HERE SOMEHOW IS CODE MISSING!!!! FIND IT AND ADD IT BELOW!!!!
        #       It should stack the layers properly...
        # adata.layers[layer] = csr


def adata_hstack_dict(
            adata_dict: dict[str, ad.AnnData]
        ) -> ad.AnnData:
    """
    Horizontally stacks multiple adata objects based on common observation names
    from a dictionary.

    This function takes a dictionary of adata objects, typically containing data
    from different modalities (e.g., RNA and protein), and combines them
    horizontally based on their shared observation names. It ensures that the
    variable names are unique across all adata objects and handles the
    integration of layers and annotations, including raw data if present.

    NOTE:
        Assumes obs_names are sorted identically in all adata objects. (BUG In
        the adata join!!!)

    Args:
        adata_dict (dict[str): A dictionary where the keys are strings
            representing data types (e.g., "rna", "prot") and the values are
            adata objects.

    Returns:
        anndata.AnnData:
            A new adata object with horizontally concatenated data
            from all adata objects in the input dictionary.

    Tags:
        integration, stats
    """
    # #########################################################
    # Identify and extract common cells across all adata objects
    common_obs_names = set(adata_dict[list(adata_dict.keys())[0]].obs_names)
    for key, adata in adata_dict.items():
        common_obs_names = common_obs_names.intersection(adata.obs_names)
    common_obs_names = sorted(list(common_obs_names))
    # #########################################################
    # Extract and copy relevant data for each adata object
    extracted_adatas = {key: adata[common_obs_names].copy() for key, adata in adata_dict.items()}
    # #########################################################
    # Ensure unique variable names in each adata object
    for adata in extracted_adatas.values():
        adata.var_names_make_unique()
    # #########################################################
    # Perform the join operation across the data frames of all adata objects
    # TODO: AHHHH PANDAS IS BUGGY for the join!!!
    combined_df = extracted_adatas[list(extracted_adatas.keys())[0]].to_df()
    for key, adata in extracted_adatas.items():
        if key == list(extracted_adatas.keys())[0]:
            continue
        combined_df = combined_df.join(adata.to_df(), rsuffix=f'_{key}')
    # #########################################################
    # Create a new adata object with the joined data
    adata_result = ad.AnnData(combined_df)
    adata_result.var_names_make_unique()
    # #########################################################
    # Handle the raw data if it exists in all adata objects
    if all(adata.raw for adata in extracted_adatas.values()):
        combined_raw_df = extracted_adatas[list(extracted_adatas.keys())[0]].raw.to_df()
        for key, adata in extracted_adatas.items():
            if key == list(extracted_adatas.keys())[0]:
                continue
            combined_raw_df = combined_raw_df.join(adata.raw.to_df(), rsuffix=f'_{key}')
        adata_result.raw = ad.AnnData(combined_raw_df)
    # #########################################################
    # Transfer observation annotations from each adata object to the new adata object
    for key, adata in extracted_adatas.items():
        for o in adata.obs:
            adata_result.obs[f"{o}_{key}"] = adata.obs[o]
    # #########################################################
    # Concatenate variable annotations across all adata objects
    for key, adata in extracted_adatas.items():
        for o in adata.var:
            if o in adata_result.var:
                adata_result.var[o] = np.concatenate((adata_result.var[o], adata.var[o]), axis=None)
    # #########################################################
    # Handle the layers in all adata objects if they have common layers
    common_layers = set(extracted_adatas[list(extracted_adatas.keys())[0]].layers.keys())
    for key, adata in extracted_adatas.items():
        common_layers = common_layers.intersection(adata.layers.keys())
    for layer in common_layers:
        combined_layer = extracted_adatas[list(extracted_adatas.keys())[0]].layers[layer].toarray()
        for key, adata in extracted_adatas.items():
            if key == list(extracted_adatas.keys())[0]:
                continue
            combined_layer = np.concatenate((combined_layer, adata.layers[layer].toarray()), axis=1)
        adata_result.layers[layer] = csr_matrix(combined_layer)
    # #########################################################
    # Incorporate the 'stats' data from each adata object if present
    stats_combined = {}
    for key, adata in extracted_adatas.items():
        if "stats" in adata.uns.keys():
            stats_combined[f"{key}_stats"] = adata.uns["stats"]
    if stats_combined:
        adata_result.uns["stats"] = stats_combined
    # #########################################################
    # Transfer obsm data (multi-dimensional annotations) from each adata object
    for key, adata in extracted_adatas.items():
        for k, v in adata.obsm.items():
            adata_result.obsm[f"{k}_{key}"] = v
    # #########################################################
    # Return the new horizontally concatenated adata object
    return adata_result


def adata_vstack(
            rna_ad: ad.AnnData,
            tcr_ad: ad.AnnData
        ) -> ad.AnnData:
    """Vertically stacks two adata objects with common variable names.

    This function takes two adata objects (``rna_ad`` and ``tcr_ad``) and stacks
    them along the obs axis (rows) based on the common var names between the two
    datasets. The resulting adata object contains concatenated and handles cases
    where metadata may be missing from either dataset.

    NOTE:
        - This function assumes that the variable names (``var_names``) in both
          input adata objects are
          sorted in the same manner. If they are not, the behavior of the
          function may be incorrect.
        - The function will modify the ``var_names`` of the input adata objects to
          ensure they are unique
          within each dataset.
        - If both input adata objects contain raw data, the function will merge
          the raw data as well.
        - The function also attempts to concatenate matching layers between the
          two datasets.

    Args:
        rna_ad (anndata.AnnData): Adata object containing RNA data.
        tcr_ad (anndata.AnnData): Adata object containing TCR data.

    Returns:
        anndata.AnnData:
            A new adata object containing the vertically stacked
            data with merged obs and var.

    TODO:
          Implement the concatenation of layers between the two adata objects,
          ensuring compatibility with sparse and dense data formats.

    Tags:
        integration
    """
    # #########################################################
    # Identify common variable names and ensure uniqueness in var_names
    usable_genes = np.intersect1d(rna_ad.var_names, tcr_ad.var_names)
    rna_ad.var_names_make_unique()
    tcr_ad.var_names_make_unique()
    # #########################################################
    # Subset and concatenate the data for common genes
    prot = tcr_ad[:, usable_genes].copy()
    rna = rna_ad[:, usable_genes].copy()
    new_df = pd.concat([rna.to_df(), prot.to_df()])
    adata = ad.AnnData(new_df)
    adata.var_names_make_unique()
    # #########################################################
    # Merge variable metadata from the RNA and TCR datasets
    for o in rna.var.keys():
        adata.var[o] = rna.var[o]
    # #########################################################
    # Concatenate the observation metadata, handling missing keys
    rna_obs_shape = rna.shape[0]
    prot_obs_shape = prot.shape[0]
    for o in np.unique([*rna.obs.keys(), *prot.obs.keys()]):
        if o in prot.obs.keys():
            if o in rna.obs.keys():
                adata.obs[o] = np.concatenate((rna.obs[o], prot.obs[o]), axis=None)
            else:
                adata.obs[o] = np.concatenate((np.array(['None'] * rna_obs_shape), prot.obs[o]), axis=None)
        else:
            adata.obs[o] = np.concatenate((rna.obs[o], np.array(['None'] * prot_obs_shape)), axis=None)
    # #########################################################
    # Merge raw data if present in both RNA and TCR datasets
    if rna_ad.raw and tcr_ad.raw:
        raw_df = rna.raw[:, usable_genes].to_df().join(prot.raw[:, usable_genes].to_df())
        adata.raw = ad.AnnData(raw_df)
    # #########################################################
    # Concatenate layers between the two datasets
    layers_to_use = [x for x in list(rna.layers.keys()) if x in list(prot.layers.keys())]
    for layer in layers_to_use:
        # ##########################################
        # Convert sparse layers to dense arrays if necessary
        if isinstance(rna.layers[layer], csr_matrix):
            layer1 = rna.layers[layer].toarray()
        else:
            layer1 = rna.layers[layer]
        if isinstance(prot.layers[layer], csr_matrix):
            layer2 = prot.layers[layer].toarray()
        else:
            layer2 = prot.layers[layer]
        # ##########################################
        # Stack the layers vertically and convert back to sparse format
        adata.layers[layer] = csr_matrix(np.vstack((layer1, layer2), axis=1))
    # #########################################################
    # Merge any additional unstructured data
    if "stats" in prot.uns.keys():
        adata.uns["prot_stats"] = prot.uns["stats"]
    if "stats" in rna.uns.keys():
        adata.uns["genx_stats"] = rna.uns["stats"]
    # #########################################################
    # Return the merged adata object
    return adata


def adata_vstack_dict(
            adata_dict: dict[str, ad.AnnData]
        ) -> ad.AnnData:
    """
    Vertically stacks adata objects from a dictionary with common variable names.

    This function takes a dictionary of adata objects and stacks them along the
    observation axis (rows) based on the common variable names across all
    datasets. The resulting adata object contains concatenated obs and handles
    cases where metadata may be missing from some datasets.

    NOTE:
        - This function assumes that the variable names (``var_names``) in all
          input adata objects are sorted in the same manner. If they are not,
          the behavior of the function may be incorrect.
        - The function will modify the ``var_names`` of the input adata objects to
          ensure they are unique within each dataset.
        - If all input adata objects contain raw data, the function will merge
          the raw data as well.
        - The function also attempts to concatenate matching layers between the
          datasets.

    Args:
        adata_dict (dict[str): Mapping from modality or dataset name to adata.

    Returns:
        anndata.AnnData:
            A new adata object containing the vertically stacked
            data with merged obs and var.

    TODO:
        Implement the concatenation of layers between the adata objects,
        ensuring compatibility with sparse and dense data formats.

    Tags:
        integration, stats
    """
    # #########################################################
    # Identify common variable names across all datasets and ensure unique var_names
    all_var_names = [ad.var_names for ad in adata_dict.values()]
    usable_genes = np.intersect1d(*all_var_names)
    for this_adata in adata_dict.values():
        this_adata.var_names_make_unique()
    # #########################################################
    # Subset and concatenate the data for common genes
    sub_adatas = [ad[:, usable_genes].copy() for ad in adata_dict.values()]
    new_df = pd.concat([sub_ad.to_df() for sub_ad in sub_adatas])
    adata = ad.AnnData(new_df)
    adata.var_names_make_unique()
    # #########################################################
    # Merge variable metadata from all datasets
    reference_ad = next(iter(adata_dict.values()))  # Use the first adata object as a reference
    for var_key in reference_ad.var.keys():
        adata.var[var_key] = reference_ad.var[var_key]
    # #########################################################
    # Concatenate the observation metadata, handling missing keys
    obs_shapes = [ad.shape[0] for ad in sub_adatas]
    all_obs_keys = np.unique([key for ad in sub_adatas for key in ad.obs.keys()])

    for obs_key in all_obs_keys:
        concatenated_obs = []
        for this_adata, obs_shape in zip(sub_adatas, obs_shapes):
            if obs_key in this_adata.obs.keys():
                concatenated_obs.append(this_adata.obs[obs_key])
            else:
                concatenated_obs.append(np.array(['None'] * obs_shape))
        adata.obs[obs_key] = np.concatenate(concatenated_obs, axis=None)
    # #########################################################
    # Merge raw data if present in all datasets
    if all(ad.raw is not None for ad in adata_dict.values()):
        raw_dfs = [ad.raw[:, usable_genes].to_df() for ad in sub_adatas]
        raw_df = pd.concat(raw_dfs)
        adata.raw = ad.AnnData(raw_df)
    # #########################################################
    # Concatenate layers between the datasets
    common_layers = set.intersection(*[set(ad.layers.keys()) for ad in adata_dict.values()])
    for layer in common_layers:
        # ##########################################
        # Convert sparse layers to dense arrays if necessary
        concatenated_layers = []
        for this_adata in sub_adatas:
            layer_data = this_adata.layers[layer]
            if isinstance(layer_data, csr_matrix):
                layer_data = layer_data.toarray()
            concatenated_layers.append(layer_data)
        # ##########################################
        # Stack the layers vertically and convert back to sparse format
        adata.layers[layer] = csr_matrix(np.vstack(concatenated_layers, axis=1))
    # #########################################################
    # Merge any additional unstructured data
    for adata_key, this_adata in adata_dict.items():
        if "stats" in this_adata.uns.keys():
            adata.uns[f"{adata_key}_stats"] = this_adata.uns["stats"]
    # #########################################################
    # Return the merged adata object
    return adata


def add_layer_to_adata(
            adata: ad.AnnData,
            bdata: ad.AnnData,
            from_layer: str = "X",
            to_layer: str | None = None
        ) -> ad.AnnData:
    """
    Add a specified layer from one adata object to another, potentially overwriting
    existing data.

    This function transfers a layer from ``bdata`` to ``adata``, with flexibility in
    specifying which layer to copy and where to place it. If you don't want to
    carry all the layers in your adata object, you can delete them at any time
    and reload the original adata object, then load, for
    example, the "count" layer again with:
    >>> bdata = sc.read_h5ad("raw_or_with_all_layers.h5ad")
    >>> add_layer_to_adata(adata, bdata, from_layer="X", to_layer="count")

    NOTE:
        from_layer and to_layer == "X" refers to the adata.X and not a layer.

    Args:
        adata (anndata.AnnData): Adata object to add the layer to.
        bdata (anndata.AnnData): Adata object with the layer to copy.
        from_layer (str, optional): "X" for bdata.X or a layer in
            bdata.layers.keys(). Defaults to "X".
        to_layer (str | None, optional): "X" for adata.X or a layer in
            adata.layers.keys(). Defaults to None.

    Returns:
        anndata.AnnData:
            The updated adata object with the new layer added or replaced.

    Raises:
        ValueError: If the ``adata`` object is larger in dimensions than the
            ``bdata`` object.
        ValueError: If ``obs_names`` or ``var_names`` between ``adata`` and ``bdata`` do
            not overlap completely.

    Tags:
        utils
    """
    # #########################################################
    # Validate the dimensions of adata and bdata to ensure compatibility
    if adata.shape[0] > bdata.shape[0] or adata.shape[1] > bdata.shape[1]:
        logger.warning("The object to add the layer to is larger, this should not be the case, please check the order!")
        return
    # #########################################################
    # Check for overlapping observation and variable names between adata and bdata
    obs_overlap = np.intersect1d(adata.obs_names, bdata.obs_names)
    var_overlap = np.intersect1d(adata.var_names, bdata.var_names)

    if len(obs_overlap) < adata.shape[0] or len(var_overlap) < adata.shape[1]:
        logger.warning("The adata objects are not coming from the same data, obs and/or var-names are not overlapping")
        return
    # #########################################################
    # Determine target layer in adata; default to from_layer if to_layer is not specified
    if to_layer is None:
        to_layer = from_layer
    # #########################################################
    # Add or replace the layer in adata using the specified from_layer and to_layer
    if from_layer in bdata.layers.keys():
        if to_layer != "X":
            adata.layers[to_layer] = bdata[adata.obs_names, adata.var_names].layers[from_layer].copy()
        else:
            adata.X = bdata[adata.obs_names, adata.var_names].layers[from_layer].copy()
    elif from_layer == "X":
        if to_layer != "X":
            adata.layers[to_layer] = bdata[adata.obs_names, adata.var_names].X.copy()
        else:
            adata.X = bdata[adata.obs_names, adata.var_names].X.copy()


def set_obs_from_other_adata(
            adata_to_change: ad.AnnData,
            adata_with_data: ad.AnnData,
            obs_key: str,
            obsm_key: str = None,
            default_obs_value: str = "none",
            overwrite: bool = False,
        ) -> None:
    """
    Replace the observation key of ``adata_to_change`` with data from
    ``adata_with_data``, and optionally add/replace an obsm key.

    This function updates the specified observation key (``obs[obs_key]``) in
    ``adata_to_change`` with data from ``adata_with_data``. Optionally, if
    ``obsm_key`` is provided, it also updates ``obsm[obsm_key]``, initializing
    missing entries to np.nan.

    NOTE:
        If the obs_key or obsm_key is not in the

    Args:
        adata_to_change (anndata.AnnData): Adata object to modify.
        adata_with_data (anndata.AnnData): Adata object providing the data.
        obs_key (str): Key in ``.obs`` to update.
        obsm_key (str, optional): Key in ``.obsm`` to update. Defaults to None.
        default_obs_value (str, optional): Default value for non-overlapping obs
            entries. Defaults to "none".
        overwrite: If True, overwrites existing values in the obs/obsm.
            Defaults to False.

    Returns:
        None

    Raises:
        ValueError:
            If ``obs_key`` is not in ``adata_with_data.obs``.
        ValueError:
            If ``obsm_key`` is provided but not found in
            ``adata_with_data.obsm``.

    Tags:
        obs, utils
    """
    # ########################################
    # Check keys exist
    if obs_key not in adata_with_data.obs.columns:
        raise ValueError(f"obs_key '{obs_key}' not found in adata_with_data.obs.")
    if obsm_key is not None and obsm_key not in adata_with_data.obsm.keys():
        raise ValueError(f"obsm_key '{obsm_key}' not found in adata_with_data.obsm.")
    # ########################################
    # Add the default key, if not there
    if obs_key not in adata_to_change.obs.columns or overwrite:
        adata_to_change.obs[obs_key] = default_obs_value
    if obsm_key is not None and obsm_key not in adata_to_change.obsm.keys() or overwrite:
        del adata_to_change.obsm[obsm_key]
    # ########################################
    # Find overlapping barcodes
    overlapping_barcodes = [b for b in adata_to_change.obs_names if b in adata_with_data.obs_names]
    # ########################################
    # Set default value for obs_key
    # Ensure both are categorical with same category set
    if (
            pd.api.types.is_categorical_dtype(adata_to_change.obs[obs_key])
            and pd.api.types.is_categorical_dtype(adata_with_data.obs[obs_key])):
        all_cats = (
            adata_to_change.obs[obs_key].cat.categories
            .union(adata_with_data.obs[obs_key].cat.categories))
        adata_to_change.obs[obs_key] = (
            adata_to_change.obs[obs_key].astype(pd.CategoricalDtype(categories=all_cats)))
        adata_with_data.obs[obs_key] = (
            adata_with_data.obs[obs_key].astype(pd.CategoricalDtype(categories=all_cats)))
    # Update overlapping
    adata_to_change.obs.loc[overlapping_barcodes, obs_key] = (
        adata_with_data.obs.loc[overlapping_barcodes, obs_key])

    if obsm_key is not None:
        # Prepare shape and dtype for obsm update
        obsm_shape = adata_with_data.obsm[obsm_key].shape[1]
        new_obsm_array = np.full((adata_to_change.n_obs, obsm_shape), np.nan, dtype=float)
        # Map rows for overlapping barcodes
        idx_to_change = [adata_to_change.obs_names.get_loc(b) for b in overlapping_barcodes]
        idx_with_data = [adata_with_data.obs_names.get_loc(b) for b in overlapping_barcodes]
        # Update values
        new_obsm_array[idx_to_change, :] = adata_with_data.obsm[obsm_key][idx_with_data, :]
        # Assign to obsm
        adata_to_change.obsm[obsm_key] = new_obsm_array


def get_minimal_adata(
            adata: ad.AnnData,
            how: list = ["all"],
            force_rpy2: bool = False,
            how_specific: dict = None,
            inplace: bool = False,
        ) -> ad.AnnData:
    """
    Minimizes the adata object by returning a minimal copy keeping only
    ``how`` (inplace=False), or stripping unwanted parts not in ``how`` (inplace=True).

    NOTE:
        To use ``how_specific``, the slot (e.g., "obs") must also be in ``how``.
        .. code-block:: python

            >>> get_minimal_adata(..., how=["obs"], how_specific={"obs": ["leiden"]})

    Args:
        adata (anndata.AnnData): Adata object.
        how (list, optional): List of keys to keep or modify. E.g., ["obs",
            "var", "obsm", "X", ...] or ["all"] or really all except for obs and
            var names ["all", "X"]. Defaults to ["all"].
        force_rpy2 (bool, optional): If True, always keep .obs even if listed in
            how. Defaults to False.
        how_specific (dict, optional): Optional per-field key retention, e.g.,
            {"obs": {"n_counts"}}. Defaults to None.
        inplace (bool, optional): If True, modifies adata directly. Otherwise,
            returns new object. Defaults to False.

    Returns:
        anndata.AnnData:
            The minimized object (same if inplace=True).

    Called By:
        get_adata_subset

    Tags:
        utils
    """
    fields = ["obs", "var", "obsm", "obsp", "varm", "varp", "uns", "layers"]

    # Handle "all" shortcut: full strip, keep only shape + names
    if "all" in how:
        if inplace:
            adata.obsm = None
            adata.obsp = None
            adata.varm = None
            adata.varp = None
            adata.uns = None
            adata.layers = None
            if not force_rpy2:
                adata.obs = adata.obs.iloc[:, 0:0]
            adata.var = adata.var.iloc[:, 0:0]
            return adata
        else:
            adata_new = ad.AnnData(X=adata.X.copy())
            adata_new.obs_names = adata.obs_names.copy()
            adata_new.var_names = adata.var_names.copy()
            if force_rpy2:
                adata_new.obs = adata.obs.copy()
            return adata_new

    # Force keep obs if requested
    if force_rpy2 and "obs" not in how:
        how.append("obs")

    # Choose target object
    if inplace:
        target = adata
    else:
        target = ad.AnnData(X=adata.X.copy() if "X" in how else csr_matrix(adata.shape, dtype=np.float32))
        target.obs_names = adata.obs_names.copy()
        target.var_names = adata.var_names.copy()
    # ##############
    # Fields to handle
    for attr in fields:
        obj = getattr(adata, attr)

        # Only keep if in how
        if attr in how:
            keep_keys = set(how_specific[attr]) if how_specific and attr in how_specific.keys() else None
            if keep_keys is not None:
                # DataFrames
                if attr in ["obs", "var"]:
                    if not inplace:
                        setattr(target, attr, obj.loc[:, list(keep_keys)].copy())
                    else:
                        drop_cols = [col for col in obj.columns if col not in keep_keys]
                        obj.drop(columns=drop_cols, inplace=True)
                # Dict-like
                else:
                    filtered = {k: obj[k] for k in obj.keys() if k in keep_keys}
                    if not inplace:
                        setattr(target, attr, filtered)
                    else:
                        for k in list(obj.keys()):
                            if k not in keep_keys:
                                del obj[k]
            else:
                if not inplace:
                    setattr(target, attr, obj.copy() if hasattr(obj, "copy") else obj)
        else:
            if inplace:
                if attr in ["obs", "var"]:
                    setattr(adata, attr, adata.obs.iloc[:, 0:0] if attr == "obs" else adata.var.iloc[:, 0:0])
                elif hasattr(obj, "clear"):
                    obj.clear()
                else:
                    setattr(adata, attr, None)
    # ##############
    # Special handling for X
    if "X" in how:
        if inplace:
            adata.X = csr_matrix(adata.shape, dtype=np.float32)
        else:
            target.X = csr_matrix(adata.shape, dtype=np.float32)

    return adata if inplace else target


def replace_raw_by_layer(
            adata: ad.AnnData,
            layer: str
        ) -> ad.AnnData:
    """Replace the ``.raw`` attribute of an adata object with a specified layer.

    This function takes an adata object (``adata``) and replaces its ``.raw``
    attribute with the data from a specified layer. This can be useful when you
    want to update or reassign the ``.raw`` data to another layer of the adata
    object.

    NOTE:
        Using the raw attribute is deprecated, so it is recommended not to use
        this functionality in new code or pipelines. This function is provided
        for backward compatibility or specific use cases where raw needs to be
        updated.

    Args:
        adata (anndata.AnnData): Adata object whose ``.raw`` attribute will be
            replaced.
        layer (str): The name of the layer to replace ``.raw`` with. This layer
            must exist within ``adata.layers``.

    Returns:
        anndata.AnnData:
            A new adata object with the updated ``.raw`` attribute.

    Raises:
        ValueError: If the specified ``layer`` does not exist in ``adata.layers``.

    TODO:
        Ensure that the deprecation warning is appropriately logged to notify
        users of this deprecated feature.

    Tags:
        utils
    """
    # #########################################################
    # Log a warning to indicate that the usage of ``.raw`` is deprecated
    logger.warning('Using .raw is deprecated, avoid using this in new code or pipelines.')
    # #########################################################
    # Check if the specified layer exists in the adata object's layers
    if layer not in adata.layers.keys():
        raise ValueError(f'{layer} is not in adata.layers!')
    # #########################################################
    # Create a copy of the adata object to avoid modifying the original
    adata = adata.copy()
    # #########################################################
    # Replace the adata.X with the specified layer
    adata.X = adata.layers[layer]
    # #########################################################
    # Set the .raw attribute to the updated adata object
    adata.raw = adata
    # #########################################################
    # Return the modified adata object
    return adata


def search_adata_keys(
            adata: ad.AnnData,
            regex: str,
            members_to_search: list[str] | None = None,
            print_string: str = "adata"
        ) -> str:
    """
    Searches for keys in specified adata object members that match a given
    regular expression.

    This function allows the user to search for keys within specific members of
    an adata object (``obs``, ``var``, ``obsm``, ``varm``, ``uns``, ``obsp``, ``varp``) that
    match a regular expression pattern. If no members are specified, it will
    search through a default list of commonly used adata members.

    NOTE:
        This function assumes that all members specified are either
        dictionaries or have a ``.keys()`` method that returns their keys. If this
        is not the case, an error may occur.

    Args:
        adata (anndata.AnnData): Adata object containing the members to be
            searched.
        regex (str): The regular expression pattern used to match keys.
            Default is an empty string, which matches all keys.
        members_to_search (list[str] | None, optional): List of member names to
            search within. Default is ``['obs', 'var', 'obsm', 'varm', 'uns',
            'obsp', 'varp']``. Defaults to None.
        print_string (str, optional): will replace the adata by this string, for
            easier copying. Defaults to "adata".

    Returns:
        str:
            A comma-separated string of all matching keys in the format
            ``adata.member_name["key"]``.

    Raises:
        AttributeError: If ``adata`` does not have one of the specified members.

    Calls:
        search_adata_keys.search_member

    TODO:
        Consider extending functionality to handle non-dict members that may
        have nested keys.

    Tags:
        utils
    """
    # #########################################################
    # Compile the provided regex pattern for searching
    search_pattern = compile(regex)
    results = []
    # #########################################################
    # Define the default members to search if none are provided
    if members_to_search is None:
        members_to_search = ['obs', 'var', 'obsm', 'varm', 'uns', 'obsp', 'varp']

    # SHIT this doesn't work out because the object is in the locals as adata and maybe only
    # globas as the original
    # for name, value in list(globals().items()):
    #     if isinstance(value, ad.AnnData):
    #         if value.__str__() == adata.__str__():
    #             print(value.__str__())
    #             print(adata.__str__())
    #             print(name)
    #             adata_name = copy(name)
    #             break
    # print(adata_name)
    # #########################################################
    # Helper function to search within a specified member
    def search_member(
                member_name,
                member
            ) -> None:
        """Helper"""
        # ##########################################
        # Iterate through each key in the member and check for a match
        for key in member.keys():
            if search_pattern.search(key):
                results.append(f'{print_string}.{member_name}["{key}"]')
        # ##########################################
    for name, value in list(globals().items()):
        if value is adata:
            print(name)
    # #########################################################
    # Check each member in the provided or default list
    for member_name in members_to_search:
        if hasattr(adata, member_name):
            member = getattr(adata, member_name)
            # ##########################################
            # Ensure the member is a dictionary or has a keys() method
            if isinstance(member, dict) or hasattr(member, 'keys'):
                search_member(member_name, member)
    # #########################################################
    # Return a comma-separated string of all matching keys
    return ", ".join(results)


def get_adata_sub_keys(
            adata: ad.AnnData,
            keys: list[str] | tuple[str, ...],
            obsm_keys: list[str] | tuple[str, ...] | None = None,
            varm_keys: list[str] | tuple[str, ...] | None = None,
            groupby: str | list[str] | tuple[str, ...] | None = None,
            var_keys_only: bool = False,
            use_all_possible_keys: bool = False,
            layer: str | None = None
        ) -> pd.DataFrame:
    """Extracts specified features from an adata object into a flat DataFrame.

    This function retrieves values for a list of keys from ``.obsm``, ``.var``,
    ``.varm``, ``.var_names``, or ``.obs`` of an adata object. It supports optional
    grouping columns and flexibility for extracting observation- or
    variable-level data.

    NOTE:
        - If ``var_keys_only`` is True, ``.obsm`` and ``.obs`` will be ignored and
          only ``.var`` and ``.varm`` will be searched.
        - If ``use_all_possible_keys`` is True, all matching sources are included with
          suffixes.

    Args:
        adata (anndata.AnnData): Adata object with ``.obs``, ``.var``, ``.obsm``, and
            ``.varm``.
        keys (list[str] | tuple[str): Feature names to extract from the adata
            object.
        obsm_keys (list[str] | tuple[str, optional): Keys to search under
            ``adata.obsm``. Only used if ``var_keys_only`` is False. Defaults to
            None.
        varm_keys (list[str] | tuple[str, optional): Keys to search under
            ``adata.varm``. Only used if ``var_keys_only`` is True. Defaults to
            None.
        groupby (str | list[str] | tuple[str, optional): Column(s) in
            ``adata.obs`` to add to the result for grouping purposes. Defaults to
            None.
        var_keys_only (bool, optional): If True, only extract from ``.var`` or
            ``.varm``, not from ``.obsm``, ``.obs``, or ``.var_names``. Defaults to
            False.
        use_all_possible_keys (bool, optional): If True, include all sources and
            append suffixes. Defaults to False.
        layer (str | None, optional): Use ``adata.layers[layer]`` instead of
            ``adata.X``. Defaults to None.

    Returns:
        pandas.DataFrame:
            A DataFrame with extracted values for each key and optional groupby
            columns.

    Raises:
        TypeError: If ``groupby``, ``obsm_keys``, or ``varm_keys`` are of invalid
            type.
        AttributeError: If any key is not found in the searched adata locations.

    Calls:
        handle_key_conflicts

    Called By:
        plot_dotplot, plot_heatmap, plot_split_dotplot, plot_split_dotplot_mpl,
        plot_umap, plot_umap_cat_splitting, plot_umap_sbs, plot_violin

    Tags:
        groupby, obs, utils, var
    """
    # #########################################################
    # Initialize the output structure and ensure key uniqueness
    data = {}  # pd.DataFrame(index=adata.obs_names)

    # Make sure the keys are unique, to not waste computation resources
    keys = list(set(keys))

    if var_keys_only and obsm_keys is not None:
        raise ValueError("Please only use obs OR var keys, not both.")

    if layer is not None and layer not in adata.layers.keys():
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    # #########################################################
    # Normalize the groupby input
    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
        elif not isinstance(groupby, (list, tuple)):
            raise TypeError("groupby must be a string or a list/tuple of strings")
        for g in groupby:
            if g not in keys:
                if isinstance(keys, np.ndarray):
                    keys = keys.tolist()
                keys.append(g)
    # #########################################################
    # Ensure obsm_keys and varm_keys are lists
    if isinstance(obsm_keys, str):
        obsm_keys = [obsm_keys]
    elif obsm_keys is not None and not isinstance(obsm_keys, (list, tuple)):
        raise TypeError("obsm_keys must be a string or a list/tuple of strings")

    if isinstance(varm_keys, str):
        varm_keys = [varm_keys]
    elif varm_keys is not None and not isinstance(varm_keys, (list, tuple)):
        raise TypeError("varm_keys must be a string or a list/tuple of strings")
    # #########################################################
    # Extract data depending on key mode
    if not var_keys_only:
        if not use_all_possible_keys:
            obs_priority_note = (
                "When resolving observation-level keys, the default priority order is: "
                "``obsm``, then ``obs``, then ``var_names``. Use ``use_all_possible_keys=True`` to include all.")
            handle_key_conflicts(
                adata,
                keys,
                obsm_keys=obsm_keys,
                extra_msg=obs_priority_note)
        for key in keys:
            found_locs = []
            # ##########################################
            # Check in obsm if keys provided
            if obsm_keys is not None:
                for obsm_key in obsm_keys:
                    if obsm_key in adata.obsm and isinstance(adata.obsm[obsm_key], pd.DataFrame):
                        if key in adata.obsm[obsm_key].columns:
                            found_locs.append("obsm")
                            if use_all_possible_keys:
                                # data[f"{key}_obsm"] = adata.obsm[obsm_key][key].to_numpy()  # not type save!
                                data[f"{key}_obsm"] = adata.obsm[obsm_key][key]
                            else:
                                # data[key] = adata.obsm[obsm_key][key].to_numpy()  # not type save!
                                data[key] = adata.obsm[obsm_key][key]
                                break
                if not use_all_possible_keys and "obsm" in found_locs:
                    continue
            # ##########################################
            # Check in obs if keys provided
            if key in adata.obs.columns:
                found_locs.append("obs")
                if use_all_possible_keys:
                    # data[f"{key}_obs"] = adata.obs[key].to_numpy()  # not type save!
                    data[f"{key}_obs"] = adata.obs[key]
                elif key not in data:
                    # data[key] = adata.obs[key].to_numpy()  # not type save!
                    data[key] = adata.obs[key]
            # ##########################################
            # Check in var_names if keys provided
            if key in adata.var_names:
                found_locs.append("var_names")
                expr_data = adata[:, key].layers[layer] if layer is not None else adata[:, key].X
                expr_data = expr_data.toarray().flatten() if hasattr(expr_data, "toarray") else expr_data.flatten()
                if use_all_possible_keys:
                    data[f"{key}_X"] = expr_data
                elif key not in data:
                    data[key] = expr_data

            if not use_all_possible_keys and key not in data:
                raise AttributeError(f"Key {key} not found in adata.obsm[{obsm_keys}], adata.obs, or adata.var_names.")

        return pd.DataFrame(data, index=adata.obs_names)

    else:
        if not use_all_possible_keys:
            var_priority_note = (
                "When resolving variable-level keys, the default priority order is: "
                "``var``, then ``obs_names``, then ``varm``. Use ``use_all_possible_keys=True`` to include all.")
            handle_key_conflicts(
                adata,
                keys,
                varm_keys=varm_keys,
                var_keys_only=True,
                extra_msg=var_priority_note)
        for key in keys:
            found_locs = []
            # ##########################################
            # Check in varm if keys provided
            if varm_keys is not None:
                for varm_key in varm_keys:
                    if varm_key in adata.varm:
                        varm_entry = adata.varm[varm_key]
                        if isinstance(varm_entry, pd.DataFrame) and key in varm_entry.columns:
                            found_locs.append("varm")
                            if use_all_possible_keys:
                                # data[f"{key}_varm"] = varm_entry[key].to_numpy()  # not type save!
                                data[f"{key}_varm"] = varm_entry[key]
                            elif key not in data:
                                # data[key] = varm_entry[key].to_numpy()  # not type save!
                                data[key] = varm_entry[key]
                            break
                        elif isinstance(varm_entry, np.ndarray):
                            raise TypeError(
                                f"Key {key} in adata.varm[{varm_key}] is an array, not DataFrame columns.")
            # ##########################################
            # Check in var if keys provided
            if key in adata.var.columns:
                found_locs.append("var")
                if use_all_possible_keys:
                    # data[f"{key}_var"] = adata.var[key].to_numpy()  # not type save!
                    data[f"{key}_var"] = adata.var[key]
                else:
                    # data[key] = adata.var[key].to_numpy()  # not type save!
                    data[key] = adata.var[key]
            # ##########################################
            # Check in obs_names if keys provided
            if key in adata.obs_names:
                found_locs.append("obs_names")
                expr_data = adata[key, :].layers[layer] if layer is not None else adata[key, :].X
                expr_data = expr_data.toarray().flatten() if hasattr(expr_data, "toarray") else expr_data.flatten()
                if use_all_possible_keys:
                    data[f"{key}_obs_names"] = expr_data
                elif key not in data:
                    data[key] = expr_data

            if not use_all_possible_keys and key not in data:
                raise AttributeError(
                    f'Key "{key}" not found in adata.var, adata.var_names, or adata.varm[{varm_keys}].')

        return pd.DataFrame(data, index=adata.var_names)


def subset_adata_random(
            adata: ad.AnnData,
            column: str | None = None,
            n: int | None = None,
            n_per_category: int | None = None,
            seed: int | None = None,
            prioritize_small_groups: int = 1,
            equal_sampling_per_class: bool = False
        ) -> ad.AnnData:
    """
    Subset an adata object randomly for each category in a categorical
    column.

    NOTE:
        You have to either set n (for a total number of cells) or
        n_per_category!

    Uses a numpy Generator for reproducible or uncontrolled randomness.

    Args:
        adata (anndata.AnnData): Adata object object to subset.
        column (str | None, optional): The categorical column to use for
            subsetting. If None, subset randomly. Defaults to None.
        n (int | None, optional): Total number of cells to subset (uniform
            across categories). Defaults to None.
        n_per_category (int | None, optional): Number of cells to subset per
            category. Defaults to None.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to None.
        prioritize_small_groups (int, optional): Number of the smallest classes
            to prioritize when filling to ``n``. Defaults to 1.
        equal_sampling_per_class (bool, optional): Whether to sample the same
            number of cells from each class. Defaults to False.

    Returns:
        anndata.AnnData:
            A subsetted adata object.

    Calls:
        get_random_generator

    Called By:
        create_random_spatial_adata

    Tags:
        groupby, obs, utils
    """
    if n_per_category is None and n is None:
        raise ValueError("Either ``n`` or ``n_per_category`` must be provided.")

    delete_column_afterwards = False
    if column is None:
        column = "BlablaBla"
        adata.obs[column] = True
        delete_column_afterwards = True

    rng = get_random_generator(adata, seed)

    sampled_indices = np.zeros(len(adata), dtype=bool)

    if equal_sampling_per_class:
        if n is None:
            raise ValueError("``n`` must be provided when ``equal_sampling_per_class`` is True.")

        categories = adata.obs[column].unique()
        num_categories = len(categories)

        n_per_category = n // num_categories
        extra_cells = n % num_categories

        sample_sizes = [n_per_category + 1 if i < extra_cells else n_per_category for i in range(num_categories)]

        for cat, sample_size in zip(categories, sample_sizes):
            cat_mask = adata.obs[column] == cat
            cat_indices = np.where(cat_mask)[0]
            if len(cat_indices) > sample_size:
                selected = rng.choice(cat_indices, sample_size, replace=False)
            else:
                selected = cat_indices

            sampled_indices[selected] = True

    elif n_per_category is not None:
        for cat in adata.obs[column].unique():
            cat_mask = adata.obs[column] == cat
            cat_indices = np.where(cat_mask)[0]
            if len(cat_indices) > n_per_category:
                selected = rng.choice(cat_indices, n_per_category, replace=False)
            else:
                selected = cat_indices

            sampled_indices[selected] = True

    elif n is not None:
        # Step 1: Calculate the sizes of the classes
        cat_sizes = adata.obs[column].value_counts()
        cat_sizes_normalized = cat_sizes * (n / cat_sizes.sum())
        # print(cat_sizes_normalized)

        rest_of_classes = adata.obs[column].unique().tolist()
        # print(rest_of_classes)
        # ##############################################
        if prioritize_small_groups > 0:
            # Step 2: Check which groups are smaller than prioritize_small_groups
            small_groups = cat_sizes_normalized[
                    cat_sizes_normalized < prioritize_small_groups  # +1 to make sure border cases have also
                    ].index.tolist()

            rest_of_classes = np.setdiff1d(rest_of_classes, small_groups).tolist()
            # ###########################
            # Iterate as long as there are small classes left
            classes_to_correct = ["_0_"]  # Dummy for the while
            while len(classes_to_correct) != 0:
                remaining_n = n - len(small_groups) * prioritize_small_groups
                rest_sizes = adata.obs[column].value_counts().loc[rest_of_classes]
                # Calculate proportional allocation
                rest_probs = rest_sizes / rest_sizes.sum()
                rest_allocations = (rest_probs * remaining_n).round().astype(int)
                classes_to_correct = (
                        rest_allocations[rest_allocations < prioritize_small_groups]
                        .index.to_numpy().tolist())  # TODO: Test i the to_numpy is required tolist should be ok
                classes_to_correct
                small_groups.extend(classes_to_correct)
                rest_of_classes = np.setdiff1d(rest_of_classes, classes_to_correct).tolist()
        # ##############################################
        # Step 3: Sample small classes with min(prioritize_small_groups, actual_size)
        sampled_indices = []
        for cat in small_groups:
            indices = np.where(adata.obs[column] == cat)[0]
            n_cat = min(prioritize_small_groups, len(indices))
            selected = rng.choice(indices, n_cat, replace=False)
            sampled_indices.extend(selected)

        already_sampled = len(sampled_indices)
        remaining_n = n - already_sampled
        if remaining_n < 0:
            raise ValueError(f"Too many required small-class samples ({already_sampled}) for n={n}")

        # Step 4: Sample proportionally from rest_of_classes
        if remaining_n > 0 and len(rest_of_classes) > 0:
            # Get actual counts for rest_of_classes
            rest_sizes = adata.obs[column].value_counts().loc[rest_of_classes]

            # Calculate proportional allocation
            rest_probs = rest_sizes / rest_sizes.sum()
            rest_allocations = (rest_probs * remaining_n).round().astype(int)

            # Adjust total in case rounding errors cause mismatch
            diff = remaining_n - rest_allocations.sum()
            if diff != 0:
                order = rest_allocations.sort_values(ascending=(diff < 0)).index
                for cat in order:
                    if diff == 0:
                        break
                    if diff > 0:
                        rest_allocations[cat] += 1
                        diff -= 1
                    elif rest_allocations[cat] > 0:
                        rest_allocations[cat] -= 1
                        diff += 1

            # Sample for each class
            for cat, n_cat in rest_allocations.items():
                indices = np.where(adata.obs[column] == cat)[0]
                if len(indices) > n_cat:
                    selected = rng.choice(indices, n_cat, replace=False)
                else:
                    selected = indices
                sampled_indices.extend(selected)

    if delete_column_afterwards:
        del adata.obs[column]

    return adata[sampled_indices, :]


def get_deg_df(
            adata: ad.AnnData,
            group: str | list[str] | None = None,
            deg_key: str = 'rank_genes_groups'
        ) -> pd.DataFrame:
    """
    Retrieves the ranked genes groups as a DataFrame from the provided
    adata object, either directly from the 'uns' attribute if already
    present or by generating it through the Scanpy method.

    Args:
        adata (anndata.AnnData): Adata object.
        group (str | list[str] | None, optional): Group or list of groups to
            filter. If None, all groups are considered. Default is None.
            Defaults to None.
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to 'rank_genes_groups'.

    Returns:
        pandas.DataFrame:
            DataFrame with ranked gene information, filtered by group
            if specified.

    Raises:
        KeyError: If ``deg_key`` is not found in ``adata.uns``.
        ValueError: If specified ``group`` is not found in the ranked gene groups.

    Called By:
        calc_DEGs, calc_DEGs.process_group_reference_pair, create_deg_table,
        get_DEG_gene_csvs, get_DEGs_per_group, plot_hallmark_group_heatmap

    TODO:
        Check the existence of deg_key before proceeding with
        adata.uns[deg_key] to avoid potential KeyErrors.
    """
    # #########################################################
    # Check if ranked genes groups are stored directly in ``adata.uns[deg_key]``
    if deg_key not in adata.uns:
        raise KeyError(f"The key '{deg_key}' was not found in ``adata.uns``.")
    # #########################################################
    # Prepare group parameter if specified
    if group is not None:
        # ##########################################
        # Ensure group is a list; wrap single group strings in a list
        if isinstance(group, str):
            group = [group]
    if isinstance(adata.uns[deg_key], pd.DataFrame):
        # ##########################################
        # If ``adata.uns[deg_key]`` is a DataFrame, copy and filter by specified groups if applicable
        deg_df = adata.uns[deg_key].copy()
    else:
        # ##########################################
        # Fallback: use Scanpy's function to retrieve the ranked genes groups DataFrame
        deg_df = sc.get.rank_genes_groups_df(adata, group=None, key=deg_key)
    # #########################################################
    # Check that each specified group exists in the 'group' column of the DataFrame
    if group is not None:
        # If group is not in the keys, the scampy code ran and we cannot subset anyway
        if "group" in deg_df.keys():
            missing_groups = [g for g in group if g not in deg_df["group"].unique()]
            if missing_groups:
                raise ValueError(f"The following groups are not found in the data: {missing_groups}")

            # Filter the DataFrame based on provided group(s)
            deg_df = deg_df[deg_df["group"].isin(group)].copy()
    # #########################################################
    # Return the final DataFrame with ranked gene information
    return deg_df


def get_adata_subset(
            adata: ad.AnnData,
            vs: tuple[str, str],
            ref: dict[str, list[str]],
            dict_keys: str | list[str] | None = None,
            cluster_key: str = "leiden",
            condition_obs_key: str = "condition",
            layer: str = "log2norm_counts",
            subset_to_groups: list = [],
            subset_to_degs: bool = False,
            deg_key: str | None = None,
            deg_df: pd.DataFrame | None = None,
            deg_obs_key: str | None = None,
            mask_non_significant_groups: bool = False,
            scale_axis: int = None,
            when_to_scale: str = "",
            kwarg_filter_genes: dict = {"perc": 0.1, "p_val_cutoff": 0.001, "lfc": 0.5},
            kwarg_filter_cells: dict = {"perc": 0.1, "p_val_cutoff": 0.001, "lfc": 0.0},
            minimal: dict | str = "default"
        ) -> tuple[ad.AnnData, list[str]]:
    """
    Processes an adata object by subsetting, filtering, and scaling
    based on differential expression analysis.

    NOTE:
        THIS IS A HELPER, and unfinished, just ignore it for now.

    Args:
        adata (anndata.AnnData): Adata object.
        vs (tuple[str], optional):
            Conditions to compare.
        ref (dict[str]):
            Marker specification used for gene subsetting. Internally passed to
            ``sc_utils.ref_to_list()``.

                - If dict: each key maps to a list of marker genes (e.g., pathway
                  or gene sets). The subset will be resolved using ``dict_keys``
                  to pick the relevant entries.
                - If list: used directly as the list of marker genes.
                - If str: treated as a single marker gene.

        dict_keys (str | list[str] | None, optional):
            Key(s) used to extract genes from the ``ref`` dictionary (if it is a
            dict). Ignored if ``ref`` is a list or string. Defaults to None.
        cluster_key (str, optional):
            Column in ``adata.obs`` that identifies groups/clusters.
            Defaults to "leiden".
        condition_obs_key (str, optional):
            Key in ``adata.obs`` specifying experimental conditions.
            Defaults to "condition".
        layer (str, optional):
            Layer to extract expression data from.
            Defaults to "log2norm_counts".
        subset_to_groups (list, optional):
            List of cluster names to subset cells to. Defaults to [].
        subset_to_degs (bool, optional):
            Whether to subset to differentially expressed genes.
            Defaults to False.
        deg_key (str | None, optional):
            Key in ``adata.uns`` containing differential expression results.
            Defaults to None which results in
            f"{condition_obs_key}_per_{cluster_key}_rank_genes_groups".
        deg_df (pandas.DataFrame | None, optional):
            Optionally precomputed DEG dataframe to reuse. If None, uses
            ``deg_key`` to extract. Defaults to None.
        deg_obs_key (str, optional):
            Column in ``adata.obs`` that maps conditions for DEG filtering.
            Defaults to None which results in
            ``cluster_key`` + "_" + ``condition_obs_key``
        mask_non_significant_groups (bool, optional):
            Whether to zero out genes not significantly expressed in specific
            groups/clusters. Defaults to False.
        scale_axis (int, optional):
            Axis along which to scale the data (0 = genes, 1 = cells).
            If None, no scaling is applied. Defaults to None.
        when_to_scale (str, optional):
            When to apply scaling ("before" or "after" filtering) if the.
            Defaults to "".
        kwarg_filter_genes (dict, optional):
            Keyword arguments passed to ``subset_degs`` for subsetting genes.
            Expected keys:

                - "perc": float, minimum % of cells
                - "p_val_cutoff": float, adjusted p-value threshold
                - "lfc": float, log fold-change threshold

            Defaults to {"perc": 0.5, "p_val_cutoff": 0.0001, "lfc": 0.5}.
        kwarg_filter_cells (dict, optional):
            Keyword arguments passed to ``subset_degs`` for zeroing
            non-significant genes by cluster. Same structure as
            ``kwarg_filter_genes``.
            Defaults to {"perc": 0.1, "p_val_cutoff": 0.0001, "lfc": 0.0}.
        minimal (dict | str, optional):
            Please only use, if you know what you are doing, read at least the
            doc of get_minimal_adata. If provided, the input ``adata`` is
            minimized using ``get_minimal_adata(**minimal)`` before any
            subsetting. The dictionary must contain keyword arguments for
            ``get_minimal_adata``, such as:

                - how: list of adata fields to retain (e.g.,
                  ["obs", "var", "layers", "uns"])
                - how_specific: nested keys to keep per field
                  (e.g., {"obs": [...], "layers": [...], "uns": [...]})
                - force_rpy2: whether to preserve .obs always
                - inplace: must be False (a copy is returned)

            If set to None (default), no minimization is performed.
            Defaults to "default".

    Returns:
        tuple[anndata.AnnData, list[str]]:
            The processed adata object and list of retained marker genes.

    Calls:
        ref_to_list, get_minimal_adata, inplace_max_scale_csr,
        set_sparse_subset_to_zero, subset_degs

    Tags:
        DEG, annotation, obs, scaling, var
    """
    # ########################################################################
    # Default keys
    if deg_obs_key is None:
        deg_obs_key = cluster_key + "_" + condition_obs_key
    if deg_key is None:
        deg_key = f"{condition_obs_key}_per_{cluster_key}_rank_genes_groups"
    # ########################################################################
    # Extract all ref (genes)
    this_ref = ref_to_list(ref, dict_keys)
    # ########################################################################
    # Prepare a most memory efficient preprocessing
    if minimal == "default":
        # Default is only the neccesary for plotting
        minimal = {
            "how": ["obs", "uns", "X"],
            "how_specific": {
                "obs": [],
                "uns": [],
            }}
        # Ensure .obs contains required columns
        obs_keys = [condition_obs_key, cluster_key, deg_obs_key]
        minimal.setdefault("how_specific", {})
        minimal["how_specific"].setdefault("obs", [])
        missing = np.setdiff1d(obs_keys, minimal["how_specific"]["obs"]).tolist()
        minimal["how_specific"]["obs"].extend(missing)
        minimal["how_specific"]["uns"].append(deg_key)

        adata_sub = get_minimal_adata(
            adata,
            **minimal)
    # Use provided one
    elif minimal is None:
        adata_sub = adata.copy()
    else:
        # Fix the necessary keys if not provided
        if "how_specific" in minimal.keys():
            if "obs" in minimal["how_specific"].keys():
                obs_keys = [condition_obs_key, cluster_key, deg_obs_key]
                missing = np.setdiff1d(obs_keys, minimal["how_specific"]["obs"]).tolist()
                minimal["how_specific"]["obs"].extend(missing)
            if "uns" in minimal["how_specific"].keys():
                if deg_key not in minimal["how_specific"]["uns"]:
                    minimal["how_specific"]["uns"].append(deg_key)
    # ############################################
    # initial checks
    if mask_non_significant_groups or subset_to_degs:
        if deg_df is None and deg_key not in adata_sub.uns.keys():
            raise ArithmeticError(f'{deg_key} is not in adata.uns')
        if deg_df is None:
            deg_df = adata_sub.uns[deg_key]
        deg_df = deg_df[
            deg_df["group"].str.contains(vs[0]) &
            deg_df["reference"].str.contains(vs[1])]
        # Determine all unique (group, reference) combinations in the DEG results
        combs = deg_df[["group", "reference"]].drop_duplicates()
        if len(combs) == 0:
            raise AttributeError(
                "vs is not matching the DEG group and reference, "
                "please make sure to use correct groups and DEGs")
    # ############################################
    # Get the desired layer
    if layer is not None and layer in adata.layers.keys():
        adata_sub.X = adata.layers[layer].copy()
    # ############################################
    # Subset the adata to the condition first (to also get a new object, to not change the old)
    adata_sub = adata_sub[adata_sub.obs[condition_obs_key].isin(vs)].copy()
    # ############################################
    # Subset to selected conditions
    # Scale data before subsetting if specified
    if scale_axis is not None and when_to_scale == "before":
        inplace_max_scale_csr(adata_sub.X, scale_axis)
    # ########################################################################
    # Subset cells to specific groups/clusters if enabled
    if len(subset_to_groups) != 0:
        if deg_obs_key not in adata_sub.obs.keys():
            raise AttributeError(f"{deg_obs_key} is not in adata.obs")
        # TOOD: Think of checking if all or some of the subset_to_groups are actually present
        adata_sub = adata_sub[adata_sub.obs[cluster_key].isin(subset_to_groups)].copy()
    # ############################################
    # 0 mask the non-significant genes if specified
    ref_genes = np.intersect1d(adata_sub.var_names.tolist(), this_ref).tolist()
    if mask_non_significant_groups:
        # ######################################################
        # Get the DEGs
        if deg_df is None:
            deg_df = adata_sub.uns[deg_key].copy()
        deg_df = subset_degs(deg_df, **kwarg_filter_cells)
        # filter to current genes
        deg_df = deg_df[deg_df["names"].isin(ref_genes)]
        # Subset to vs
        deg_df = deg_df[
            deg_df["group"].str.contains(vs[0]) &
            deg_df["reference"].str.contains(vs[1])]
        # Determine all unique (group, reference) combinations in the DEG results
        combs = deg_df[["group", "reference"]].drop_duplicates()

        # Iterate over the remaining groups
        for group, reference in combs.itertuples(index=False):
            # Get significant genes for this (group, reference) combination
            sig_genes = deg_df.loc[
                (deg_df["group"] == group) & (deg_df["reference"] == reference),
                "names"
            ].unique().tolist()  # The unique is not strictly necessary for proper DEG dataframes
            # Only consider marker genes not significant
            gene_mask = adata_sub.var_names.isin(ref_genes)
            genes_to_zero = gene_mask & ~adata_sub.var_names.isin(sig_genes)

            # Mask cells matching this cluster (group) and condition (reference)
            # cell_mask = (adata.obs[deg_obs_key] == group) & (adata.obs[condition_obs_key] == reference)
            cell_mask = adata_sub.obs[deg_obs_key].isin([group, reference])
            set_sparse_subset_to_zero(adata_sub.X, cell_mask, genes_to_zero)
    # ########################################################################
    # Subset genes based on differential expression
    if subset_to_degs:
        if deg_df is None:
            if deg_key not in adata_sub.uns.keys():
                raise ArithmeticError(f'{deg_key} is not in adata.uns')
            deg_df = adata_sub.uns[deg_key].copy()
        # subset to conditions
        deg_df = subset_degs(deg_df, **kwarg_filter_genes)
        deg_df = deg_df[
            deg_df["group"].str.contains(vs[0]) &
            deg_df["reference"].str.contains(vs[1])]

        this_ref = np.intersect1d(deg_df["names"].unique(), this_ref).tolist()
    else:
        adata_sub = adata_sub[:, this_ref].copy()
    # ############################################
    # Scale data after processing if specified
    if scale_axis is not None and when_to_scale == "after":
        inplace_max_scale_csr(adata_sub.X, scale_axis)

    return adata_sub, this_ref


def shift_qc_metrics(
            adata: ad.AnnData,
            operation: Literal["move", "copy", "del"] = "move",
            direction: Literal[
                "obs_to_obsm",
                "obsm_to_obs",
                "var_to_varm",
                "varm_to_var",
                "obs",
                "obsm",
                "var",
                "varm",
            ] = "obs_to_obsm",
            keys: list[str] | None = None,
            store_key: str = "qc_metrics",
            force_merge: bool = False,
            config: dict | None = None,
            qc_config_key: str = "keys_for_qc_all",
            var_keys_only: bool = False,
        ) -> ad.AnnData:
    """Shift, copy, or delete QC metrics between AnnData slots.

    This function handles all QC metric moves between obs/var and
    obsm/varm, including copy and delete modes.

    NOTE:
        - When ``keys=None``, they are resolved from
          ``config['general'][qc_config_key]['obs' or 'var']``.
        - When ``force_merge=False``, existing columns in obs/var are
          preserved. If the same key exists in both source and target,
          the column is skipped silently. To overwrite in such cases,
          set ``force_merge=True``.

    Args:
        adata (ad.AnnData):
            AnnData object to operate on.
        operation (Literal["move","copy","del"], optional):
            Type of action.

                - "move": transfer and remove from source.
                - "copy": transfer and keep source.
                - "del" : delete target data only.

            Defaults to "move".
        direction (Literal[..., optional]):
            Data axis and transfer direction.

                - For move/copy: one of
                  {"obs_to_obsm","obsm_to_obs","var_to_varm","varm_to_var"}.
                - For del: one of {"obs","obsm","var","varm"}.

            Defaults to "obs_to_obsm".
        keys (list[str] | None, optional):
            QC keys to operate on. If None, all columns in the relevant
            source are used. Defaults to None and qc_config_key will be used.
        store_key (str, optional):
            Name of slot in obsm/varm for compact storage.
            Defaults to "qc_metrics".
        force_merge (bool, optional):
            Whether to overwrite existing columns/slots.
            Defaults to False.
        config (dict | None, optional):
            Config dict. Defaults to ``adata.uns['config']`` if None.
        qc_config_key (str, optional):
            Key in config["general"] for QC keys. Defaults to "keys_for_qc_all".
        var_keys_only (bool, optional):
            Whether to only use var keys from config. Defaults to False.

    Returns:
        ad.AnnData: Modified AnnData object.

    Raises:
        ValueError:
            - If keys are missing for delete from obs/var.
            - If direction is invalid.
        KeyError:
            - If config does not contain qc_config_key.
            - If storing into obsm/varm where slot exists and
              ``force_merge=False``.
            - If restoring from obsm/varm where slot not found.
    """
    # ########################################################################
    # setup
    if config is None:
        config = adata.uns["config"]
    # ########################################################################
    # resolve keys from config if not provided
    if keys is None and operation != "del":
        if qc_config_key not in config["general"]:
            raise KeyError(
                f"qc_config_key '{qc_config_key}' not found in "
                "adata.uns['config']['general']")
        # #############################
        # obs or var keys from config
        keys = (
            config["general"][qc_config_key]["var"].copy()
            if var_keys_only
            else config["general"][qc_config_key]["obs"].copy())
    # ########################################################################
    # delete operation
    if operation == "del":
        # #############################
        # obs delete
        if direction == "obs":
            if keys is None:
                raise ValueError("Must specify keys for del in obs.")
            adata.obs.drop(columns=keys, inplace=True, errors="ignore")
        # #############################
        # var delete
        elif direction == "var":
            if keys is None:
                raise ValueError("Must specify keys for del in var.")
            adata.var.drop(columns=keys, inplace=True, errors="ignore")
        # #############################
        # obsm delete
        elif direction == "obsm":
            if store_key in adata.obsm:
                del adata.obsm[store_key]
        # #############################
        # varm delete
        elif direction == "varm":
            if store_key in adata.varm:
                del adata.varm[store_key]
        else:
            raise ValueError(f"Invalid del direction: {direction}")
        return adata
    # ########################################################################
    # move / copy operations
    if direction == "obs_to_obsm":
        # #############################
        # obs to obsm
        keys = keys or adata.obs.columns.tolist()
        df = adata.obs[keys].copy()
        if store_key in adata.obsm and not force_merge:
            raise KeyError(f"obsm['{store_key}'] already exists.")
        adata.obsm[store_key] = df
        if operation == "move":
            adata.obs.drop(columns=keys, inplace=True, errors="ignore")
    elif direction == "obsm_to_obs":
        # #############################
        # obsm to obs
        if store_key not in adata.obsm:
            raise KeyError(f"No obsm['{store_key}'] found.")
        df = pd.DataFrame(adata.obsm[store_key], index=adata.obs.index)
        for col in df.columns:
            if col in adata.obs and not force_merge:
                continue
            adata.obs[col] = df[col].values
        if operation == "move":
            del adata.obsm[store_key]
    elif direction == "var_to_varm":
        # #############################
        # var to varm
        keys = keys or adata.var.columns.tolist()
        df = adata.var[keys].copy()
        if store_key in adata.varm and not force_merge:
            raise KeyError(f"varm['{store_key}'] already exists.")
        adata.varm[store_key] = df
        if operation == "move":
            adata.var.drop(columns=keys, inplace=True, errors="ignore")
    elif direction == "varm_to_var":
        # #############################
        # varm to var
        if store_key not in adata.varm:
            raise KeyError(f"No varm['{store_key}'] found.")
        df = pd.DataFrame(adata.varm[store_key], index=adata.var.index)
        for col in df.columns:
            if col in adata.var and not force_merge:
                continue
            adata.var[col] = df[col].values
        if operation == "move":
            del adata.varm[store_key]
    else:
        raise ValueError(f"Invalid direction: {direction}")
    return adata


def _test_column_external(
            column_index: int,
            matrix: np.ndarray | csr_matrix,
            var_names: pd.Index
        ) -> tuple[str, float, float]:
    """Test normality of values in a matrix column.

    Args:
        column_index (int): Index of the column to test.
        matrix (numpy.ndarray | scipy.sparse.csr_matrix): Data matrix.
        variable_names (pandas.Index): Variable names.

    Returns:
        tuple[str, float, float]: Variable name, test statistic, p-value.
    """
    gene = var_names[column_index]
    values = matrix[:, column_index]
    values = values[~np.isnan(values)]
    if values.size < 8:
        return gene, np.nan, np.nan
    stat, pval = normaltest(values)
    return gene, stat, pval


def test_gene_normality_adata(
            adata: ad.AnnData,
            n_jobs: int = 7,
            plot_n: int = 0,
            layer: str = "scaled",
        ) -> pd.DataFrame:
    """Test all genes for normality in parallel.

    Low values Mean indicate less normal distributed.
    This function shows, that it makes no sense at all to use PCA on scaled
    log or lognorm counts. Consider creating the neighborhood graph on the log
    counts directly.

    Stores per-gene p-values and statistics in `adata.var` as
    '<layer>_normality_pvalue' and '<layer>_normality_stat'.

    Args:
        adata (anndata.AnnData): Adata object with samples x genes.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        plot_n (int, optional): Number of worst genes to plot. Defaults to 0.
        layer (str, optional): Data layer to use. Defaults to "scaled".
            If "X", uses adata.X.

    Returns:
        pd.DataFrame: DataFrame with columns [gene, stat, pvalue].
    """
    # #########################################################
    # Validate inputs
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an adata object.")
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer.")
    if not isinstance(plot_n, int) or plot_n < 0:
        raise ValueError("plot_n must be a 0 or greater integer.")
    if not isinstance(layer, str):
        raise TypeError("layer must be a string.")
    # #########################################################
    # Extract matrix efficiently
    if layer == "X":
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        matrix = adata.layers[layer]
    # #########################################################
    # Run tests in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_test_column_external)(i, matrix, adata.var_names)
        for i in range(matrix.shape[1]))
    results_df = pd.DataFrame(results, columns=["gene", "stat", "pvalue"])
    results_df = results_df.dropna().sort_values("pvalue", ascending=True)
    # #########################################################
    # Store results in adata.var with guaranteed alignment
    stats_series = results_df.set_index("gene")["stat"]
    pval_series = results_df.set_index("gene")["pvalue"]

    adata.var[f"{layer}_normality_stat"] = stats_series.reindex(
        adata.var_names, fill_value=np.nan)
    adata.var[f"{layer}_normality_pvalue"] = pval_series.reindex(
        adata.var_names, fill_value=np.nan)
    # #########################################################
    # Plot worst genes
    worst = results_df.head(plot_n)
    n_plots = len(worst)
    if n_plots > 0:
        nrows = min(n_plots, 5)
        ncols = 2 if n_plots > 1 else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2 * nrows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, gene in zip(axes, worst["gene"]):
            idx = adata.var_names.get_loc(gene)
            if issparse(matrix):
                gene_values = matrix[:, idx].toarray().ravel()
            else:
                gene_values = np.asarray(matrix[:, idx]).ravel()
            gene_values = gene_values[~np.isnan(gene_values)]
            ax.hist(gene_values, bins=100, alpha=0.7, color="steelblue")
            pval = worst.loc[worst["gene"] == gene, "pvalue"].iloc[0]
            ax.set_title(f"{gene} (p={pval:.2e})")

        for ax in axes[n_plots:]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    return results_df


# ###########################################################################################################
# Highly variable genes
def get_highly_variable(
            adata: ad.AnnData,
            return_adata: bool = True,
            return_genes: bool = False,
            sort_by_variance: bool = False
        ) -> ad.AnnData | list[str] | list:
    """
    Retrieve and optionally sort highly variable genes or a subsetted adata object.

    This function identifies highly variable genes in the provided adata object,
    ``adata``. Depending on the parameters, it returns a subsetted adata object
    containing only these highly variable genes, a list of these genes, or both.
    If sorting by variance is requested, the function will return the highly
    variable genes sorted by their variance according to the specified flavor in
    the configuration.

    NOTE:
        If both ``return_adata`` and ``return_genes`` are True, the function returns
        a list containing the subsetted adata object and the list of highly
        variable genes.

    Args:
        adata (anndata.AnnData): Adata object object containing the adata and
            associated metadata.
        return_adata (bool, optional): Determines whether to return a subsetted
            adata object. Defaults to True.
        return_genes (bool, optional): Determines whether to return a list of
            highly variable genes. Defaults to False.
        sort_by_variance (bool, optional): If True, returns the highly variable
            genes sorted by variance. Defaults to False.

    Returns:
        anndata.AnnData | list[str] | list:
            Depending on the arguments, the function returns one of the following:

                - A subsetted adata object (if
                  ``return_adata`` is True and ``return_genes`` is False).
                - A list of highly variable genes (if ``return_genes`` is True and
                  ``return_adata`` is False).
                - A list containing both the subsetted adata object and the list of
                  highly variable genes (if both ``return_adata`` and
                  ``return_genes`` are True).
                - A list of highly variable genes sorted by variance (if
                  ``sort_by_variance`` is True).

    Calls:
        mark_highly_variable_genes

    Called By:
        cluster_SC_scanpy_like, downstream_preprocessing, get_proper_random_ref,
        get_random_genes, run_downstream, score_genes, score_genes_parallel

    TODO:
        Extend sorting behavior for additional flavors as needed.

    Tags:
        config, normalization
    """
    # #########################################################
    # Check if 'highly_variable_rank' is already calculated
    # If not, calculate highly variable genes and update adata object
    if "highly_variable_rank" not in adata.var.keys():
        mark_highly_variable_genes(adata)
    # #########################################################
    # Handle the sorting of highly variable genes by variance if requested
    if sort_by_variance:
        # #######################
        # Handle different flavors for sorting highly variable genes
        if adata.uns["config"]["pp"]["highly_variable_genes"]["flavor"] == 'seurat_v3':
            # ##########################################
            # Return sorted genes by highly_variable_rank for 'seurat_v3' flavor
            sorted_genes = adata.var["highly_variable_rank"].dropna().sort_values(ascending=True).index.tolist()
            # ##########################################
        elif adata.uns["config"]["pp"]["highly_variable_genes"]["flavor"] == 'cell_ranger':
            # ##########################################
            # Return sorted genes by dispersions for 'cell_ranger' flavor
            sorted_genes = (
                adata[adata.var["highly_variable"]]
                .var["dispersions"]
                .dropna().sort_values(ascending=True)
                .index.tolist())
            # ##########################################
        elif adata.uns["config"]["pp"]["highly_variable_genes"]["flavor"] == 'seurat':
            # ##########################################
            # Return sorted genes by dispersions_norm for 'seurat' flavor
            sorted_genes = (
                adata[adata.var["highly_variable"]]
                .var["dispersions"]
                .dropna().sort_values(ascending=True)
                .index.tolist())
            # ##########################################
        elif adata.uns["config"]["pp"]["highly_variable_genes"]["flavor"] == "ours":
            # ##########################################
            # Return sorted genes by highly_variable_rank for custom 'ours' flavor
            sorted_genes = (
                adata.var["highly_variable_rank"]
                .dropna().sort_values(ascending=False)
                .index.tolist())
            # ##########################################
        # Return the sorted genes if no other return type is requested
        if not return_adata and not return_genes:
            return sorted_genes
    # #########################################################
    # Return the appropriate data based on the function parameters
    if return_adata and return_genes:
        # ##########################################
        # Return both the subsetted adata object and the list of highly variable genes
        return adata[:, adata.var.highly_variable], adata.var_names[adata.var.highly_variable]
    elif return_adata:
        # ##########################################
        # Return only the subsetted adata object
        return adata[:, adata.var.highly_variable]
    if return_genes:
        # ##########################################
        # Return only the list of highly variable genes
        return adata.var_names[adata.var.highly_variable]


def calc_highly_variable_genes_unique_based(
            adata: ad.AnnData,
            inplace: bool = True,
            key: str = "highly_variable"
        ) -> pd.DataFrame | None:
    """
    Identifies highly variable genes based on unique counts and updates the
    adata object.

    This function calculates highly variable genes in the adata object based on
    the number of unique counts (``n_unique``) for each gene. It ranks the genes
    based on their cumulative sum and marks those that are above the median as
    highly variable. The results are either updated in-place or returned as a
    DataFrame depending on the ``inplace`` parameter.

    NOTE:
        Ensure that ``n_unique`` is calculated and present in ``adata.var`` before
        using this function.

    Args:
        adata (anndata.AnnData): Adata object where ``adata.var`` contains gene
            metadata.
        inplace (bool, optional): If True, updates ``adata.var`` with the results.
            If False, returns the results as a DataFrame. Default is True.
            Defaults to True.
        key (str, optional): The key under which the highly variable genes and
            their rank will be stored in ``adata.var``. Default is
            "highly_variable". Defaults to "highly_variable".

    Returns:
        pandas.DataFrame | None:
            pd.DataFrame or None: If ``inplace`` is False, returns a DataFrame
            containing the ranked genes and their highly variable status.
            Otherwise, returns None.

    Raises:
        AttributeError: If ``n_unique`` is not present in ``adata.var``, an
            exception is raised indicating that the necessary preprocessing has
            not been performed.

    Called By:
        Run_all_prep_steps_clustering, mark_highly_variable_genes

    TODO:
        Consider extending the function to allow different thresholds or
        methods for identifying highly variable genes.

    Tags:
        calculation, normalization
    """
    # #########################################################
    # Validate the existence of 'n_unique' in adata.var
    if "n_unique" not in adata.var.keys():
        raise AttributeError("adata.var has not 'n_unique' key! Please run the default Preprocessing.")
    # #########################################################
    # Sort 'n_unique' values and calculate the cumulative sum
    this_data = adata.var["n_unique"].copy().sort_values()
    cumsum = this_data.cumsum()
    cumsum_sum = cumsum.values[-1]
    cumsum = cumsum.to_frame()

    # Select genes that have cumulative sums above the median
    cumsum[key] = False
    cumsum.iloc[np.argmax(cumsum["n_unique"] > cumsum_sum / 2):, np.argmax(cumsum.columns == key)] = True
    # Get the rank for each gene NOTE: It is sorted a arange should be sufficient
    # cumsum[f'{key}_rank'] = np.arange(cumsum.shape[0], dtype=np.uint16) + 1
    cumsum[f'{key}_rank'] = cumsum["n_unique"].rank(method="dense").astype(np.uint16)
    # OLD:
    # cumsum = pd.DataFrame(cumsum[np.argmax(cumsum > cumsum_sum / 2):])
    # Rank the selected genes and mark them as highly variable
    # cumsum[key] = True
    cumsum[key] = cumsum[key].astype(np.bool_).astype("category")
    # #########################################################
    # Update adata.var with the highly variable genes and their rank
    if inplace:
        # ##########################################
        # Set initial values to False and NaN
        adata.var[key] = cumsum.loc[adata.var_names, key]
        adata.var[f'{key}_rank'] = cumsum.loc[adata.var_names, f'{key}_rank']

        # Update adata.var with the new calculated values
        # adata.var.update(cumsum)

        # Convert the key column to a boolean data type
        # adata.var[key] = adata.var[key].astype(np.bool_)
    else:
        # ##########################################
        # Return the DataFrame containing the results
        # Remove 'n_unique' from the resulting DataFrame
        del cumsum["n_unique"]
        return cumsum


# ###########################################################################################################
# DEG
def merge_for_deg(
            adata: ad.AnnData,
            reference_adata: ad.AnnData,
            groupby: str = "leiden",
            reference_name: str = "reference"
        ) -> ad.AnnData:
    """Add reference Cells to the adata for DEG analysis.

    Use-case: You want to calculate DEGs for sub-clusters vs clusters of the
        general clustering.

    The resulting adata will contain a unified ``obs[groupby]`` column,
    using the original group values for ``query_adata`` and a constant label
    for ``reference_adata``.

    Args:
        adata (anndata.AnnData): Adata object containing the groups to test.
        reference_adata (anndata.AnnData): Adata object to use as
            background/reference.
        groupby (str, optional): Column in ``query_adata.obs`` to use for
            grouping. Defaults to "leiden".
        reference_name (str, optional): Label to assign to all reference cells.
            Defaults to "reference".

    Returns:
        anndata.AnnData:
            Merged object with unified ``.obs[groupby]``.

    Calls:
        validate_groupby_column

    Tags:
        DEG, groupby
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    validate_groupby_column(
        reference_adata.obs, groupby, check_categorical=True, print_name="reference_adata.obs")
    # #########################################################
    query = adata.copy()
    reference = reference_adata.copy()
    reference.obs[groupby] = reference_name

    merged = ad.concat([query, reference], join="inner", merge="same")
    merged.uns = adata.uns.copy()

    # make the groupby categorical again.
    merged.obs[groupby] = merged.obs[groupby].astype("category")
    return merged


def calc_DEGs(
            adata: ad.AnnData,
            groupby: str | None = None,
            group_ref: list[list[str]] | None = None,
            groups: list[str] | str | None = None,
            references: list[str] | str | None = None,
            n_cores: int = 7,
            inplace: bool = True,
            one_df_key: str = "rank_genes_groups",
            layer: str = "log2norm_counts",
            lfc_layer: str = "norm_counts",
        ) -> dict[str, pd.DataFrame]:
    """
    Perform differential gene expression (DGE) analysis based on Wilcoxon rank
    test for one or more groups versus one or more references within the adata
    object.

    This function computes the DGE between specified groups and corresponding
    references in the adata object, using the Wilcoxon rank test. If references
    is a single string, it will be used as the reference for all groups. The
    results are stored in the ``uns`` attribute of the adata object under keys
    generated based on the group and reference names or under a custom key if
    provided.

    NOTE:
        - The groups and references must be present in
          ``adata.uns["obs"][<groupby>]``.
        - The default lfc_layer for the logfold changes may be the norm counts
          because people tend to use it. Using the acutal counts is less biased,
          consider using "counts".

    Args:
        adata (anndata.AnnData): Adata object.
        groupby (str | None, optional): The key in ``adata.obs`` used to group the
            data by. Defaults to None.
        group_ref (list[list[str]] | None, optional): A list of [group,
            reference] pairs. If provided, it overrides ``groups`` and
            ``references``. Defaults to None.
        groups (list[str] | str | None, optional): A list of group labels or a
            single group label within ``adata.uns["obs"][<groupby>]`` to calculate
            DEGs for. Ignored if ``group_ref`` is provided. Defaults to None.
        references (list[str] | str | None, optional): Reference labels within
            ``adata.uns["obs"][<groupby>]`` to compare against. If a list, must
            match the length of ``groups``. If a string, it will be used for all
            groups. Ignored if ``group_ref`` is provided. Defaults to None.
        key (str, optional): A custom key to store the results in ``adata.uns``.
            Defaults to none, which results in
            {group}_vs_{reference}_rank_gene".
        n_cores (int, optional): Number of threads to use for parallel
            processing. Defaults to 7.
        one_df_key (str, optional): If string, report only one dataframe in the
            adata.uns[one_df_key] Defaults to "rank_genes_groups".
        layer (str, optional): Layer in ``adata`` to use instead of the default
            expression matrix. Defaults to "log2norm_counts".
        lfc_layer (str, optional): Layer in ``adata`` to use instead for the
            calculation of the logfold changes. Defaults to "norm_counts".

    Returns:
        dict[str, pandas.DataFrame]:
            Dictionary with each element in ``groups`` as one key with a DEG
            pandas.DataFrame.

    Raises:
        AttributeError:
            If ``references`` is a list and does not match the size of``groups``.
        AttributeError:
            If ``groupby`` is not in ``adata.obs`` keys.
        AttributeError:
            If adata.X is not a csr_matrix (to ensure no scaled data used).

    Calls:
        get_deg_df, validate_groupby_column

    Called By:
        calc_DEGs_multi_group, downstream_preprocessing

    Tags:
        DEG, groupby
    """
    # ###############################################
    # Input validation and setup
    if layer is not None:
        if layer not in adata.layers.keys():
            raise ValueError(f"layer {layer} not in adata.layers!")
        else:
            if not isinstance(adata.layers[layer], csr_matrix):
                raise AttributeError(
                    "Please make sure to use a sparse csr_matrix with log2norm counts")
            adata.X = adata.layers[layer].copy()
    elif adata.X is None:
        raise ValueError("adata.X is None!")
    # #########################################################
    # Get the default leiden cluster key from the config
    if groupby is None:
        if "cluster_algorithm" in adata.uns["config"]["general"].keys():
            groupby = adata.uns["config"]["general"]["cluster_algorithm"]
        else:
            raise AttributeError(f'No default cluster key set in  {groupby} '
                                 'adata.uns["config"]["general"]["cluster_algorithm"], please check!')

    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True, groups=groups)
    # ###################
    # If group_ref is provided, extract groups and references
    if group_ref is not None:
        groups, references = zip(*group_ref)
    else:
        if groups is None:
            if adata.obs[groupby].dtype.name == "category":
                groups = adata.obs[groupby].cat.categories
            else:
                groups = adata.obs[groupby].unique().sort()
                # TODO: is a check necessary for the parallel run?!?
                # if len(groups) > adata.shape[0] * .3:
                #    raise AttributeError("You try to calculate the DEG on many clusters")
        elif isinstance(groups, str):
            groups = [groups]
        if references is None:
            references = ["rest"] * len(groups)
        elif isinstance(references, str):
            # Remove the reference from the group, doesn't make sense to caluclate DEGs for the same cells
            groups = [g for g in groups if g != references]
            references = [references] * len(groups)
        if len(groups) != len(references):
            raise AttributeError("The number of groups must match the number of references!")
    # ###############################################
    # Check if all the groups are there and don't have too variable numbers
    # TODO: implement check for large imbalance because it degs doesn't perform well!
    # if False:
    #     vc = adata.obs[groupby].value_counts()
    #     for g, r in zip(groups, references):
    #         pass
    # ###############################################
    # Initialize the configuration dictionary and result container
    dict_ = copy(adata.uns["config"]["tl"]["rank_genes_groups"])
    # TODO: for actual developement you can delete it
    if "layer" in dict_.keys():
        if dict_["layer"] == "None":
            dict_["layer"] = None

    dict_["groupby"] = groupby
    results = {}
    # ###############################################
    # Function to process each group-reference pair

    def process_group_reference_pair(
                group,
                reference,
                dict_,
                groupby
            ) -> tuple[str, pd.DataFrame]:
        """Helper"""
        dict_["groups"] = [group]
        dict_["reference"] = reference

        temp_key = f'{group}_vs_{reference}_rank_gene'
        dict_["key_added"] = temp_key
        # Perform the differential gene expression analysis
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
            sc.tl.rank_genes_groups(adata, **dict_)
        # ###############################################
        # Retrieve and process the results + fix missing pct zero
        result_df = get_deg_df(adata, group, deg_key=temp_key)
        # ###############################################
        # Delete the key
        if temp_key in adata.uns.keys():
            del adata.uns[temp_key]
        # ###############################################
        # Get rid of all empty genes for the conditions
        result_df = result_df[result_df["scores"] != 0]
        # ###############################################
        if reference == "rest":
            gene_non_zero = adata[adata.obs[groupby] != group, result_df["names"]].X.getnnz(0)
            len_of_group = sum(adata.obs[groupby] != group)
            # Calculate the reference mean
            if lfc_layer in adata.layers:
                X_ref = adata[adata.obs[groupby] != group, result_df["names"]].layers[lfc_layer].copy()
            else:
                X_ref = adata[adata.obs[groupby] != group, result_df["names"]].X.copy()
                X_ref.data = np.expm1(X_ref.data)
        else:
            gene_non_zero = adata[adata.obs[groupby] == reference, result_df["names"]].X.getnnz(0)
            len_of_group = sum(adata.obs[groupby] == reference)
            # Calculate the reference mean
            if lfc_layer in adata.layers:
                X_ref = adata[adata.obs[groupby] == reference, result_df["names"]].layers[lfc_layer].copy()
            else:
                X_ref = adata[adata.obs[groupby] == reference, result_df["names"]].X.copy()
                X_ref.data = np.expm1(X_ref.data)

        # TODO: if the len_of_group is 0, the other calculations are not necessary!!!!
        # Calculate the lfc
        X_ref = np.array(X_ref.mean(axis=0))[0]
        if lfc_layer in adata.layers:
            X_group = adata[adata.obs[groupby] == group, result_df["names"]].layers[lfc_layer].copy()
        else:
            X_group = adata[adata.obs[groupby] == group, result_df["names"]].X.copy()
            X_group.data = np.expm1(X_group.data)
        X_group = np.array(X_group.mean(axis=0))[0]
        offset = 1  # Mathematically makes sense, using scanpys 1e-9 overestimates close to 0 in one direction
        result_df["logfoldchanges"] = np.log2(
                (X_group + offset)
                / (X_ref + offset))

        if len_of_group == 0:
            result_df["pct_nz_reference"] = 0
        else:
            # For the scampy inconveniance dividing by 0, we replace the nans with 0
            result_df["pct_nz_reference"] = [
                k / len_of_group for k in gene_non_zero]
        result_df["pct_nz_group"] = result_df["pct_nz_group"].fillna(0, inplace=False)
        # #################
        # add the group and reference to the DEG df
        keys_ = result_df.columns.tolist()

        result_df["group"] = group
        result_df["group"] = result_df["group"].astype("category")
        result_df["reference"] = reference
        result_df["reference"] = result_df["reference"].astype("category")
        result_df = result_df[["group", "reference"] + keys_]
        # #################
        return temp_key, result_df
    # ###############################################
    # Perform the differential gene expression analysis in parallel
    try:
        with ThreadPoolExecutor(max_workers=min(n_cores, len(groups))) as executor:
            future_to_group = {
                    executor.submit(
                            process_group_reference_pair, group, reference,
                            dict_, groupby): group for group, reference in zip(groups, references)}
            for future in as_completed(future_to_group):
                group, result_df = future.result()
                results[group] = result_df
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
    # ###############################################
    if one_df_key is not None:
        result_df = pd.concat(
            [x for x in results.values() if not x.empty],
            axis=0, ignore_index=True)
        # result_df = result_df.sort_values(
        #    by=["group", "reference"],
        #    ascending=[True, True])
        result_df = result_df.sort_values(
            by=["group", "reference", "scores"],
            ascending=[True, True, False],
            key=lambda col: col.abs() if col.name == "scores" else col)

    if inplace:
        if one_df_key is None:
            for temp_key, res in results.items():
                adata.uns[temp_key] = res
        else:
            adata.uns[one_df_key] = result_df
    else:
        if one_df_key is None:
            return results
        else:
            return result_df


'''
long_doc.
    Long explaination for calc_DEGs_multi_group()
    Mode table for parameter combinations:

    | # | group_ref | groups | references | group_ref_cond | group_cond | reference_cond | Meaning / Behavior                                                                                    |  # noqa: E501
    |---|-----------|--------|------------|----------------|------------|----------------|-------------------------------------------------------------------------------------------------------|  # noqa: E501
    | 1 |    Y      |   N    |     N      |       N        |     N      |       N        | Use group_ref directly. No expansion. Overrides all.                                                  |  # noqa: E501
    | 2 |    N      |   Y    |     Y      |       N        |     N      |       N        | Use groups and references to build explicit pairs. Lengths must match.                                |  # noqa: E501
    | 3 |    N      |   Y    |     N      |       N        |     N      |       N        | INVALID - references missing, cannot build pairs.                                                     |  # noqa: E501
    | 4 |    N      |   Y    |   str      |       N        |     N      |       N        | Broadcast references string to match groups, remove groups matching references.                       |  # noqa: E501
    | 5 |    N      |   N    |     N      |       Y        |     N      |       N        | Expand group_ref_cond per cluster using cluster_key.                                                  |  # noqa: E501
    | 6 |    N      |   N    |     N      |       N        |     Y      |       Y        | Auto-expand per cluster: group_cond vs reference_cond.                                                |  # noqa: E501
    | 7 |    N      |   N    |     N      |       N        |     N      |       Y        | Default ref mode: In each cluster, compare each condition (not reference) vs reference_cond.          |  # noqa: E501
    | 8 |    N      |   N    |     N      |       N        |     Y      |       N        | INVALID - missing reference_cond. Both required for auto-expand.                                      |  # noqa: E501
    | 9 |    N      |   N    |     N      |       N        |     N      |       N        | Full default sweep: In each cluster, compare every condition vs every other condition (one-vs-rest).  |  # noqa: E501

    Legend: Y = set, N = not set, str = string broadcast case.
    # ###############################
    # Create dummy AnnData
    obs = pd.DataFrame({
        "leiden": ["0", "0", "1", "1", "2", "2", "2"],
        "condition": ["A", "B", "A", "C", "A", "B", "C"]
    })
    X = np.random.rand(obs.shape[0], 5)
    dummy_adata = ad.AnnData(X=X, obs=obs)
    # ###############################
    # Case 1
    res1 = calc_DEGs_multi_group(dummy_adata, group_ref=[["0_A", "0_B"], ["1_A", "1_C"]], clusters=["0"])
    print("Case 1:", res1)
    # >>> Case 1: [['0_A', '0_B']]
    # ###############################
    # Case 2
    res2 = calc_DEGs_multi_group(dummy_adata, groups=["0_A", "1_A"], references=["0_B", "1_C"], clusters=["0"])
    print("Case 2:", res2)
    # >>> Case 2: [['0_A', '0_B']]
    # ###############################
    # Case 3 (invalid, missing references)
    try:
        res3 = calc_DEGs_multi_group(dummy_adata, groups=["0_A", "1_A"], clusters=["0"])
    except Exception as e:
        print("Case 3 (expected error):", e)
    # >>> Case 3 (expected error): references must be provided when groups are specified explicitly!
    # ###############################
    # Case 4 (broadcast string)
    res4 = calc_DEGs_multi_group(dummy_adata, groups=["0_A", "1_A", "2_B"], references="0_C", clusters=["0"])
    print("Case 4:", res4)
    # >>> Case 4: [['0_A', '0_C']]
    # ###############################
    # Case 5
    res5 = calc_DEGs_multi_group(dummy_adata, group_ref_cond=[["A", "B"]], clusters=["0"])
    print("Case 5:", res5)
    # >>> Case 5: [['0_A', '0_B']]
    # ###############################
    # Case 6
    res6 = calc_DEGs_multi_group(dummy_adata, group_cond=["A", "C"], reference_cond="B", clusters=["0"])
    print("Case 6:", res6)
    # >>> Case 6: [['0_A', '0_B']]
    # ###############################
    # Case 7
    res7 = calc_DEGs_multi_group(dummy_adata, reference_cond="A", clusters=["0"])
    print("Case 7:", res7)
    # >>> Case 7: [['0_B', '0_A']]
    # ###############################
    # Case 8 (invalid, missing reference_cond)
    try:
        res8 = calc_DEGs_multi_group(dummy_adata, group_cond=["A", "C"], clusters=["0"])
    except Exception as e:
        print("Case 8 (expected error):", e)
    # >>> Case 8 (expected error): references must be provided when groups are specified explicitly!
    # ###############################
    # Case 9
    res9 = calc_DEGs_multi_group(dummy_adata, clusters=["0"])
    print("Case 9:", res9)
    # >>> Case 9: [['0_B', '0_A'], ['0_A', '0_B']]
'''


def calc_DEGs_multi_group(
            adata: "ad.AnnData",
            cluster_key: str = "leiden",
            condition_obs_key: str = "condition",
            deg_obs_key: str | None = None,
            deg_key: str | None = None,
            group_ref: list[list[str]] | None = None,
            groups: list[str] | None = None,
            references: list[str] | None = None,
            group_ref_cond: list[list[str]] | None = None,
            group_cond: str | list[str] | None = None,
            reference_cond: str | None = None,
            clusters: list[str] | None = None,
            min_cells: int = 100,
            n_cores: int = 10,
        ) -> None:
    """
    Run differential expression analysis using explicit group pairs or
    automatically build per-cluster condition-based comparisons.

    Args:
        adata ("ad.AnnData"): Adata object.
        cluster_key (str, optional): Key in obs for clusters (default:
            "leiden"). Defaults to "leiden".
        condition_obs_key (str, optional): Key in obs for conditions (default:
            "condition"). Defaults to "condition".
        deg_obs_key (str | None, optional): Name of the combined
            cluster-condition column in obs. Defaults to None.
        deg_key (str | None, optional): Key to store DEG results in adata.uns.
            Defaults to None.
        group_ref (list[list[str]] | None, optional): Explicit list of [group,
            reference] pairs using deg_obs_key. Defaults to None.
        groups (list[str] | None, optional): List of group names (deg_obs_key
            values). Defaults to None.
        references (list[str] | None, optional): Corresponding list of reference
            names (deg_obs_key values). Defaults to None.
        group_ref_cond (list[list[str]] | None, optional):
            List of [group_condition, reference_condition] pairs to expand per
            cluster. Defaults to None.
        group_cond (str | list[str] | None, optional): Condition(s) to compare
            against the reference condition per cluster. Defaults to None.
        reference_cond (str | None, optional): Condition used as reference for
            per-cluster comparisons. Defaults to None.
        clusters (list[str] | None, optional): Subset of clusters to use
            (default: all clusters). Defaults to None.
        min_cells (int, optional): Minimal number of cells per group to consider
            for DEG calculation. Defaults to 100.
        n_cores (int, optional): Number of cores for parallel computation
            (default: 10). Defaults to 10.

    Returns:
        None:
            Updates adata.uns with DEG results.

    Raises:
        AttributeError: If group/reference specification is inconsistent or
            missing.

    Calls:
        calc_DEGs, check_ct_abundances

    TODO:
        This could be generalized to n categorical columns by using
        utils.join_categorical_columns and have a list of keys instead
        cluster_key and condition_obs_key, nested group_ref, groups and
        references.

    Tags:
        DEG, groupby
    """
    # ###############################
    # Default keys
    if deg_obs_key is None:
        deg_obs_key = cluster_key + "_" + condition_obs_key
    if deg_key is None:
        deg_key = f"{condition_obs_key}_per_{cluster_key}_rank_genes_groups"
    # ###############################
    # Create combined key in obs
    adata.obs[deg_obs_key] = (
        adata.obs[cluster_key].astype(str) + "_" +
        adata.obs[condition_obs_key].astype(str)
        ).astype("category")
    combined_uniques = adata.obs[deg_obs_key].unique()
    selected_clusters = clusters if clusters is not None else adata.obs[cluster_key].unique()
    # ###############################
    # Mode 1: Case 1 - group_ref provided directly
    if group_ref is not None:
        if clusters is not None:
            final_group_ref = [
                [g, r]
                for g, r in group_ref
                if g.split("_")[0] in clusters and r.split("_")[0] in clusters]
        else:
            final_group_ref = group_ref
    # ###############################
    # Mode 2, 3, 4 - groups and references
    elif groups is not None:
        # Case 2, 3, 4
        if isinstance(groups, str):
            groups = [groups]

        if references is None:
            # Case 3 - INVALID
            raise AttributeError("references must be provided when groups are specified explicitly!")

        elif isinstance(references, str):
            # Case 4 - broadcast
            groups = [g for g in groups if g != references]  # Case 4
            references = [references] * len(groups)         # Case 4

        if len(groups) != len(references):
            raise AttributeError("The number of groups must match the number of references!")

        # Case 2, 4 - final pairing
        final_group_ref = [[g, r] for g, r in zip(groups, references)]
    # ###############################
    # Mode 5: Case 5 - group_ref_cond expansion per cluster
    elif group_ref_cond is not None:
        # Check if flat list of exactly two strings - convert to nested
        if (isinstance(group_ref_cond, list)
                and all(isinstance(x, str) for x in group_ref_cond)
                and len(group_ref_cond) == 2):
            group_ref_cond = [group_ref_cond]

        final_group_ref = []

        for cluster in selected_clusters:
            for cond_group, cond_ref in group_ref_cond:
                group = f"{cluster}_{cond_group}"
                ref = f"{cluster}_{cond_ref}"

                if group in combined_uniques and ref in combined_uniques:
                    final_group_ref.append([group, ref])
                else:
                    logger.warning(f"Skipping invalid group-ref pair: {group} vs {ref}")
    # ###############################
    # Mode 6: Case 6 - group_cond and reference_cond both set
    elif group_cond is not None and reference_cond is not None:
        group_conditions = [group_cond] if isinstance(group_cond, str) else group_cond
        final_group_ref = []

        for cluster in selected_clusters:
            ref = f"{cluster}_{reference_cond}"  # Case 6

            for g_cond in group_conditions:
                group = f"{cluster}_{g_cond}"  # Case 6

                if group == ref:
                    continue

                if group in combined_uniques and ref in combined_uniques:
                    final_group_ref.append([group, ref])
                else:
                    logger.warning(f"Skipping invalid group-ref pair: {group} vs {ref}")
    # ###############################
    # Mode 7: Case 7 - only reference_cond set
    elif reference_cond is not None:
        final_group_ref = []

        for cluster in selected_clusters:
            ref = f"{cluster}_{reference_cond}"  # Case 7

            for cond in adata.obs[condition_obs_key].unique():
                if cond == reference_cond:
                    continue

                group = f"{cluster}_{cond}"  # Case 7

                if group in combined_uniques and ref in combined_uniques:
                    final_group_ref.append([group, ref])
                else:
                    logger.warning(f"Skipping invalid group-ref pair: {group} vs {ref}")
    # ###############################
    # Mode 9: Case 9 - full sweep
    else:
        final_group_ref = []

        for cluster in selected_clusters:
            cluster_conditions = adata.obs.loc[adata.obs[cluster_key] == cluster, condition_obs_key].unique()

            for ref_cond in cluster_conditions:
                ref = f"{cluster}_{ref_cond}"  # Case 9

                for group_cond in cluster_conditions:
                    if group_cond == ref_cond:
                        continue

                    group = f"{cluster}_{group_cond}"  # Case 9

                    if group in combined_uniques and ref in combined_uniques:
                        final_group_ref.append([group, ref])
                    else:
                        logger.warning(f"Skipping invalid group-ref pair: {group} vs {ref}")
    # ###############################
    # Sanity filter for the comparions
    final_group_ref = check_ct_abundances(
        adata, deg_obs_key, final_group_ref, min_cells)
    # ###############################
    # Return or pass final_group_ref
    calc_DEGs(
        adata,
        groupby=deg_obs_key,
        group_ref=final_group_ref,
        n_cores=n_cores,
        one_df_key=deg_key)


def check_ct_abundances(
            adata: ad.AnnData,
            deg_obs_key: str,
            group_ref: list[list[str]],
            min_cells: int = 100,
            report_only: bool = False,
        ) -> pd.DataFrame | list[list[str]]:
    """
    Check cell abundances for each group-reference pair, and log pairs below a
    cell threshold.

    Args:
        adata (anndata.AnnData): Adata object.
        deg_obs_key (str): Combined key column in obs to count cells.
        group_ref (list[list[str]]): List of [group, reference] pairs.
        min_cells (int, optional): Minimum number of cells required per group
            and reference (default: 10). Defaults to 100.
        report_only (bool, optional): If True, log low-abundance pairs and
            return full DataFrame. If False, log and return filtered list above
            threshold. Defaults to False.

    Returns:
        pandas.DataFrame | list[list[str]]:

            - DataFrame with all pairs and counts if report_only=True.
            - Filtered list of pairs above threshold if report_only=False.

    Raises:
        ValueError: If all group-reference pairs are removed when report_only is
            False.

    Calls:
        StatKeeper.get

    Called By:
        calc_DEGs_multi_group

    Tags:
        DEG, obs, utils
    """
    value_counts = adata.obs[deg_obs_key].value_counts()

    filtered_pairs = []
    for group, ref in group_ref:
        g_count = value_counts.get(group, 0)
        r_count = value_counts.get(ref, 0)
        keep = (g_count >= min_cells) and (r_count >= min_cells)

        if keep:
            filtered_pairs.append([group, ref])
        else:
            if report_only:
                logger.info(
                    f"Low-abundance pair: {group} ({g_count} cells) vs {ref} ({r_count} cells)")
            else:
                logger.warning(
                    f"Removed low-abundance pair: {group} ({g_count} cells) vs {ref} ({r_count} cells)")

    if not filtered_pairs:
        raise ValueError("All group-reference pairs were removed due to insufficient cell counts.")
    else:
        return filtered_pairs


def get_DEGs_per_group(
            adata: ad.AnnData,
            groups: str | list[str] | None = None,
            n_genes: int = 100,
            perc: float = 0.1,
            p_val_cutoff: float = 0.01,
            lfc: float = 0.0,
            direction: str = "up_n_down",
            deg_key: str = "rank_genes_groups"
        ) -> dict[str, pd.DataFrame]:
    """Get filtered DEGs per cluster from the adata object.

    This function retrieves and filters differentially expressed genes (DEGs)
    for specified groups/clusters (or all groups if none are specified) based on
    various criteria such as the number of genes, percentage of cells expressing
    the gene, p-value cutoff, log-fold change, and direction of regulation.

    NOTE:
        - n_genes: Set to a very high number if you want to retrieve all
          genes without a limit.
        - perc: Ensure this percentage is appropriate for your dataset.

    Args:
        adata (anndata.AnnData): Adata object.
        groups (str | list[str] | None, optional): The cluster or list of groups
            to get the DEGs for. Defaults to None, meaning all groups are
            considered. Defaults to None.
        n_genes (int, optional): Maximum number of DEGs to return per cluster.
            Defaults to 100.
        perc (float, optional): Minimum percentage of cells in the cluster
            expressing the gene. Defaults to 0.1.
        p_val_cutoff (float, optional): Adjusted p-value cutoff for filtering.
            Defaults to 0.01.
        lfc (float, optional): Minimum log-fold change to display.
            Defaults to 0.0.
        direction (str, optional): Direction of regulation to consider. Options
            are "up_n_down", "up", "down", and "force_up_n_down". Defaults to
            "up_n_down". Defaults to "up_n_down".
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".

    Returns:
        dict[str, pandas.DataFrame]:
            Dictionary with the DEGs and stats per cluster.

    Calls:
        get_deg_df, subset_degs

    Called By:
        get_DEGs_per_group_for_plotting, plot_per_group_DEG_umaps,
        run_downstream

    TODO:
        - direction: Update the documentation if new direction options are
          added.
    """
    # #########################################################
    # Check if the deg_key is present in the adata object
    if deg_key not in adata.uns.keys():
        raise AttributeError(f"Attention: The key={deg_key} is not in adata.uns. You need to run "
                             "sc_code.downstream_preprocessing first!")
    # #########################################################
    # Check if the direction parameter is valid
    if direction not in ["up_n_down", "force_up_n_down", "up", "down"]:
        raise ValueError(f'Parameter direction {direction} is not correct. Please use one of: '
                         '"up_n_down", "force_up_n_down", "up", "down"')
    # #########################################################
    # Set default for n_genes if None and ensure it is an integer
    if n_genes is None:
        n_genes = int(1e200)
    if not isinstance(n_genes, int):
        n_genes = int(n_genes)
    # #########################################################
    # Handle the groups parameter: use all groups if None, or convert to a list if it's a string
    if groups is None:
        # groups = list(adata.uns[deg_key]["names"].dtype.names)
        groups = adata.uns[deg_key]["group"].unique().tolist()
    elif isinstance(groups, str):
        groups = [groups]
    # #########################################################
    # Retrieve the DEGs dataframe from the adata object
    df = get_deg_df(adata, deg_key=deg_key).copy()
    # #########################################################
    # Filter the DEGs dataframe using the specified criteria
    filtered_df = subset_degs(
        df, groups=groups, n_genes=n_genes,
        perc=perc, p_val_cutoff=p_val_cutoff,
        lfc=lfc, direction=direction)
    # #########################################################
    # Prepare the output dictionary with DEGs per cluster
    dict_ = {str(group): group_df["names"].tolist() for group, group_df in filtered_df.groupby("group")}

    return dict_


def subset_degs(
            df: pd.DataFrame,
            groups: str | list[str] | None = None,
            n_genes: int = int(1e300),
            perc: float = 0.0,
            p_val_cutoff: float = 1.0,
            lfc: float = 0.0,
            direction: str = "up_n_down"
        ) -> pd.DataFrame:
    """Subset the rank gene groups.

    All default parameters are no filtering and it always returns a copy!

    A good baseline for multiple cell types would be::
        n_genes=50, perc=.2, p_val_cutoff=1-3, lfc=0.5

    A good baseline for one cell types while subclustering would be::
        n_genes=20, perc=.33, p_val_cutoff=1e-5, lfc=1
        The idea is, to get a more specific group and ignore all the genes
        that are actually present in all the clusters, because they are the same
        cell type.

    Both are debatable, so you should ask your collaboration partners if they
    want specific parameters.

    Args:
        df (pandas.DataFrame): The Dataframe with the DEGS. Can contain multiple
            groups or only one.
        subset_degs (str, optional): The groups to subset, if None, then all are
            used, Defaults to None.
        n_genes (int, optional): The maximal number of genes to show.
            Defaults to int(1e300.
        perc (float, optional): The minimal percentage of cells in the group
            expressing the gene. Defaults to .0.
        p_val_cutoff (int, optional): Adjusted p-value cutoff. Defaults to 1.
        lfc (int, optional): Minimal Log-Fold-Change to show. Defaults to 0.

    Returns:
        pandas.DataFrame:
            Subsetted dataframe

    Called By:
        create_deg_table, get_DEGs_per_group, plot_hallmark_group_heatmap,
        get_adata_subset, process_and_save_degs
    """
    if direction == "up_n_down":
        def abs_wrapper(x):
            """Helper"""
            if pd.api.types.is_numeric_dtype(x.dtype):
                return x.abs()
            else:
                return x
        df = df[df["scores"].abs() > 0].sort_values(["scores"], ascending=[False], key=abs_wrapper)

    elif direction == "up":
        df = df[df["scores"] > 0].sort_values(["scores"], ascending=[False])
    elif direction == "down":
        df = df[df["scores"] < 0].sort_values(["scores"], ascending=[True])

    if "group" in df.keys():
        if groups is None:
            groups = df["group"].unique()
        elif isinstance(groups, list):
            groups = np.array(groups)
        if isinstance(groups, np.ndarray):
            df = pd.concat([
                x for x in [
                    df.loc[
                        (df["group"] == group)
                        & ((df["pct_nz_group"] >= perc) | (df["pct_nz_reference"] >= perc))
                        & (df["pvals_adj"] <= p_val_cutoff)
                        & (df["logfoldchanges"].abs() >= lfc)
                    ][:min(n_genes, df.shape[0])]
                    for group in sorted(groups)
                ] if not x.empty])
        elif isinstance(groups, str):
            df = df[(df["group"] == groups)
                    & ((df["pct_nz_group"] >= perc) | (df["pct_nz_reference"] >= perc))
                    & (df["pvals_adj"] <= p_val_cutoff)
                    & (df["logfoldchanges"].abs() >= lfc)][:min(n_genes, df.shape[0])]
    return df


def subset_degs_by_ref(
            adata: ad.AnnData,
            deg_key: str = "rank_genes_groups",
            deg_df: pd.DataFrame | None = None,
            leiden_to_ref: dict[str, list[str]] | None = None,
            leiden_to_celltype: dict[str, str] | None = None,
            celltype_to_ref: dict[str, list[str]] | None = None,
            ref_df: pd.DataFrame | None = None,
            return_as_dict: bool = True
        ) -> tuple[dict[str, list[str]], dict[str, pd.DataFrame]] | dict[str, list[str]]:
    """
    Subsets differentially expressed genes (DEGs) based on provided marker
    gene information.


    NOTE:
        - Don't use this function for now, a general version is under
          developement.
        - if you want to subset the DEGs prior to this analysis, run the
          sc_utils.subset_degs.

    Args:
        adata: adata object.
        deg_key (str, optional): Key for the DEG results stored in
            ``adata.uns``. Defaults to "rank_genes_groups".
        deg_df: DataFrame containing DEG information. If None, extracted from
            adata.uns[deg_key].
        leiden_to_ref: Dictionary mapping Leiden cluster IDs to lists of marker
            genes.
        leiden_to_celltype: Dictionary mapping Leiden cluster IDs to cell type
            names.
        celltype_to_ref: Dictionary mapping cell type names to lists of marker
            genes.
        ref_df: DataFrame containing marker gene information.
        return_as_dict: If True, it returns one dataframe containing all the
            information.

    Returns:
        tuple[dict[str, list[str]], dict[str, pandas.DataFrame]] | dict[str, list[str]]:
            - leiden_to_deg_mapping or celltype_to_deg_mapping: Dictionary
              mapping cluster IDs
              (Leiden or cell type) to lists of DEG names.
            - leiden_to_deg_ref_df_mapping or celltype_to_deg_ref_df_mapping:
              (Optional) Dictionary mapping cluster IDs to DataFrames containing
              DEG information and marker gene information.

    Raises:
        ValueError: If both leiden_to_ref and celltype_to_ref are None.
        KeyError: If deg_key not in adata.uns.keys().

    TODO:
        - Generalize the usage of leiden_to_ref, leiden_to_celltype and
          celltype_to_ref
        - Find the notebook, where this was used.
    """
    if deg_df is None:
        if deg_key not in adata.uns.keys():
            raise KeyError(f"Key '{deg_key}' not found in adata.uns.")
        deg_df = adata.uns[deg_key]

    # Determine mapping based on provided information
    if leiden_to_ref is not None:
        cluster_to_ref = leiden_to_ref
        return_cluster = False
    elif leiden_to_celltype is not None and celltype_to_ref is not None:
        cluster_to_ref = {k: celltype_to_ref[v]
                          if v in celltype_to_ref.keys() else [] for k, v in leiden_to_celltype.items()}
        return_cluster = True
    else:
        raise ValueError(
            "Either leiden_to_ref or both leiden_to_celltype and celltype_to_ref must be provided.")

    # Initialize output dictionaries
    cluster_to_deg_mapping = {}
    cluster_to_deg_ref_df_mapping = {}

    # Iterate through clusters
    for cluster_id, ref in cluster_to_ref.items():
        cluster_degs = deg_df[((deg_df["group"] == cluster_id)
                               & (deg_df["names"].isin(ref)))]  # ["names"].tolist()
        cluster_to_deg_mapping[cluster_id] = cluster_degs

        if ref_df is not None and return_cluster:
            # NOTE: The ref_df has only a cell_type column, no cluster column
            # Subset ref_df and deg_df based on intersection
            ct = leiden_to_celltype[cluster_id]
            cluster_deg_ref_df = ref_df[
                        (ref_df["gene"].isin(cluster_degs["names"].values))
                        & (ref_df["cell_type"] == ct)]
            cluster_deg_df = cluster_degs[cluster_degs["names"].isin(cluster_deg_ref_df["gene"])]

            # Merge DataFrames for comprehensive information
            cluster_to_deg_ref_df_mapping[cluster_id] = pd.merge(
                cluster_deg_df, cluster_deg_ref_df, left_on="names", right_on="gene")
            cluster_to_deg_ref_df_mapping[cluster_id]["group"].map(leiden_to_celltype)

    if ref_df is not None:
        if return_as_dict:
            return cluster_to_deg_ref_df_mapping
        else:
            return pd.concat(
                [x for x in cluster_to_deg_ref_df_mapping.values()
                 if not x.empty])
    else:
        if return_as_dict:
            return cluster_to_deg_mapping
        else:
            return pd.concat(
                [x for x in cluster_to_deg_mapping.values()
                 if not x.empty])


def convert_ref_dict_to_gs_per_row(
            ref_dict: dict[str, list[str]]
        ) -> pd.DataFrame:
    """Converts a dictionary of marker genes to a GeneSet per row format.

    This function takes a dictionary containing marker genes and converts it
    into a DataFrame where each row corresponds to a GeneSet with the associated
    gene symbols concatenated into a comma-separated string. The function first
    converts the dictionary into a CSV format, melts the DataFrame to
    restructure it, removes empty and NaN values, and finally groups the data by
    'GeneSet', concatenating the gene symbols.

    NOTE:
        Ensure that the input dictionary is properly formatted with non-empty
        strings for gene symbols.

    Args:
        ref_dict (dict[str, list[str]]]): A dictionary where the keys are GeneSets and the
            values are lists of gene symbols.

    Returns:
        pandas.DataFrame:
            A DataFrame with two columns:

                - 'GeneSet': The name of the gene set.
                - 'geneSymbols': A comma-separated string of gene symbols.

    Raises:
        ValueError: If the input dictionary is empty.

    Calls:
        convert_ref_dict_to_df

    Called By:
        map_geneset_to_degs
    """
    # #########################################################
    # Convert the dictionary of marker genes to a CSV format DataFrame
    df = convert_ref_dict_to_df(ref_dict)
    # #########################################################
    # Melting the DataFrame to convert it into a long format
    melted_df = df.melt(var_name="GeneSet", value_name="geneSymbols")
    # #########################################################
    # Filtering out empty strings and NaN values from the 'geneSymbols' column
    # ##########################################
    # Remove rows where 'geneSymbols' is an empty string
    melted_df = melted_df[melted_df["geneSymbols"] != ""]
    # ##########################################
    # Drop rows with NaN values in the 'geneSymbols' column
    melted_df.dropna(subset=["geneSymbols"], inplace=True)
    # #########################################################
    # Group by 'GeneSet' and aggregate gene symbols into a comma-separated string
    result_df = melted_df.groupby("GeneSet")["geneSymbols"].agg(",".join).reset_index()
    # #########################################################
    # Return the final DataFrame
    return result_df


# ###################################################################################################
# Marker dict handling
def deduplicate_ref_dict(
            ref_dict: dict[str, list[str]],
            return_duplicates: bool = False
        ) -> dict[str, list[str]] | tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Deduplicates the marker gene dictionary by removing overlapping genes
    between different keys and deletes any keys that end up with an empty
    list after deduplication.

    The function takes a dictionary where each key corresponds to a marker gene
    set. It identifies and removes overlapping genes from these sets, ensuring
    that each gene is uniquely associated with one key. If a key ends up with an
    empty list after deduplication, that key is deleted from the dictionary.

    If return_duplicates is True, returns a second dictionary containing only
    the duplicate ref/markers that were removed from each key.

    NOTE:
        - The function only handles a dictionary with list values!
        - This is an upgrade of uniqueify_markers() which is removed.

    Args:
        ref_dict (dict[str, list[str]]): A dictionary where keys are categories
            and values are lists of marker genes.
        return_duplicates (bool, optional): If True, also return a dict of
            removed duplicate genes per key. Defaults to False.

    Returns:
        dict[str, list[str]] | tuple[dict[str, list[str]], dict[str, list[str]]]:
            Deduplicated dictionary, or a tuple with deduplicated and duplicates dict.

    Raises:
        None

    Called By:
        plot_mediods_heatmap
    """
    # ###############################
    # Identify genes that occur in more than one key
    # ###############################
    all_genes = [gene for genes in ref_dict.values() for gene in genes]
    duplicate_genes = {gene for gene, count in Counter(all_genes).items() if count > 1}
    # ###############################
    # Filter genes per key and optionally collect duplicates
    # ###############################
    ref_dict_non_overlap = {}
    duplicate_dict = {} if return_duplicates else None

    for k, genes in ref_dict.items():
        # Keep only unique genes (not found in multiple keys)
        genes_to_keep = [g for g in genes if g not in duplicate_genes]

        # Warn and skip keys that become empty after deduplication
        if genes_to_keep:
            ref_dict_non_overlap[k] = genes_to_keep
        else:
            logger.warning(f'Caution, we remove {k} from the dict, because it is empty after deduplication')

        # Optionally track which genes were removed per key
        if return_duplicates:
            duplicate_dict[k] = [g for g in genes if g in duplicate_genes]
    # ###############################
    # Return result depending on mode
    # ###############################
    if return_duplicates:
        return ref_dict_non_overlap, duplicate_dict
    return ref_dict_non_overlap


def process_and_save_degs(
            deg_df: pd.DataFrame,
            hallmark_df: pd.DataFrame,
            ref_dict: dict[str, dict[str, list[str]]] | None,
            save_path: str,
            config: dict[str, float],
            direction: str,
            filtered: bool,
            suffix: str,
        ) -> None:
    """
    Subsets DEGs, maps hallmark and geneset annotations, and saves them to CSV
    and Excel.

    Args:
        deg_df: DataFrame containing ranked DEGs.
        hallmark_df: DataFrame with hallmark gene sets (row-wise).
        ref_dict: Optional dictionary of additional gene sets to map, keyed by
            category name.
        save_path: Output path prefix (no file extension).
        config: Dictionary with filtering parameters (e.g., n_genes, pct_min,
            etc.).
        direction: One of 'up', 'down', or 'up_n_down'.
        filtered: If True, apply filtering criteria from config; else return all
            genes.
        suffix: Output filename suffix (appended to save_path).

    Returns:
        None

    Calls:
        map_geneset_to_degs, save_dataframe, subset_degs

    Called By:
        get_DEG_gene_csvs

    Tags:
        DEG, annotation, config, io
    """
    # #############################################
    # Save unfiltered or filtered marker genes for all clusters
    if filtered:
        config_params = config["to_create"]["down"]["DEG_csv"]
        temp_df = subset_degs(
            deg_df,
            groups=None,
            n_genes=config_params["n_genes"],
            perc=config_params["pct_min"],
            p_val_cutoff=config_params["pval_cutoff"],
            lfc=config_params["log2fc_min"],
            direction=direction)
    else:
        temp_df = subset_degs(
            deg_df,
            groups=None,
            n_genes=1e300,
            perc=0.,
            p_val_cutoff=1.,
            lfc=0.,
            direction=direction)
    # ################
    # Merge the hallmarks and the provided geneset supersets in the DEG
    temp_df = map_geneset_to_degs(temp_df, geneset_df_row_wise=hallmark_df, key_to_add="hallmarks")

    if ref_dict is not None:
        for geneset_name, genesets in ref_dict.items():
            temp_df = map_geneset_to_degs(temp_df, geneset_dict=genesets, key_to_add=geneset_name)
    # ################
    # Save the DEGs
    save_dataframe(
        temp_df,
        [f"{save_path}{suffix}{ft}"
            for ft in config["general"]["save_df_types"]],
        index=False)


def get_DEG_gene_csvs(
            adata: ad.AnnData,
            deg_df: pd.DataFrame | None = None,
            config: dict | None = None,
            groups: list[str] | None = None,
            key: str = "rank_genes_groups",
            deg_path: str | None = None,
        ) -> None:
    """
    Generate and save CSV files containing marker genes for each group/cluster
    in the given adata object.

    This function is designed to add the marker/hallmarg genes to the DEG CSV
    files and save unfiltered and filtered versions of the DEG with marker genes
    in the config['general']['save_path']}DEGs/ directory.

    NOTE:
        Ensure that the ``config`` dictionary is correctly formatted with all
        necessary keys, particularly under ``to_create.down.DEG_csv``,
        for this function to execute properly.

    Args:
        adata (anndata.AnnData): Adata object from which to extract marker
            genes. deg_df ()
        config (dict | None, optional): Configuration dictionary containing
            paths and filtering parameters. If not provided, the function will
            use ``adata.uns["config"]``. Defaults to None.
        groups (list[str] | None, optional): List of groups to generate marker
            gene CSVs for. If not provided, the function will use the default
            grouping from ``adata.uns[key]["params"]["groups"]``. Defaults to
            None.
        key (str, optional): The key under which the differential expression
            results are stored in ``adata.uns``. Default is "rank_genes_groups".
            Defaults to "rank_genes_groups".
        deg_path (str, optional): If None, is uses the
            f"{config['general']['save_path']}DEGs/". Defaults to None.

    Returns:
        None

    Raises:
        KeyError: If necessary keys are missing in the ``adata`` object or the
            ``config`` dictionary.
        FileNotFoundError: If the specified save path does not exist and cannot
            be created.

    TODO:
        Change the key to deg_key

    Calls:
        create_deg_table, get_deg_df, get_msigdb_df, process_and_save_degs

    Called By:
        run_downstream

    Tags:
        DEG, annotation, config, io
    """
    # #########################################################
    # Initialize the configuration if not provided
    if config is None:
        config = adata.uns["config"]

    if deg_df is None:
        deg_df = get_deg_df(adata, deg_key=key)

    if "genesets" in adata.uns.keys():
        genesets = adata.uns["genesets"]
    else:
        genesets = None
    # #########################################################
    # Create the directory for saving the CSVs if it doesn't exist
    if deg_path is None:
        deg_path = f"{config['general']['save_path']}DEGs/"
    if not os.path.exists(deg_path):
        os.makedirs(deg_path)
    # #########################################################
    # Determine the groups to use if not provided
    if groups is None:
        # groups = adata.uns[key]["params"]["groups"]
        groups = deg_df["group"].unique()
    # #########################################################
    # Get the Hallmarks from MsigDb
    hallmark_df = get_msigdb_df(
            organism=adata.uns["config"]["general"]["organism"], only_hallmarks=True)
    # #########################################################
    # Save marker gene CSVs for all groups, both filtered and unfiltered
    if "rna" in config["to_create"]["down"]["DEG_csv"]["marker_mod"]:
        for direction in ["up_n_down", "up", "down"]:
            process_and_save_degs(
                deg_df, hallmark_df, genesets,
                deg_path, config,
                direction, filtered=False, suffix=f"all_{direction}_unfiltered")
            process_and_save_degs(
                deg_df, hallmark_df, genesets,
                deg_path, config,
                direction, filtered=True, suffix=f"all_{direction}_filtered")
    # #########################################################
    # Optionally save marker gene CSVs for each individual group
    if config["to_create"]["down"]["DEG_csv"]["one_csv_per_cluster"]:
        for group in groups:
            if "rna" in config["to_create"]["down"]["DEG_csv"]["marker_mod"]:
                create_deg_table(
                    adata, deg_path, config,
                    group, name="rna", key=key, deg_df=deg_df)


def save_genesets_to_csv(
            adata: ad.AnnData,
            config: dict | None = None,
            uns_key: str = "genesets",
            genesets: dict | None = None
        ) -> None:
    """
    Save the specified dictionary from ``adata.uns`` or directly provided genesets
    into CSV files in the 'markers' directory.

    The function handles cases where the inner values are lists of different
    lengths by converting them into DataFrames and saving them as CSV files. If
    ``genesets`` is provided, it overrides the ``uns_key`` argument, and the data is
    saved directly from the ``genesets`` dictionary.

    NOTE:
        Ensure that the 'markers' directory exists or can be created before
        saving the files.

    Args:
        adata (anndata.AnnData): Adata object containing the dictionary in
            ``adata.uns``.
        config (dict | None, optional): Configuration dictionary containing
            paths. If not provided, the function will use ``adata.uns['config']``.
            Defaults to None.
        uns_key (str, optional): The key in ``adata.uns`` whose dictionary will be
            saved as CSV files. Default is "genesets". Defaults to "genesets".
        genesets (dict | None, optional): A dictionary containing genesets to be
            saved directly as CSV files. If provided, this will override the use
            of ``uns_key``. Defaults to None.

    Returns:
        None

    Raises:
        KeyError: If the specified ``uns_key`` is not found in ``adata.uns`` and
            ``genesets`` is not provided.
        FileNotFoundError: If the specified save path does not exist and cannot
            be created.

    Calls:
        StatKeeper.get

    Called By:
        run_downstream

    TODO:
        Consider implementing additional error handling for cases where the data
        in ``genesets`` or ``adata.uns[uns_key]`` is not in the expected format.

    Tags:
        annotation, config, io
    """
    # #########################################################
    # Validate and retrieve the genesets
    if genesets is None:
        # ##########################################
        # If genesets is not provided, use the dictionary from adata.uns
        if uns_key not in adata.uns:
            raise KeyError(f"The key '{uns_key}' is not found in adata.uns")
        genesets = adata.uns[uns_key]
    # #########################################################
    # Initialize the configuration if not provided
    if config is None:
        config = adata.uns.get("config", {})

    # Set the save path
    save_path = os.path.join(config.get('general', {}).get('save_path', './'), "markers/")

    # Create the directory for saving the CSVs if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # #########################################################
    # Save each dictionary entry in genesets as a CSV
    for key, geneset in genesets.items():
        # ##########################################
        # Convert the dictionary of lists to a DataFrame, filling missing values with NaN
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in geneset.items()]))
        df.to_csv(f'{save_path}{key}.csv', index=False)
        df.to_excel(f'{save_path}{key}.xlsx', index=False)
    # #########################################################
    # Confirm that the files have been saved
    logger.info(f"All entries under '{uns_key}' have been saved to {save_path}")


def get_valide_ref_dicts(
            adata: ad.AnnData,
            ref_dict: dict[str, list[str]],
            return_subset: bool = False
        ) -> tuple[dict[str, list[str]], dict[str, list[str]] | None]:
    """Subset the marker dictionary to genes present in the adata.

    This function curates the input marker dictionary by intersecting the gene
    lists with the genes present in the provided adata object. It also allows
    for an optional subsetting based on a configuration stored within the adata
    object.

    NOTE:
        - This will replace all non ascii chars in the keys! If you don't want
          that behaviour, please replace all the non-ascii chars yourself.
        - Ensure that the 'config' key exists in the 'uns' attribute of the
          adata object before using this function. The function does not
          handle the absence of this key, which will result in a KeyError.

    Args:
        adata (anndata.AnnData): Adata object. The object should
            have a 'config' key in its 'uns' attribute with information about
            which markers to subset.
        ref_dict (dict[str, list[str]]): Marker dictionary where keys are names and values
            are lists of gene names.
        return_subset (bool, optional): If True, returns an additional
            dictionary subset according to the 'marker_subset' configuration.
            Default is False. Defaults to False.

    Returns:
        tuple[dict[str, list[str]], dict[str, list[str]] | None]:
            - dict, dict: Curated marker dictionary with only genes present in
              the adata object and subset.
            - dict or None: Subset of the curated marker dictionary if
              return_subset is True; otherwise, None.

    Raises:
        KeyError: If the 'config' key is not found in the 'uns' attribute of the
            adata object.

    Calls:
        replace_special_chars

    Called By:
        convert_ref_dict_to_df, downstream_preprocessing,
        ref_dict_to_valide_data_frame, score_genes, score_genes_parallel
    """
    # #########################################################
    # Retrieve configuration from the adata object
    config = adata.uns["config"]
    # #########################################################
    # Clean the dict keys, biologists like to use non ascii chars -.-
    ref_dict = replace_special_chars(ref_dict)
    # #########################################################
    # Intersect each gene list in ref_dict with genes in adata
    ref_dict_upd = {}

    for key, marker_list in ref_dict.items():
        intersect = np.intersect1d(marker_list, adata.var_names).tolist()
        if intersect:
            ref_dict_upd[key] = intersect
    # #########################################################
    # If there's a subset defined in the config, apply additional subsetting
    if len(config["to_plot"]["down"]["marker_subset"]) > 0:
        ref_dict_upd_subset = {}
        for k, v in ref_dict_upd.items():
            ref_dict_upd_subset[k] = np.intersect1d(v, config["to_plot"]["down"]["marker_subset"]).tolist()
        # Remove any keys from the subset dictionary that have empty lists
        ref_dict_upd_subset = {k: v for k, v in ref_dict_upd_subset.items() if v}

        # If return_subset is True, return both the full and subset dictionaries
        if return_subset:
            return ref_dict_upd, ref_dict_upd_subset
    # #########################################################
    # Return the curated dictionary and None if subset was not requested
    return ref_dict_upd, None


def data_frame_to_valide_ref_dict(
            df: pd.DataFrame,
            adata: ad.AnnData | None = None,
            fix_empyt_string: bool = True
        ) -> dict[str, list[str]]:
    """Extract a dataframe to a marker dictionary.

    This function filters and maps gene sets from a DataFrame to an adata
    object. Specifically, it extracts gene sets that are present in the adata
    object, returning them as a curated dictionary. The function is particularly
    useful for preparing marker gene sets for downstream analysis.

    NOTE:
        This function assumes that the DataFrame contains gene sets as columns,
        with each cell holding gene names. Genes not found in the adata object
        will be excluded.

    Args:
        df (pandas.DataFrame): DataFrame where each column represents a gene
            set. The function will curate this DataFrame by including only those
            genes that are found in the adata object.
        adata (anndata.AnnData | None, optional): adata object containing the
            gene expression data. The function will check for the presence of
            genes from the DataFrame in this object's variables. Defaults to
            None.

    Returns:
        dict[str, list[str]]:
            A dictionary where each key is a gene set name (from the
            DataFrame's columns), and each value is an array of genes that are
            present both in the gene set and in the adata object.

    Called By:
        map_geneset_to_degs

    TODO:
        Consider adding error handling for cases where adata or df are not of
        the expected types.
    """
    if fix_empyt_string:
        df.loc[:, df.dtypes == object] = df.select_dtypes(include=[object]).replace('', np.nan)
        # df = df.select_dtypes(include=[object]) = df.select_dtypes(include=[object]).replace('', np.NaN)
    # #########################################################
    # Initialize an empty dictionary to store the curated gene sets
    genesets = {}
    # #########################################################
    # Iterate over each column (gene set) in the DataFrame
    for k in df.columns:
        # ##########################################
        # Extract the genes that are present in both the current gene set and the adata object
        if adata is not None:
            genes = np.intersect1d(df[k].dropna(), adata.var_names)
        else:
            genes = df[k].dropna().values
        # ##########################################
        # If the intersection is not empty, add it to the dictionary
        if len(genes) != 0:
            genesets[k] = genes
    # #########################################################
    # Return the curated dictionary of gene sets
    return genesets


def ref_dict_to_valide_data_frame(
            ref_dict: dict[str, list[str]],
            adata: ad.AnnData | None = None,
            fill_value: Any = np.nan,
            save_path: str | None = None
        ) -> pd.DataFrame:
    """Convert a dictionary of gene sets into a DataFrame.

    This function takes a dictionary where keys are gene set names and values
    are arrays of genes, and transforms it into a DataFrame where each column
    represents a gene set. If ``save_path`` is provided, the DataFrame is saved as
    a the file suffix and returned.

    NOTE:
        The function assumes that the gene sets in the dictionary are
        well-formed (i.e., each value is an array of genes). Gene sets with no
        genes will result in empty columns.

    Args:
        ref_dict (dict[str): Dictionary mapping gene set names to lists of gene
            symbols.
        adata (anndata.AnnData | None, optional): If provided, filters each gene
            list to include only genes present in ``adata.var_names``. Defaults to
            None.
        fill_value (Any, optional): Value used to pad shorter gene lists. nan.
            Defaults to np.nan.
        save_path (str | None, optional): If not None, saves the resulting
            DataFrame to the given path as the file suffix. Defaults to None.
        Defaults to None.

    Returns:
        pandas.DataFrame:
            A DataFrame where each column represents a gene set,
            with genes listed in the cells.

    Calls:
        get_valide_ref_dicts

    TODO:
        Consider handling variable gene set lengths by filling shorter columns
        with NaN where necessary.
    """
    if adata is not None:
        ref_dict, _ = get_valide_ref_dicts(adata, ref_dict)

    # Create a DataFrame from the dictionary, handling different lengths of gene arrays by filling with NaN
    max_len = max((len(v) for v in ref_dict.values()), default=0)
    df = pd.DataFrame({k: v + [fill_value] * (max_len - len(v)) for k, v in ref_dict.items()})

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        return None

    return df


def convert_ref_dict_to_df(
            ref_dict: dict[str, list[str]],
            path: str | None = None,
            overwrite: bool = False,
            pad_value: str | None = None,
            adata: ad.AnnData | None = None
        ) -> pd.DataFrame | None:
    """
    Convert a dictionary of gene sets into a DataFrame and optionally save it
    as a CSV file.

    This function can preprocess the dictionary if 'adata' is not None, handle
    different lengths of gene lists, and save the result to a CSV file if a path
    is provided.

    Args:
        ref_dict (dict[str, list[str]]): A dictionary where each key is a gene set name and
            each value is an array of genes.
        path (str | None, optional): The file path to save the DataFrame as a
            CSV. If empty, the DataFrame is returned. Defaults to None.
        overwrite (bool, optional): If True, overwrites the existing file at the
            specified path. If False, and the file exists, a warning is logged,
            and the function returns without saving. Defaults to False.
        pad_value (str | None, optional): The value used to pad the lists of
            genes in the DataFrame. Defaults to None.
        adata: Additional data structure for preprocessing (optional).

    Returns:
        pandas.DataFrame | None:
            The DataFrame constructed from the gene dictionary, or
            None if the DataFrame is saved.

    Raises:
        FileExistsError: If the path exists and overwrite is False.

    Calls:
        get_valide_ref_dicts

    Called By:
        convert_ref_dict_to_gs_per_row
    """
    if adata is not None:
        ref_dict, _ = get_valide_ref_dicts(adata, ref_dict)

    # Handling different lengths of gene arrays by filling with pad_value or NaN
    if pad_value is not None:
        max_length = max((len(v) for v in ref_dict.values()), default=0)
        df = pd.DataFrame({k: v + [pad_value] * (max_length - len(v)) for k, v in ref_dict.items()})
    else:
        df = pd.DataFrame({k: pd.Series(v) for k, v in ref_dict.items()})

    # Save to CSV if path is provided
    if path is not None:
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"File '{path}' already exists. Set overwrite=True to overwrite it.")
        df.to_csv(path, index=False)
    else:
        return df


def get_all_markers_from_ref_dict(
            ref_dict: dict[str, list[str]],
            unique: bool = False
        ) -> list[str]:
    """Extract all gene markers from a given marker dictionary.

    This function iterates over the values of a given marker dictionary,
    where each key represents a name and the corresponding value is a list of
    gene markers. The function can optionally return only unique gene markers.

    NOTE:
        Ensure that the input dictionary is properly formatted with keys as
        marker names and values as lists of gene markers.

    Args:
        ref_dict (dict[str, list[str]]): A dictionary where keys are marker names and values
            are lists of gene markers.
        unique (bool, optional): If True, the function will return a list of
            unique gene markers. Defaults to False.

    Returns:
        list[str]:
            A list of gene markers extracted from the dictionary.
            If ``unique`` is set to True, the list contains only unique gene
            markers.

    Called By:
        downstream_preprocessing, get_random_geneset_reference,
        plot_mediods_heatmap, score_genes, score_genes_parallel

    TODO:
        Optimize the function for handling large dictionaries efficiently.
    """
    # #########################################################
    # Extract the values from the input dictionary.
    values = [v for v in ref_dict.values()]
    # #########################################################
    # Flatten the list of lists into a single list of gene markers.
    flat_values = []
    for v in values:
        flat_values.extend(v)
    # #########################################################
    # Return unique gene markers if specified, otherwise return all markers.
    if unique:
        # ##########################################
        # Return unique gene markers using pandas' unique function.
        return pd_unique(flat_values).tolist()
    else:
        # ##########################################
        # Return all gene markers without filtering for uniqueness.
        return flat_values


def create_deg_table_parallel(
            adata: ad.AnnData,
            save_path: str,
            config: dict,
            name: str = "",
            overwrite: bool = True,
            key: str = "rank_genes_groups"
        ) -> None:
    """
    Creates a dataframe for the DEGs (Differentially Expressed Genes) per cluster
    and saves it as a CSV file.

    This function processes the results of differential gene expression analysis
    stored in the ``uns`` attribute of the adata object (``adata``). It generates a
    dataframe that contains the names, scores, log fold changes, adjusted
    p-values, and percentages for both the target group and the rest for each
    gene in every cluster. The dataframe is then saved as a CSV file at the
    specified path.

    NOTE:
        The default behavior of this function is to overwrite any existing
        file at the save path. If you do not want this behavior, use the
        ``overwrite=False`` option.

    Args:
        adata (anndata.AnnData): Adata object the single-cell RNA sequencing
            data and analysis results.
        save_path (str): Path where the resulting CSV file will be saved.
        config (dict): Configuration dictionary that specifies the parameters
            for creating the CSV file.
        name (str, optional): A string to append to the filename of the saved
            CSV. Defaults to "".
        overwrite (bool, optional): Whether to overwrite an existing file at the
            save path. Defaults to True.
        key (str, optional): The key in the ``uns`` attribute of ``adata`` where the
            differential expression results are stored. Defaults to
            "rank_genes_groups".

    Returns:
        None

    TODO:
        REWRITE, the percentage doesn't work
    """
    # #########################################################
    # Check if the output file already exists and handle the overwrite option
    path_deg_per_cluster = f"{save_path}DEGS_parallel{name}.csv"

    if os.path.exists(path_deg_per_cluster) and overwrite:
        # ##########################################
        # Warn the user about overwriting and remove the existing file
        logger.warning("NOTE: The default ranked create_deg_table_parallel overwrites the files."
                       "If you don't want this behaviour, use the overwrite=False option")
        os.remove(path_deg_per_cluster)
    # #########################################################
    # Extract differential gene expression results from the adata object
    result = adata.uns[key]
    groups = result['names'].dtype.names
    # #########################################################
    # Create a dataframe containing relevant data for each cluster
    df = pd.DataFrame(
        {group + '_' + key: result[key][group]
            for group in groups for key in ['names', "scores", "logfoldchanges", "pvals_adj", 'pts', "pts_rest"]})

    # Reset index and save the top N genes to CSV based on the config
    df.reset_index(drop=True)
    df.head(config["to_create"]["down"]["DEG_csv"]["parallel_n_genes"]).to_csv(path_deg_per_cluster)


def create_deg_table(
            adata: ad.AnnData,
            save_path: str,
            config: dict,
            cluster: str | list[str],
            name: str = "",
            deg_df: pd.DataFrame | None = None,
            overwrite: bool = True,
            key: str = "rank_genes_groups",
            skip_warn: bool = True,
        ) -> None:
    """
    Creates CSV files containing ranked genes for the specified clusters based
    on differential expression analysis results stored in an adata object.

    This function generates CSV files for both upregulated and downregulated
    genes, as well as a combined list of both, saving them to the specified
    directory. The directory structure and filenames vary depending on whether
    the input ``cluster`` is a single value or a list. If ``overwrite`` is set to
    True, any existing files with the same name will be replaced.

    NOTE:
        The default behavior of this function is to overwrite existing files.
        If this is not desired, set the ``overwrite`` parameter to False.

    Args:
        adata (anndata.AnnData): Adata object containing DEGs.
        save_path (str): The path where the CSV files should be saved.
        config (dict): Configuration dictionary containing parameters for
            filtering genes based on criteria like the number of genes, minimum
            percentage, p-value cutoff, and log fold change.
        cluster (str | list[str]): The cluster(s) for which to create the ranked
            genes CSV files. If a single cluster is provided, the files are
            saved in a 'per_cluster' directory. If a list of clusters is
            provided, the files are saved in a 'DEGS' directory.
        name (str, optional): Additional string to append to the filename.
            Defaults to an empty string.
        overwrite (bool, optional): If True, existing files with the same name
            will be overwritten. Defaults to True.
        deg_df (pandas.DataFrame | None, optional):
            Optionally precomputed DEG dataframe to reuse. If None, uses
            adata.uns["rank_genes_groups"]. Defaults to None.
        key (str, optional): The key used to retrieve the differential
            expression results from the adata object. Defaults to
            "rank_genes_groups".
        skip_warn (bool, optional): If True, skip the overwrite=False warning.
            Defaults to True.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified directory for saving the files does
            not exist and cannot be created.
        ValueError: If the cluster is not a string or a list of strings.

    Calls:
        get_deg_df, subset_degs

    Called By:
        get_DEG_gene_csvs
    """
    # #########################################################
    # Retrieve configuration from adata if not provided
    if config is None:
        config = adata.uns["config"]
    if deg_df is None:
        deg_df = get_deg_df(adata)
    # #########################################################
    # Determine file paths based on whether cluster is a single value or a list
    if isinstance(cluster, list):
        path_deg_per_cluster = f'{save_path}DEGS/{name}.csv'
        path_deg_per_cluster_up = f'{save_path}DEGS/up/{name}.csv'
        path_deg_per_cluster_down = f'{save_path}DEGS/down/{name}.csv'
    else:
        path_deg_per_cluster = f'{save_path}per_cluster/cluster_{cluster}_{name}.csv'
        path_deg_per_cluster_up = f'{save_path}per_cluster/up/cluster_{cluster}_{name}.csv'
        path_deg_per_cluster_down = f'{save_path}per_cluster/down/cluster_{cluster}_{name}.csv'
    # #########################################################
    # Ensure that the required directories exist, create if necessary
    path_deg_per_cluster_folder = os.path.dirname(path_deg_per_cluster)
    if not os.path.exists(path_deg_per_cluster_folder):
        os.makedirs(path_deg_per_cluster_folder)

    path_deg_per_cluster_up_folder = os.path.dirname(path_deg_per_cluster_up)
    if not os.path.exists(path_deg_per_cluster_up_folder):
        os.makedirs(path_deg_per_cluster_up_folder)

    path_deg_per_cluster_down_folder = os.path.dirname(path_deg_per_cluster_down)
    if not os.path.exists(path_deg_per_cluster_down_folder):
        os.makedirs(path_deg_per_cluster_down_folder)
    # #########################################################
    # Remove existing files if overwrite is enabled
    if os.path.exists(path_deg_per_cluster) and overwrite:
        if not skip_warn:
            logger.warning("NOTE: The default ranked create_deg_table overwrites the files. "
                           "If you don't want this behaviour, use the overwrite=False option")
        os.remove(path_deg_per_cluster)
    if os.path.exists(path_deg_per_cluster_up) and overwrite:
        if not skip_warn:
            logger.warning("NOTE: The default ranked create_deg_table overwrites the files. "
                           "If you don't want this behaviour, use the overwrite=False option")
        os.remove(path_deg_per_cluster_up)
    if os.path.exists(path_deg_per_cluster_down) and overwrite:
        if not skip_warn:
            logger.warning("NOTE: The default ranked create_deg_table overwrites the files. "
                           "If you don't want this behaviour, use the overwrite=False option")
        os.remove(path_deg_per_cluster_down)
    # #########################################################
    # Generate and save the ranked gene groups for the specified cluster(s)
    # TODO: Speed improvements: Subset all beforehand for qc and afterwards to cluster
    ddf = subset_degs(
        deg_df,
        groups=cluster,
        n_genes=config["to_create"]["down"]["DEG_csv"]["n_genes"],
        perc=config["to_create"]["down"]["DEG_csv"]["pct_min"],
        p_val_cutoff=config["to_create"]["down"]["DEG_csv"]["pval_cutoff"],
        lfc=config["to_create"]["down"]["DEG_csv"]["log2fc_min"],
        direction="up_n_down")
    ddf.to_csv(path_deg_per_cluster, index=False)
    ddf.to_excel(sub(".csv", ".xlsx", path_deg_per_cluster), index=False)

    ddf = subset_degs(
        deg_df,
        groups=cluster,
        n_genes=config["to_create"]["down"]["DEG_csv"]["n_genes"],
        perc=config["to_create"]["down"]["DEG_csv"]["pct_min"],
        p_val_cutoff=config["to_create"]["down"]["DEG_csv"]["pval_cutoff"],
        lfc=config["to_create"]["down"]["DEG_csv"]["log2fc_min"],
        direction="up")
    ddf.to_csv(path_deg_per_cluster_up, index=False)
    ddf.to_excel(sub(".csv", ".xlsx", path_deg_per_cluster_up), index=False)

    ddf = subset_degs(
        deg_df,
        groups=cluster,
        n_genes=config["to_create"]["down"]["DEG_csv"]["n_genes"],
        perc=config["to_create"]["down"]["DEG_csv"]["pct_min"],
        p_val_cutoff=config["to_create"]["down"]["DEG_csv"]["pval_cutoff"],
        lfc=config["to_create"]["down"]["DEG_csv"]["log2fc_min"],
        direction="down")
    ddf.to_csv(path_deg_per_cluster_down, index=False)
    ddf.to_excel(sub(".csv", ".xlsx", path_deg_per_cluster_down), index=False)


def ref_dict_long_value_split(
            ref_dict: dict[str, list[str]],
            split_on: int = 100
        ) -> dict[str, list[str]]:
    """
    Splits dictionary values into smaller chunks if they exceed a specified length.

    This function is designed to handle scenarios where the values in a
    dictionary are too large to be processed or displayed effectively.
    Specifically, it addresses cases where a plot cannot handle more than a
    specified number of genes (default is 350), so it divides the values into
    smaller lists and assigns them to new keys in the output dictionary.

    NOTE:
        The plot sc.pl.dotplot and sc.pl.stacked_violins can't handle more
        than 350 genes, use the first 350 for now.

    Args:
        ref_dict (dict[str): The original dictionary containing keys and lists
            of genes as values.
        split_on (int, optional): The maximum number of items allowed per value
            in the output dictionary. Default is 350. Defaults to 100.

    Returns:
        dict[str, list[str]]:
            A new dictionary where values from the original dictionary are
            split into smaller chunks, each assigned to a new key.

    Called By:
        plot_ref_dotplots

    TODO:
        Consider parameterizing the split_on value to make the function more
        flexible for different use cases.
    """
    # #########################################################
    # Initialize the output dictionary for storing the split values
    ref_dict_split = {}
    # #########################################################
    # Iterate over the items in the original dictionary
    for k, v in ref_dict.items():
        # ##########################################
        # Split the list in the dictionary value into chunks of size 'split_on'
        for i in range(0, len(v), split_on):
            # Create a new key indicating the chunk number
            new_key = f'{k}_{i // split_on + 1}'
            # Assign the chunk to the new key in the output dictionary
            ref_dict_split[new_key] = v[i:i + split_on]
    # #########################################################
    return ref_dict_split


def ref_dict_sort_values_hvg_like(
            adata: ad.AnnData,
            genesets: dict[str, list[str]] | None = None,
            geneset: list[str] | None = None,
            hvg_key: str = "highly_variable_rank"
        ) -> dict[str, list[str]] | list[str]:
    """
    Sorts the genes within a geneset or multiple genesets in the 'genesets'
    dictionary based on their highly variable rank in the 'adata' object.

    Args:
        adata (anndata.AnnData): Adata object from which the highly variable
            gene ranks are obtained.
        genesets (dict[str, optional): A dictionary where keys are geneset names
            and values are lists of genes to be sorted according to their highly
            variable rank.
        geneset (list[str] | None, optional): A single list of genes to be
            sorted according to their highly variable rank. Defaults to None.
        hvg_key (str, optional): The highly variable ranks. Default is
            'highly_variable_rank'. Defaults to "highly_variable_rank".

    Returns:
        dict[str, list[str]] | list[str]:
            If ``genesets`` is provided, returns a dictionary with sorted gene lists.
            If ``geneset`` is provided, returns a sorted list of genes.

    Called By:
        downstream_preprocessing
    """
    if genesets is None and geneset is None:
        raise ValueError("Either 'genesets' (dict) or 'geneset' (list) must be provided.")

    if genesets is not None:
        sorted_genesets = {
            name: sorted(genes, key=lambda gene: adata.var.loc[gene, hvg_key], reverse=True)
            for name, genes in genesets.items()}
        return sorted_genesets

    if geneset is not None:
        return sorted(geneset, key=lambda gene: adata.var.loc[gene, hvg_key], reverse=True)


def get_ref_gensests(
            adata: ad.AnnData,
            database: pd.DataFrame,
            reference_tissue: str | list[str],
            control_tissue: str | list[str] | None,
            n_genesets: int = 4,
            to_return: list[str] = ["all_in_one"],
            min_geneset_len: int = 3,
            rnd_gen: int | np.random.Generator | None = None,
            remove_overlapping_gs: bool = True
        ) -> list[dict[str, list[str]]]:
    """
    Generate marker genesets from a reference tissue with optional random controls.

    Args:
        adata (anndata.AnnData): Adata object.
        database (pandas.DataFrame): Marker gene database containing tissue,
            cell type, and genesymbol.
        reference_tissue (str | list[str]): Tissue(s) for which to create marker
            gene sets.
        control_tissue (str | list[str] | None): Optional tissue(s) used for
            background comparison.
        n_genesets (int, optional): Number of random genesets to generate. Use
            -1 to select all. Defaults to 4.
        to_return (list[str], optional): Determines returned structure.
            Defaults to ["all_in_one"].
        min_geneset_len (int, optional): Minimum number of genes required per
            geneset. Defaults to 3.
        rnd_gen (int | np.random.Generator | None, optional): Random generator
            or seed. Defaults to None.
        remove_overlapping_gs (bool, optional): Whether to remove genesets
            overlapping across groups. Defaults to True.

    Returns:
        list[dict[str, list[str]]]:
            List of generated marker and control geneset dictionaries.

    Calls:
        get_random_generator, get_random_geneset_reference,
        replace_special_chars
    """
    # ######################################
    # Create the random generator
    if rnd_gen is None:
        rnd_gen = get_random_generator(adata)
    elif isinstance(rnd_gen, int):
        rnd_gen = get_random_generator(adata, seed=rnd_gen)
    # ######################################
    # to_return options = ["all" ["all", "separate", "dict"]
    if isinstance(reference_tissue, str):
        reference_tissue = [reference_tissue]

    if isinstance(control_tissue, str):
        control_tissue = [control_tissue]
    # #####################################################################
    reference_cell_types = {}
    for rt in reference_tissue:
        reference_cell_types.update(
            {f'{db}_{rt}_{k}': v if isinstance(v, list) else v.split(",")
             for k, v, db in database[database["tissue"].isin([rt])][["cell_type", "genesymbol", "db"]].values})
    # Update the genesets based on the adata
    for rt, v in reference_cell_types.items():
        reference_cell_types[rt] = np.intersect1d(adata.var_names, v).tolist()
    # remove genesets, that are too small:
    rt_vals = copy(list(reference_cell_types.keys()))
    for rt in rt_vals:
        if len(reference_cell_types[rt]) < min_geneset_len:
            del reference_cell_types[rt]

    reference_cell_types = replace_special_chars(reference_cell_types)
    # #####################################################################
    control_cell_types = {}
    if control_tissue is not None:
        for ct in control_tissue:
            control_cell_types.update(
                {f'{db}_{ct}_{k}': v if isinstance(v, list) else v.split(",")
                 for k, v, db in database[database["tissue"].isin([ct])][["cell_type", "genesymbol", "db"]].values})
        # Update the genesets based on the adata
        for ct, v in control_cell_types.items():
            control_cell_types[ct] = np.intersect1d(adata.var_names, v).tolist()
        # remove genesets, that are too small:
        ct_vals = copy(list(control_cell_types.keys()))
        for ct in ct_vals:
            if len(control_cell_types[ct]) < min_geneset_len:
                del control_cell_types[ct]

    control_cell_types = replace_special_chars(control_cell_types)
    # #####################################################################
    if n_genesets == -1:
        n_genesets = len(reference_cell_types)

    if n_genesets == 0:
        rnd_genesets_high = []
        rnd_genesets_low = []
        rnd_genesets_all = []
        rnd_genesets_ref = []
    else:
        # #####################################################################
        # Get ranodm genesets
        rnd_genesets_high = get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="same", highly_variable=True,
                inplace=False, rnd_key="rnd_high_same", rnd_gen=rnd_gen)
        rnd_genesets_high.update(get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="uniform", highly_variable=True,
                inplace=False, rnd_key="rnd_high_uniform", rnd_gen=rnd_gen))
        rnd_genesets_low = get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="same", highly_variable=False,
                exclude_highly_variable=True,
                inplace=False, rnd_key="rnd_low_same", rnd_gen=rnd_gen)
        rnd_genesets_low.update(get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="uniform", highly_variable=False,
                exclude_highly_variable=True,
                inplace=False, rnd_key="rnd_low_uniform", rnd_gen=rnd_gen))
        rnd_genesets_all = get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="same", highly_variable=False,
                inplace=False, rnd_key="rnd_all_same", rnd_gen=rnd_gen)
        rnd_genesets_all.update(get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="uniform", highly_variable=False,
                inplace=False, rnd_key="rnd_all_uniform", rnd_gen=rnd_gen))
        rnd_genesets_ref = get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="same", use_only_geneset=True,
                highly_variable=False,
                inplace=False, rnd_key="rnd_ref_same", rnd_gen=rnd_gen)
        rnd_genesets_ref.update(get_random_geneset_reference(
                adata, reference_cell_types, n_genesets=n_genesets,
                dist_sample="uniform", use_only_geneset=True,
                highly_variable=False,
                inplace=False, rnd_key="rnd_ref_uniform", rnd_gen=rnd_gen))

    returnings = []
    if "all_in_one" in to_return:
        random_genes_cell_types = rnd_genesets_high
        random_genes_cell_types.update(rnd_genesets_low)
        random_genes_cell_types.update(rnd_genesets_all)
        random_genes_cell_types.update(rnd_genesets_ref)
        # ##################################
        # Combine them into one for scoring
        all_g = copy(reference_cell_types)
        all_g.update(control_cell_types)
        all_g.update(random_genes_cell_types)
        returnings.append(all_g)
    elif "separate" in to_return:
        returnings.extend([
            reference_cell_types, control_cell_types, rnd_genesets_high,
            rnd_genesets_low, rnd_genesets_all, rnd_genesets_ref, random_genes_cell_types])

    elif "dict" in to_return:
        dict_ = {
            "ref": reference_cell_types, "ctrl": control_cell_types,
            "rnd_high": rnd_genesets_high, "rnd_low": rnd_genesets_low,
            "rnd_ref": rnd_genesets_ref, "rnd_all": rnd_genesets_all}
        returnings.append(dict_)

    if len(returnings) == 1:
        return returnings[0]
    else:
        return returnings


def list_overlapping_gs(
            ref_dict: dict[str, list[str]]
        ) -> list[str]:
    """Identify and return keys of genesets that are fully overlapped by others.

    Args:
        ref_dict (dict[str, list[str]]): Dictionary of genesets keyed by identifier.

    Returns:
        list[str]:
            List of geneset identifiers with 100% overlap with others.
            calculation, utils

    Tags:
        calculation, utils
    """
    overlap_data = []
    for var1, var2 in combinations(ref_dict.keys(), 2):
        # Calculate the intersection count
        set1, set2 = set(ref_dict[var1]), set(ref_dict[var2])
        overlap_count = len(set1 & set2)

        # Calculate overlap percentages for each list in the pair
        overlap_percentage_var1 = (overlap_count / len(set1)) * 100 if len(set1) > 0 else 0
        overlap_percentage_var2 = (overlap_count / len(set2)) * 100 if len(set2) > 0 else 0

        overlap_data.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Overlap Count': overlap_count,
            'Overlap Percentage Var1': overlap_percentage_var1,
            'Overlap Percentage Var2': overlap_percentage_var2
        })

    # Convert overlap data to a DataFrame
    overlap_df = pd.DataFrame(overlap_data)

    result = []
    for _, row in overlap_df.iterrows():
        if row['Overlap Percentage Var1'] == 100:
            result.append(row['Variable 1'])
        elif row['Overlap Percentage Var2'] == 100:
            result.append(row['Variable 2'])

    return result


def ref_to_list(
            ref: str | list[str] | dict[str, list[str]],
            dict_keys: str | list[str] | None = None
        ) -> list[str]:
    """
    Extracts a list of marker genes from various marker specifications.
    Helper for get_adata_subset

    Args:
        markers (str | list[str] | dict[str, list[str]]): Marker specification.
            - If str: treated as a single marker gene.
            - If list: used directly.
            - If dict: extracts markers from specified keys.
        dict_keys (str | list[str] | None, optional): Keys to extract from the
            dict, if ``markers`` is a dict. If None, use all keys.
            Defaults to None.

    Returns:
        list[str]:
            The resolved list of marker genes.

    Called By:
        get_adata_subset
    """
    if isinstance(ref, str):
        return [ref]
    elif isinstance(ref, list):
        return ref
    elif isinstance(ref, dict):
        if dict_keys is None:
            dict_keys = list(ref.keys())
        elif isinstance(dict_keys, str):
            dict_keys = [dict_keys]
        return list(set(gene for k in dict_keys if k in ref for gene in ref[k]))
    else:
        raise ValueError("Unsupported ``ref`` type. Must be str, list, or dict.")


# ###########################################################################################################
# Genset Enrichment
def score_genes_parallel(
            adata: ad.AnnData,
            ref_dict: dict[str, list[str]],
            inplace: bool = True,
            add_to: str = "obs",
            key_to_add: str = "geneset_scores"
        ) -> pd.DataFrame | None:
    """Efficiently implements the scanpy.pl.score_genes function in parallel.

    This function scores gene sets efficiently by implementing parallel
    processing. It operates on an adata object and updates it in place or
    returns the computed scores based on user preference. The function is
    designed to handle large datasets and includes options for customizing where
    the scores are stored within the adata object.

    NOTE:
        The key_to_add is only used if inplace is True and add_to is in
        ["obsm", "uns"]. The key add_to is only used if inplace is True.

    Args:
        adata (anndata.AnnData): Adata object.
        ref_dict (dict[str, list[str]]): Dictionary containing marker genes for
            different cell types/conditions/etc.
        inplace (bool, optional): If True, the results are added to the 'add_to'
            location within the adata object. Defaults to True.
        add_to (str, optional): Specifies where to add the results. Options are
            "obs", "obsm", or "uns". Defaults to "obs".
        key_to_add (str, optional): The key used for storing results when add_to
            is "obsm" or "uns". Defaults to "geneset_scores".

    Returns:
        pandas.DataFrame | None:
            Returns a DataFrame of gene set scores if inplace is False,
            otherwise modifies the adata object in place.

    Calls:
        get_all_markers_from_ref_dict, get_highly_variable,
        get_valide_ref_dicts, score_genes_efficient

    Tags:
        annotation, calculation, config, obs
    """
    # #########################################################
    # Return immediately if the reference dictionary is empty, else update it with valid genes
    if not ref_dict:
        return
    ref_dict_upd, _ = get_valide_ref_dicts(adata, ref_dict)
    # #########################################################
    # Temporarily replace adata.X with 'log2norm_counts' layer if it exists
    if "log2norm_counts" in adata.layers.keys():
        x_view = adata.X
        adata.X = adata.layers["log2norm_counts"]
    else:
        x_view = None
    # #########################################################
    # Get and extend highly variable genes with marker genes from the reference dictionary
    highly_variables = get_highly_variable(
        adata, return_adata=False, return_genes=True, sort_by_variance=True)
    adata.uns["highly_variables"] = copy(highly_variables)
    highly_variables.extend(get_all_markers_from_ref_dict(ref_dict_upd))
    highly_variables = pd.unique(highly_variables)
    # #########################################################
    # Define a function for parallel scoring of gene sets

    def func(key):
        """Helper"""
        return score_genes_efficient(
            adata, gene_list=ref_dict_upd[key], score_name=key, ctrl_size=len(highly_variables),
            n_bins=adata.uns["config"]["tl"]["score_genes"]["n_bins"],
            gene_pool=highly_variables, inplace=False)

    with parallel_config(backend='threading', n_jobs=adata.uns["config"]["general"]["n_cores"]):
        pds = Parallel()(delayed(func)(i) for i in ref_dict_upd.keys())
    # #########################################################
    # Concatenate results into a DataFrame
    df = pd.concat([x for x in pds if not x.empty], axis=1)
    # #########################################################
    # Restore the original adata.X layer if it was modified
    if x_view is not None:
        adata.X = x_view
    # #########################################################
    # Return the results or update the adata object in place
    if inplace:
        if add_to == "obs":
            intersection = np.intersect1d(df.columns, adata.obs.columns)
            for inter in intersection:
                del adata.obs[inter]
            # TODO: AHHHH PANDAS IS BUGGY for the join!!!
            adata.obs = adata.obs.join(df)
        elif add_to == "uns":
            logger.warning("You should consider adding the geneset scores to the obsm.")
            adata.uns[key_to_add] = df
        elif add_to == "obsm":
            adata.obsm[key_to_add] = df
    else:
        return df


def score_genes(
            adata: ad.AnnData,
            ref_dict: dict[str, list[str]],
            inplace: bool = True,
            add_to: str = "obs",
            key_to_add: str = "geneset_scores"
        ) -> pd.DataFrame | None:
    """This is only for benchmarking "score_genes_paralelle".

    Scores genes in an adata object based on a reference dictionary and updates
    the object with the results or returns the scores.

    This function is primarily for benchmarking the "score_genes_paralelle"
    function. It calculates scores for genes in an adata object and can update
    the object in place or return the scores. Benchmarking tests indicate a
    significant speed increase with parallelization for over 300 genesets.

    NOTE:
        - Approx spead increase is 3x
        - This may be faster if you only want to score below 5 genesets, but in
          this case, the speed is anyway irrelevant. Delete after benchmarking
          paper.
        - The key_to_add is only used if inplace is True and add_to is in
          ["obsm", "uns"]. The key add_to is only used if inplace is True.

    Args:
        adata (anndata.AnnData): Adata object containing the data to be
            processed.
        ref_dict (dict[str): Marker dictionary where keys are gene sets and
            values are lists of marker genes.
        inplace (bool, optional): If True, adds the data to the specified
            attribute of ``adata``. Defaults to True.
        add_to (str, optional): Attribute to which the data should be added, one
            of ["obs", "obsm", "uns"]. Defaults to "obs".
        key_to_add (str, optional): Key used for adding to "obsm" or "uns"
            attributes if ``add_to`` is "obsm" or "uns". Defaults to
            "geneset_scores".

    Returns:
        pandas.DataFrame | None:
            DataFrame containing the scores if ``inplace`` is False,
            otherwise None.

    Calls:
        get_all_markers_from_ref_dict, get_highly_variable,
        get_valide_ref_dicts, score_genes_efficient

    Called By:
        downstream_preprocessing

    Tags:
        annotation, calculation, config, obs
    """
    # #########################################################
    # Validate and update the reference dictionary based on the genes in adata
    if not ref_dict:
        return
    # Update the marker gene dictionary for genes that are valid
    ref_dict_upd, _ = get_valide_ref_dicts(adata, ref_dict)
    # #########################################################
    # Check if "log2norm_counts" is present in adata layers and temporarily replace adata.X
    if "log2norm_counts" in adata.layers.keys():
        x_view = adata.X
        adata.X = adata.layers["log2norm_counts"]
    else:
        x_view = None
    # #########################################################
    # Get highly variable genes and update adata.uns with them
    highly_variables = get_highly_variable(
        adata, return_adata=False, return_genes=True, sort_by_variance=True)
    adata.uns["highly_variables"] = copy(highly_variables)
    # Extend the list with all marker genes for cell type scoring and remove duplicates
    highly_variables.extend(get_all_markers_from_ref_dict(ref_dict_upd))
    highly_variables = pd.unique(highly_variables)
    # #########################################################
    # Calculate scores for each gene set using efficient scoring
    pds = []
    for key in ref_dict_upd.keys():
        pds.append(score_genes_efficient(
            adata, gene_list=ref_dict_upd[key], score_name=key, ctrl_size=len(highly_variables),
            n_bins=adata.uns["config"]["tl"]["score_genes"]["n_bins"],
            gene_pool=highly_variables, inplace=False))
    # #########################################################
    # Merge the scores into a single DataFrame
    df = pd.concat([x for x in pds if not x.empty], axis=1)
    # #########################################################
    # Restore the original adata.X layer if it was modified
    if x_view is not None:
        adata.X = x_view
    # #########################################################
    # Update the adata object or return the scores based on ``inplace`` and ``add_to`` arguments
    if inplace:
        if add_to == "obs":
            # TODO: AHHHH PANDAS IS BUGGY for the join!!!
            adata.obs = adata.obs.join(df)
        elif add_to == "uns":
            logger.warning("You should consider adding the geneset scores to the obsm.")
            adata.uns[key_to_add] = df
        elif add_to == "obsm":
            adata.obsm[key_to_add] = df
    else:
        return df


def map_geneset_to_degs(
            deg_df: pd.DataFrame,
            geneset_df_row_wise: pd.DataFrame | None = None,
            geneset_df_col_wise: pd.DataFrame | None = None,
            geneset_dict: dict[str, list[str]] | None = None,
            key_to_add: str = "hallmarks"
        ) -> pd.DataFrame:
    """
    Maps gene names in a Differentially Expressed Genes (DEG) DataFrame to
    their corresponding hallmarks from a hallmark DataFrame.

    This function creates a dictionary from the hallmark DataFrame, where each
    gene is mapped to one or more hallmark gene sets. It then uses this
    dictionary to annotate the DEG DataFrame by adding a new column that lists
    the associated hallmarks for each gene.

    NOTE:
        This function assumes that the 'names' column in deg_df and the
        'geneSymbols' column in geneset_df_row_wise contain string
        representations of gene names. Ensure that the DataFrame structures
        match the expected format before applying this function.

    Args:
        deg_df (pandas.DataFrame): A DataFrame containing the differentially
            expressed genes with a column named 'names' that holds the gene
            names.
        geneset_df_row_wise (pandas.DataFrame | None, optional): A DataFrame
            containing hallmark gene sets with columns 'geneSymbols' (a
            comma-separated string of gene names) and 'GeneSet' (the name of the
            hallmark gene set). Defaults to None.

    Returns:
        pandas.DataFrame:
            The DEG DataFrame with an additional column 'hallmarks' that
            contains the associated hallmark gene sets for each gene.

    Calls:
        convert_ref_dict_to_gs_per_row,
        data_frame_to_valide_ref_dict

    Called By:
        plot_hallmark_group_heatmap, process_and_save_degs

    Tags:
        DEG, annotation
    """
    # #########################################################
    # Check if exactly one geneset argument is used
    if sum(x is not None for x in [
            geneset_df_row_wise, geneset_df_col_wise, geneset_dict]) != 1:
        raise AttributeError(
            "Please Provide only one of these arguments: geneset_df_row_wise, geneset_df_col_wise, geneset_dict!")
    # #########################################################
    # Create a dictionary from the hallmark DataFrame
    # The dictionary maps each gene to its corresponding hallmark gene sets
    if geneset_dict is not None:
        geneset_df_row_wise = convert_ref_dict_to_gs_per_row(geneset_dict)

    elif geneset_df_col_wise is not None:
        geneset_df_row_wise = convert_ref_dict_to_gs_per_row(
                data_frame_to_valide_ref_dict(geneset_df_col_wise))

    geneset_dict = {}
    for _, row in geneset_df_row_wise.iterrows():
        # ##########################################
        # Split the 'geneSymbols' by comma to get individual gene names
        genes = row['geneSymbols'].split(',')
        for gene in genes:
            # Add the gene to the dictionary with its corresponding hallmark gene set
            if gene in geneset_dict:
                geneset_dict[gene].append(row['GeneSet'])
            else:
                geneset_dict[gene] = [row['GeneSet']]
    # #########################################################
    # Define a function to map gene names to hallmarks using the dictionary
    # def get_genesets(gene_name):
    #     """
    #     Retrieve hallmark gene sets associated with a specific gene.

    #     Args:
    #         gene_name (str): The name of the gene to retrieve hallmarks for.

    #     Returns:
    #         str: A comma-separated string of hallmark gene sets associated with the gene.
    #              Returns an empty string if the gene has no associated hallmarks.
    #     """
    #     return ','.join(geneset_dict.get(gene_name, []))
    # #########################################################
    # Apply the mapping function to the 'names' column of the DEG DataFrame
    # The result is a new column 'hallmarks' in deg_df
    # deg_df[key_to_add] = deg_df['names'].apply(get_genesets)
    # TODO: Test this, but it should work, was chatGPT shit, I didn't diff -.-
    deg_df[key_to_add] = deg_df['names'].apply(lambda x: ','.join(geneset_dict.get(x, [])))
    # #########################################################
    return deg_df


def count_genesets_per_group(
            updated_deg_df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Counts and aggregates hallmark information per group from the input
    DataFrame. This function calculates the total, positive, and negative counts
    of hallmarks as well as their associated scores across different groups in
    the DataFrame. Additionally, it normalizes these counts and scores by the
    total number of genes and the number of non-NA hallmark entries within each
    group.

    NOTE:
        Ensure that the input DataFrame has no missing values in the ``group`` and
        ``scores`` columns. The function assumes that ``hallmarks`` column entries
        are either comma-separated strings or NaN. If the format is different,
        preprocess the DataFrame accordingly.

    Args:
        updated_deg_df (pandas.DataFrame):
            A DataFrame that contains at least the following columns:
                - ``group``: Group identifier for each row.
                - ``hallmarks``: Comma-separated string of hallmarks associated
                  with each gene.
                - ``scores``: Numerical score associated with each gene's hallmark.

    Returns:
        pandas.DataFrame:
            A DataFrame containing aggregated hallmark counts, scores,
            and their normalized values per group. The returned DataFrame includes

    The Columns in the returned DataFrame are:
        - ``hallmark``: Name of the hallmark.
        - ``total_count``: Total count of each hallmark within the group.
        - ``group``: Group identifier.
        - ``pos_count``: Count of the hallmark where the associated score is
          non-negative.
        - ``neg_count``: Count of the hallmark where the associated score is
          negative.
        - ``total_scores``: Sum of scores for each hallmark within the group.
        - ``total_abs_scores``: Sum of absolute scores for each hallmark
          within the group.
        - ``pos_scores``: Sum of positive scores for each hallmark.
        - ``neg_scores``: Sum of negative scores for each hallmark.
        - ``total_normalized_by_genes``: Normalized total hallmark count by
          total gene number.
        - ``pos_normalized_by_genes``: Normalized positive hallmark count by
          total gene number.
        - ``neg_normalized_by_genes``: Normalized negative hallmark count by
          total gene number.
        - ``total_normalized_by_non_na``: Normalized total hallmark count by
          non-NA hallmark count.
        - ``pos_normalized_by_non_na``: Normalized positive hallmark count by
          non-NA hallmark count.
        - ``neg_normalized_by_non_na``: Normalized negative hallmark count by
          non-NA hallmark count.
        - ``normalized_total_scores_by_genes``: Normalized total scores by
          total gene number.
        - ``normalized_total_abs_scores_by_genes``: Normalized total abs
          scores by total gene number.
        - ``normalized_pos_scores_by_genes``: Normalized positive scores by
          total gene number.
        - ``normalized_neg_scores_by_genes``: Normalized negative scores by
          total gene number.

    Called By:
        plot_hallmark_group_heatmap

    TODO:
        - Consider optimizing the function for larger datasets by parallelizing
          the group-level computations.

    Tags:
        DEG, calculation, groupby
    """
    # #########################################################
    # Initialize an empty list to store the results
    results = []
    # #########################################################
    # Iterate over each unique group in the DataFrame
    for group in updated_deg_df['group'].unique():
        # ##########################################
        # Filter the DataFrame for the current group
        group_df = updated_deg_df[updated_deg_df['group'] == group]

        # Calculate the total number of genes and non-NA hallmarks
        total_genes = len(group_df)
        non_na_hallmarks = group_df['hallmarks'].apply(lambda x: pd.notna(x)).sum()
        # ##########################################
        # Initialize Counters to count the hallmarks and sum scores
        pos_counter = Counter()
        neg_counter = Counter()
        total_counter = Counter()
        pos_score_counter = Counter()
        neg_score_counter = Counter()
        total_score_counter = Counter()
        total_abs_score_counter = Counter()
        # ##########################################
        # Iterate over each row in the group to count hallmarks and sum scores
        for _, row in group_df.iterrows():
            hallmarks = row['hallmarks']
            score = row['scores']
            if hallmarks:  # Only count if there are hallmarks
                hallmark_list = hallmarks.split(',')
                total_counter.update(hallmark_list)
                if score >= 0:
                    pos_counter.update(hallmark_list)
                    for hallmark in hallmark_list:
                        pos_score_counter[hallmark] += score
                else:
                    neg_counter.update(hallmark_list)
                    for hallmark in hallmark_list:
                        neg_score_counter[hallmark] += score
                for hallmark in hallmark_list:
                    total_abs_score_counter[hallmark] += abs(score)
                    total_score_counter[hallmark] += score
        # ##########################################
        # Convert the counter to a DataFrame and add the group information
        hallmark_count_df = pd.DataFrame(total_counter.items(), columns=['hallmark', 'total_count'])
        hallmark_count_df['group'] = group

        # Add the total, positive, and negative counts and scores
        hallmark_count_df['pos_count'] = hallmark_count_df['hallmark'].map(pos_counter).fillna(0)
        hallmark_count_df['neg_count'] = hallmark_count_df['hallmark'].map(neg_counter).fillna(0)

        hallmark_count_df['total_scores'] = hallmark_count_df['hallmark'].map(total_score_counter)
        hallmark_count_df['total_abs_scores'] = hallmark_count_df['hallmark'].map(total_abs_score_counter)
        hallmark_count_df['pos_scores'] = hallmark_count_df['hallmark'].map(pos_score_counter).fillna(0)
        hallmark_count_df['neg_scores'] = hallmark_count_df['hallmark'].map(neg_score_counter).fillna(0)
        # ##########################################
        # Calculate normalized counts and scores by genes and non-NA hallmarks
        hallmark_count_df['total_normalized_by_genes'] = hallmark_count_df['total_count'] / total_genes
        hallmark_count_df['pos_normalized_by_genes'] = hallmark_count_df['pos_count'] / total_genes
        hallmark_count_df['neg_normalized_by_genes'] = hallmark_count_df['neg_count'] / total_genes

        hallmark_count_df['total_normalized_by_non_na'] = hallmark_count_df['total_count'] / non_na_hallmarks
        hallmark_count_df['pos_normalized_by_non_na'] = hallmark_count_df['pos_count'] / non_na_hallmarks
        hallmark_count_df['neg_normalized_by_non_na'] = hallmark_count_df['neg_count'] / non_na_hallmarks

        hallmark_count_df['normalized_total_scores_by_genes'] = hallmark_count_df['total_scores'] / total_genes
        hallmark_count_df['normalized_total_abs_scores_by_genes'] = hallmark_count_df['total_abs_scores'] / total_genes
        hallmark_count_df['normalized_pos_scores_by_genes'] = hallmark_count_df['pos_scores'] / total_genes
        hallmark_count_df['normalized_neg_scores_by_genes'] = hallmark_count_df['neg_scores'] / total_genes
        # ##########################################
        # Append the result to the results list
        results.append(hallmark_count_df)
    # #########################################################
    # Concatenate all group DataFrames into one final DataFrame
    final_df = pd.concat([x for x in results if not x.empty], ignore_index=True)
    # #########################################################
    # Optionally, sort the DataFrame by group and total_count
    final_df = final_df.sort_values(by=['group', 'total_count'], ascending=[True, False]).reset_index(drop=True)

    return final_df


def get_msigdb_df(
            organism: str = "human",
            only_hallmarks: bool = False,
            path: str | None = None
        ) -> pd.DataFrame:
    """
    Loads the MSigDB data from a JSON file and converts it into a pandas
    DataFrame.

    This function reads a JSON file containing MSigDB (Molecular Signatures
    Database) data, processes it, and returns a pandas DataFrame with the data.
    The function expects the JSON file to be structured in a way where each
    top-level key represents a GeneSet, and the corresponding value is a
    dictionary of properties.

    NOTE:
        This function assumes a specific structure in the JSON file and may
        not work correctly if the structure is different.

    Args:

        organism (str, optional): The organism for which to load MSigDB data.
            Defaults to "human".
        only_hallmarks (bool, optional): If True, only load the hallmark gene
            sets. Defaults to False.
        path (str | None, optional): The path to the JSON file containing the
            MSigDB data. If not provided, a default path will be used. Defaults
            to None.

    Returns:
        pandas.DataFrame:
            A pandas DataFrame where each row represents a GeneSet and
            its associated data.

    Raises:
        FileExistsError: If the specified path to the JSON file does not exist.

    Called By:
        get_DEG_gene_csvs, plot_hallmark_group_heatmap

    Tags:
        io
    """
    # #########################################################
    # Default path assignment if no path is provided
    if path is None:
        # ##########################################
        # Define default path and filename for MSigDB data
        possible_organisms = list(
            GENESET_FILENAMES["PATH_TO_GENESETS_DATABASES"].keys())
        if organism in possible_organisms:
            if "msigdb" in GENESET_FILENAMES[
                    "PATH_TO_GENESETS_DATABASES"][organism]:
                msigdb_path = GENESET_FILENAMES[
                    "PATH_TO_GENESETS_DATABASES"][organism]["msigdb"]
            else:
                raise NotImplementedError(
                        f'Msigdb is not accessible for this {organism}')
        else:
            raise NotImplementedError(
                "Only the organisms are "
                f'accessibleare: {", ".join(possible_organisms)}')

        path = ALL_PATHS["PATH_TO_DATABASE"] + msigdb_path
    # #########################################################
    # Check if the specified path exists and raise an error if it does not
    if not os.path.exists(path):
        raise FileExistsError("Path to the MSigDB JSON file does not exist!")
    # #########################################################
    # Load the JSON data from the file
    with open(path, 'r', encoding='utf-8') as file:
        data = json_load(file)
    # #########################################################
    # Initialize a list to store processed rows for the DataFrame
    rows = []
    # #########################################################
    # Process each GeneSet in the JSON and convert it into a row for the DataFrame
    for main_key, inner_dict in data.items():
        row = {"GeneSet": main_key}  # Start a new row with the GeneSet name

        # Iterate over each key-value pair in the inner dictionary
        for inner_key, value in inner_dict.items():
            if isinstance(value, list):
                # Convert list values into a comma-separated string
                row[inner_key] = ','.join(map(str, value))
            else:
                row[inner_key] = value

        rows.append(row)  # Add the row to the list of rows
    # #########################################################
    # Create a dataframe
    df = pd.DataFrame(rows)
    # #########################################################
    if only_hallmarks:
        if organism == "human":
            hallmark_accessor = "H"
        elif organism == "mouse":
            hallmark_accessor = "MH"
        else:
            raise NotImplementedError("Only the organisms human and mouse are accessible.")

        df = df[df["collection"] == hallmark_accessor][["GeneSet", "geneSymbols"]]
    # #########################################################
    # Convert the list of rows into a pandas DataFrame and return it
    return df


# ###########################################################################################################
# Pipeline Alterations
def disable_all_run_downstream(
            adata: ad.AnnData,
            not_to_plot: list[str] = ["cl", "down"],
            not_to_create: list[str] = ["DEG", "marker", "cluster_stats"]
        ) -> None:
    """
    Disables all plotting, saving, and CSV creating functions for ``run_downstream``.

    This function modifies the ``adata`` object to disable specific plotting and
    data creation functionalities within the ``run_downstream`` process by setting
    configuration flags to ``False``. It can be used to streamline the analysis by
    selectively enabling or disabling specific outputs.

    NOTE:
        Run this before you want to plot only some plots, then you don't need to
        disable all manually.

    Args:
        adata (anndata.AnnData): Adata object containing the analysis data and
            configurations.
        not_to_plot (list[str], optional): A list of keys in
            ``adata.uns["config"]["to_plot"]`` to disable plotting
            functionalities. Defaults to ["cl".
        not_to_create (list[str], optional): A list of keys in
            ``adata.uns["config"]["to_create"]`` to disable creation of CSV files
            and other outputs. Defaults to ["DEG".

    Returns:
        None

    Tags:
        config
    """
    # #########################################################
    # Disabling plotting functionalities based on ``not_to_plot``
    if "cl" in not_to_plot:
        # ##########################################
        # Disable specific plot visualization in the "cl" configuration
        # NOTE: This is not actually part of run_downstream but disable it anyway
        adata.uns["config"]["to_plot"]["cl"]["visualize_pca_variance_ratio"] = False

    if "down" in not_to_plot:
        # ##########################################
        # Disable various plots within the "down" configuration
        adata.uns["config"]["to_plot"]["down"]["visualize_leiden_umap"] = False
        adata.uns["config"]["to_plot"]["down"]["visualize_qc_umap"] = False
        adata.uns["config"]["to_plot"]["down"]["marker_dendrograms"]["visualize"] = False
        adata.uns["config"]["to_plot"]["down"]["gradients_umap"]["visualize"] = False
        adata.uns["config"]["to_plot"]["down"]["cluster_violins"]["visualize"] = False
        adata.uns["config"]["to_plot"]["down"]["rank_genes_groups"]["visualize"] = True
        adata.uns["config"]["to_plot"]["down"]["cell_type_score_umap"]["visualize"] = False
        adata.uns["config"]["to_plot"]["down"]["cluster_dotplot"]["visualize"] = False
    # #########################################################
    # Disabling data creation functionalities based on ``not_to_create``
    if "DEG" in not_to_create:
        # ##########################################
        # Disable DEG CSV creation in the "down" configuration
        adata.uns["config"]["to_create"]["down"]["DEG_csv"]["create"] = False
    if "marker" in not_to_create:
        # ##########################################
        # Disable DEG CSV creation in the "down" configuration
        adata.uns["config"]["to_create"]["down"]["DEG_csv"]["create"] = False

    if "cluster_stats" in not_to_create:
        # ##########################################
        # Disable cluster statistics CSV creation in the "down" configuration
        adata.uns["config"]["to_create"]["down"]["cluster_stats_csv"]["create"] = False


# ###########################################################################################################
# Celltype annotation functions
def check_group_overlap(
            adata: ad.AnnData,
            groupby: str = "cell_type",
            delimiter: str = ", "
        ) -> None:
    """Removes duplicate groups from an obs column in an adata object.

    This function processes a specific obs column in the provided adata object,
    typically named "cell_type". It ensures that any duplicate groups listed
    within the column are removed, leaving a unique set of groups for each obs.

    Args:
        adata (anndata.AnnData): Adata object.
        groupby (str, optional): The key in the obs DataFrame (adata.obs) that
            specifies the column to be processed. Default is "cell_type".
            Defaults to "cell_type".
        delimiter (str, optional): The delimiter between the groups.
            Defaults to ", ".

    Returns:
        None:
            The function modifies the adata object in place and does not
            return a value.

    Raises:
        KeyError: If the specified ``groupby`` is not found in the obs DataFrame
            (adata.obs).

    Calls:
        validate_groupby_column

    Tags:
        groupby, obs
    """
    # #########################################################
    # Check if the specified obs key exists in the data
    validate_groupby_column(
        adata.obs, groupby)
    # #########################################################
    # Process each entry in the specified obs column to remove duplicates
    replacement = []
    for groups in adata.obs[groupby].unique():
        # ##########################################
        # Split the groups by comma and space, then remove duplicates
        all_groups = []
        for group in groups.split(delimiter):
            if group not in all_groups:
                all_groups.append(group)
        # sort the groups, to ensure the same final names for all groups
        all_groups.sort()
        # Join the unique groups back into a single string
        replacement.append(delimiter.join(all_groups))
    # #########################################################
    # Update the obs column with the processed groups
    adata.obs[groupby] = replacement


def name_groups(
            adata: ad.AnnData,
            name_to_clusters: dict[str, list[str]],
            config: dict | None = None,
            groupby: str = "cell_type",
            subset_to_group: bool = False,
            keep_group_type_boolean: bool = False
        ) -> ad.AnnData:
    """
    Creates a column in the adata object for group annotation based on the
    provided groupby.

    This function populates an obs column in the adata object with groups
    based on a mapping from cluster names to groups. It allows for optional
    subsetting and management of temporary columns used during the process.

    Deprecated: Don't know why I wrote it, if you read this and have an idea,
        please write a proper documentation.

    NOTE:
        This function is deprecated and may be removed in future versions.
        Please consult the author if further clarification is needed.

    Args:
        adata (anndata.AnnData): Adata object.
        name_to_clusters (dict[str, list[str]]): A dictionary mapping group names to lists
            of cluster identifiers.
        config (dict | None, optional): A dictionary containing configuration
            parameters. If None, the configuration is pulled from
            ``adata.uns["config"]``. Defaults to None.
        groupby (str, optional): The name of the column to be created in
            ``adata.obs`` for group annotation. Defaults to "cell_type".
        subset_to_group (bool, optional): If True, the adata object will be
            subsetted to include only groups with identified group types.
            Defaults to False.
        keep_group_type_boolean (bool, optional): If False, temporary columns
            used to mark groups belonging to each type will be deleted after the
            final group column is created. Defaults to False.

    Returns:
        anndata.AnnData:
            The modified adata object with the added group column
            and potentially additional modifications based on the provided
            arguments.

    Raises:
        KeyError: If the specified ``config`` does not contain the necessary
            keys.

    Calls:
        validate_groupby_column

    Tags:
        annotation, config, groupby, obs
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    # #########################################################
    # Configuration setup and dictionary preparation
    if config is None:
        config = adata.uns["config"]
    # #########################################################
    # Annotate groups by group and update the adata object
    for group_type, clusters in name_to_clusters.items():
        # Annotate groups belonging to each cluster group
        adata.obs[name_to_clusters[group_type]] = (
            adata.obs[config["general"]["cluster_algorithm"]].isin(clusters))

    # Create a unified observation column for all groups
    adata.obs[groupby] = "False"
    for group_type in name_to_clusters.keys():
        adata.obs.loc[adata.obs[name_to_clusters[group_type]], groupby] = group_type
    # #########################################################
    # Optional subsetting of the adata object
    if subset_to_group:
        adata = adata[adata.obs[groupby] != "False"]
    # #########################################################
    # Convert the group column to a categorical data type
    adata.obs[groupby] = adata.obs[groupby].astype("category")
    # #########################################################
    # Optional cleanup of temporary boolean columns
    if not keep_group_type_boolean:
        for name in name_to_clusters.keys():
            del adata.obs[name_to_clusters[name]]

    return adata


def update_grouping(
            adata: ad.AnnData,
            sub_adata: ad.AnnData,
            adata_obs_key: str,
            sub_adata_obs_key: str,
            inplace: bool = True,
            new_adata_obs_key: str | None = None
        ) -> pd.Series | None:
    """
    Update the clustering of cells in an adata object based on a sub-adata object.

    This function modifies the clustering information in the main adata object
    (``adata``) by updating the specified observation key (``adata_obs_key``) with
    values from a corresponding key (``sub_adata_obs_key``) in the sub-adata
    object (``sub_adata``). If a new observation key is provided via
    ``new_adata_obs_key``, the updated values are stored under this new key. If
    ``inplace`` is set to False, the function returns the updated observation
    column; otherwise, it performs the update in place and returns None.

    NOTE:
        Ensure that both ``adata`` and ``sub_adata`` have matching observation
        names (cells).

    Args:
        adata (anndata.AnnData): The main adata object containing the original
            clustering data.
        sub_adata (anndata.AnnData): The sub-adata object containing the new
            clustering information.
        adata_obs_key (str): The key in ``adata.obs`` representing the original
            clustering.
        sub_adata_obs_key (str): The key in ``sub_adata.obs`` representing the new
            clustering.
        inplace (bool, optional): If True, update ``adata.obs`` in place. If
            False, return the updated column. Default is True. Defaults to True.
        new_adata_obs_key (str | None, optional): The key under which to store
            the updated clustering data in ``adata.obs``. If None, updates the
            original key. Default is None. Defaults to None.

    Returns:
        pandas.Series | None:
            The updated clustering data as a pandas Series if ``inplace`` is False,
            otherwise None.

    Raises:
        ValueError: If ``new_adata_obs_key`` already exists in ``adata.obs``.

    TODO:
        Extend this function to handle cases where ``sub_adata_obs_key`` is not
        directly mapped to ``adata_obs_key`` but needs a more complex update
        strategy.
    """
    # #########################################################
    # Check if a new key is provided, otherwise use the original key
    if new_adata_obs_key:
        # ##########################################
        # Check if the new key already exists to avoid overwriting existing data
        if new_adata_obs_key in adata.obs:
            raise ValueError(
                f"{new_adata_obs_key} already exists in adata.obs. Choose a different name or set inplace=True.")
        # ##########################################
        # Create a copy of the original data under the new key
        adata.obs[new_adata_obs_key] = adata.obs[adata_obs_key].copy()
        target_key = new_adata_obs_key
    else:
        # ##########################################
        # If no new key is provided, use the original key for updates
        target_key = adata_obs_key
    # #########################################################
    # Ensure that the target column and sub_adata column are of string type
    adata.obs[target_key] = adata.obs[target_key].astype(str)
    sub_adata.obs[sub_adata_obs_key] = sub_adata.obs[sub_adata_obs_key].astype(str)
    # #########################################################
    # Update the clustering for cells that are present in both adata and sub_adata
    overlapping_cells = adata.obs_names.intersection(sub_adata.obs_names)
    adata.obs.loc[overlapping_cells, target_key] = sub_adata.obs.loc[overlapping_cells, sub_adata_obs_key]
    # #########################################################
    # Return None if inplace is True, otherwise return the updated column
    if inplace:
        return None
    else:
        return adata.obs[target_key]


def calc_group_overlap_scores(
            adata: ad.AnnData,
            config: dict,
            other_cell_mapping_inverse: dict[str, str],
            other_cell_mapping: dict[str, str],
            print_res: bool = False
        ) -> tuple[float, float, float]:
    """
    Calculate the overlap scores between two cluster mappings, including
    the Jaccard index and Adjusted Rand Index (ARI).

    This function compares cluster assignments from two different mappings
    (typically from different algorithms or datasets) and calculates various
    overlap metrics. It is designed to assist in understanding how similar the
    clustering results are.

    NOTE:
        If the number of clusters in the dataset does not match the number of
        clusters in the provided mappings, the function will return early
        without completing the calculations.

    Args:
        adata (anndata.AnnData): Adata object.
        config (dict): Configuration dictionary containing clustering and
            analysis parameters.
        other_cell_mapping_inverse (dict[str): Inverse mapping of cell barcodes
            to clusters from the other dataset or clustering result.
        other_cell_mapping (dict[str): Mapping of cell barcodes to clusters from
            the other dataset or clustering result.
        print_res (bool, optional): If True, print intermediate and final
            results to logger. Default is False. Defaults to False.

    Returns:
        tuple[float, float, float]:
              A tuple containing:

                  - ``j_score`` (float): The average Jaccard score across all
                    clusters.
                  - ``overall_j_score`` (float): The overall Jaccard score
                    considering all overlaps.
                  - ``adjusted_rand_score_`` (float): The Adjusted Rand Index (ARI)
                    score comparing the two clusterings.

    Raises:
        ValueError: If the number of clusters in the dataset does not match the
            number of clusters in the provided mappings.

    TODO:
        - Implement additional overlap metrics if required.
        - Improve the efficiency of the overlap calculation for larger datasets.
    """
    # #########################################################
    # Check if the number of clusters in the dataset matches the expected number
    if len(adata.obs.leiden.cat.categories) != len(other_cell_mapping_inverse.keys()):
        if not config["general"]["analysis_only"]:
            logger.warning("ATTENTION!!! the number of clusters, doesn't match!")
        return

    # Print configuration settings if print_res is True
    if print_res:
        logger.info(
            config["tl"]["leiden"]["resolution"],
            config["pp"]["neighbors"]["n_pcs"],
            config["pp"]["neighbors"]["n_neighbors"],
            config["tl"]["umap"]["spread"],
            config["tl"]["umap"]["min_dist"])
    # #########################################################
    # Mapping cells to clusters
    current_cell_mapping = dict(adata.obs[config["general"]["cluster_algorithm"]])
    current_cell_mapping_inverse = {}
    for uni in np.sort(np.array(np.unique(list(current_cell_mapping.values())), dtype=int)):
        current_cell_mapping_inverse[int(uni)] = []
    for k, v in current_cell_mapping.items():
        current_cell_mapping_inverse[int(v)].append(k)

    # Prepare lists for comparison of cluster assignments
    namesort_current = []
    namesort_other = []
    for barcode in current_cell_mapping.keys():
        namesort_current.append(current_cell_mapping[barcode])
        namesort_other.append(other_cell_mapping[barcode])

    # Calculate Adjusted Rand Index (ARI)
    adjusted_rand_score_ = adjusted_rand_score(namesort_current, namesort_other)
    if print_res:
        logger.info("Overall Adjusted rand index: ", adjusted_rand_score_)
    # #########################################################
    # Calculate overlaps and Jaccard scores for each cluster
    overlaps = []
    unions = []
    jaccards = []
    overlaps_perc = []
    lengths = []
    ids = []
    for i in range(len(other_cell_mapping_inverse.keys())):
        overlaps.append([])
        unions.append([])
        jaccards.append([])
        overlaps_perc.append([])
        lengths.append([])
        ids.append([])
        len_current = len(current_cell_mapping_inverse[i])
        for j in range(len(current_cell_mapping_inverse.keys())):
            len_other = len(other_cell_mapping_inverse[j])
            overlap = [x for x in other_cell_mapping_inverse[j] if x in current_cell_mapping_inverse[i]]
            len_overlap = len(overlap)
            overlaps[-1].append(len(overlap))
            len_union = len_other + len_current - len_overlap
            unions[-1].append(len_union)
            jaccards[-1].append(len_overlap / len_union)
            overlaps_perc[-1].append([len(overlap) / len_other, len_overlap / len_current])
            lengths[-1].append([len_other, len_current])
            ids[-1].append([i, j])
    # #########################################################
    # Summarize and log the results
    all_overlaps = 0
    all_unions = 0
    all_jaccard_scores = 0
    cluster_counter = 0
    for a, i, j, k, ja, un in zip(overlaps_perc, overlaps, lengths, ids, jaccards, unions):
        max_ = np.flip(np.argsort(np.array(a).sum(axis=1), axis=0))[0]
        cluster_counter += 1
        if not config["general"]["analysis_only"]:
            logger.info("Cluster numbers: ", k[max_])
            logger.info("overlap Perc: ", a[max_])
            logger.info("jaccard score: ", ja[max_])
            logger.info("overlap num: ", i[max_])
            logger.info("num samples in cluster: ", j[max_])
        all_overlaps += i[max_]
        all_unions += un[max_]
        all_jaccard_scores += ja[max_]

    # Calculate overall Jaccard scores
    j_score = all_jaccard_scores / cluster_counter
    overall_j_score = all_overlaps / all_unions
    if print_res:
        logger.info("Clustermapping Jaccard score of all overlap: ", overall_j_score)
        logger.info("Clustermapping Jaccard score avg and sum: ", j_score, all_jaccard_scores)
    if not config["general"]["analysis_only"]:
        logger.info("'" * 80)
        logger.info("'" * 80)
    return j_score, overall_j_score, adjusted_rand_score_


# ###########################################################################################################
# Data extraction
def get_pseudobulk(
            adata: ad.AnnData,
            obs_key: str = "",
            layer: str | None = None,
            save_path: str = "",
            transpose: bool = False,
            include_all: bool = True,
            weight: bool = False,
            weight_n_counts: float = 1e6,
            normalize_by_cells: bool = False
        ) -> pd.DataFrame:
    """Create un-/weighted pseudo bulk data from adata object.

    This function aggregates data from an adata object to generate pseudo bulk
    data, either for all cells or based on a specified observation key. The data
    can be sourced from a specified layer, weighted to match a specific total
    count, normalized by the number of cells per category, transposed for
    compatibility with R, and optionally saved to a file.

    NOTE:
        If weight=True, verify the weight_n_counts parameter, which defaults
        to one million.

    Args:
        adata (anndata.AnnData): Adata object containing the single-cell data.
        obs_key (str, optional): The key in the observation (obs) field used to
            group cells. If specified, pseudo bulk data is generated for each
            unique value in this key. Defaults to "".
        layer (str | None, optional): The key of the layer in the adata object
            to be used for data extraction. If specified, data will be taken
            from this layer instead of the default X matrix. Defaults to None.
        save_path (str, optional): Path to save the resulting pseudo bulk data
            as a CSV file. If empty, the function returns the dataframe instead.
            Defaults to "".
        transpose (bool, optional): Whether to transpose the resulting
            dataframe. Transposing is often needed for R compatibility.
            Defaults to False.
        include_all (bool, optional): Whether to include all cells in a single
            pseudo bulk group. If True, a pseudo bulk dataset is created for all
            cells combined. Defaults to True.
        weight (bool, optional): Whether to weight the pseudo bulk data to match
            a specific total count. This option normalizes each group by its sum
            and scales it to the provided weight_n_counts. Defaults to False.
        weight_n_counts (float, optional): The target total count for weighting.
            Used only if weight is True. Defaults to 1e6.
        normalize_by_cells (bool, optional): Whether to normalize the pseudo
            bulk data by the number of cells per category in obs_key. If True,
            each category's data is divided by the number of cells in that
            category. Defaults to False.

    Returns:
        pandas.DataFrame:
            A dataframe containing the pseudo bulk data, either
            returned or saved as a CSV file.

    Raises:
        ValueError: If no pseudo bulk data is generated due to missing obs_key
            and include_all is False.
        KeyError: If the specified layer is not found in the adata object.

    Tags:
        calculation, groupby, io, obs
    """
    df = {}
    # #########################################################
    # Check if the specified layer exists in the adata object
    if layer and layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in the adata object.")
    # #########################################################
    # Select the appropriate adata: layer if specified, otherwise X matrix
    data_matrix = adata.layers[layer] if layer else adata.X
    # #########################################################
    # Create pseudo bulk data for all cells if include_all is True
    if include_all:
        df["all"] = data_matrix.sum(0).A1 if hasattr(data_matrix, "A1") else data_matrix.sum(0).flatten()
    # #########################################################
    # Create pseudo bulk data for each unique value in obs_key if provided
    if len(obs_key) > 0:
        for k in adata.obs[obs_key][adata.obs[obs_key].notna()].unique():
            subset_matrix = data_matrix[adata.obs[obs_key] == k]
            df[k] = subset_matrix.sum(0).A1 if hasattr(subset_matrix, "A1") else subset_matrix.sum(0).flatten()
            # #########################################################
            # Normalize by the number of cells per category if normalize_by_cells is True
            if normalize_by_cells:
                df[k] = df[k] / (adata.obs[obs_key] == k).sum()
    # #########################################################
    # Sanity check to ensure there is at least one pseudo bulk dataset
    if len(df.keys()) == 0:
        raise ValueError("Result is empty! Specify include_all for a complete pseudo-bulk or an obs_key.")
    # #########################################################
    # Apply weighting to normalize and scale each group's pseudo bulk data
    if weight:
        for k in df.keys():
            df[k] = df[k] / df[k].sum() * weight_n_counts
    # #########################################################
    # Convert dictionary to a DataFrame, with rows corresponding to genes
    df = pd.DataFrame(df, columns=adata.var_names.values.tolist()).T
    # #########################################################
    # Transpose the DataFrame if transpose is False (i.e., default for Python usage)
    if not transpose:
        df = df.T
    # #########################################################
    # Save the DataFrame to a CSV file if a path is provided, otherwise return it
    if save_path:
        df.to_csv(save_path)
    else:
        return df


def get_group_key_percentages(
        adata: ad.AnnData,
        groupby: str = "leiden",
        obs_key: str = None,
        pct_threshold: float = 0.2,
        return_series: bool = False,
        include_counts: bool = True) -> pd.DataFrame | pd.Series:
    """
    Calculate per-cluster percentages of a obs_key in an adata object,
    ptionally returning as a DataFrame or Series. Also includes the
    percentage of cells with None in ``obs_key`` and per-key percentages
    excluding None cells.

    Args:
        adata (anndata.AnnData): Adata object.
        groupby (str, optional): Column in adata.obs for grouping (e.g.,
            clusters). Defaults to "leiden".
        obs_key (str, optional): Column in adata.obs to compute percentages for.
            Defaults to None.
        pct_threshold (float, optional): Minimum percentage threshold to
            include. Defaults to 0.2.
        return_series (bool, optional): Whether to return as a MultiIndex
            Series. Defaults to False.
        include_counts (bool, optional): Whether to include raw counts and
            percentages in DataFrame. Defaults to True.

    Returns:
        pandas.DataFrame | pandas.Series:
            DataFrame with counts and percentages or Series with percentages only.

    Calls:
        validate_groupby_column
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    # #########################################################
    # Group by cluster and obs_key, count occurrences
    counts_df = adata.obs.groupby([groupby, obs_key], observed=True).size().reset_index(name="count")

    # Get total cells per cluster
    cluster_sizes = adata.obs.groupby(groupby, observed=True).size().reset_index(name="total_count")

    # Count non-None cells per cluster
    non_none_df = (
        adata.obs[adata.obs[obs_key].notna()]
        .groupby(groupby, observed=True)
        .size().reset_index(name="non_none_count"))

    # Merge counts with cluster sizes and non-None counts
    merged_df = counts_df.merge(cluster_sizes, on=groupby)
    merged_df = merged_df.merge(non_none_df, on=groupby, how="left")
    merged_df["non_none_count"] = merged_df["non_none_count"].fillna(0)

    # Compute percentages
    merged_df["pct"] = merged_df["count"] / merged_df["total_count"]
    merged_df["pct_non_none"] = merged_df.apply(
        lambda row: row["count"] / row["non_none_count"] if row["non_none_count"] > 0 else 0,
        axis=1)

    # Calculate pct_none per cluster
    none_counts = (
        adata.obs[adata.obs[obs_key].isna()]
        .groupby(groupby, observed=True)
        .size().reset_index(name="none_count"))
    cluster_with_none = cluster_sizes.merge(none_counts, on=groupby, how="left")
    cluster_with_none["none_count"] = cluster_with_none["none_count"].fillna(0)
    cluster_with_none["pct_none"] = cluster_with_none["none_count"] / cluster_with_none["total_count"]

    # Merge pct_none back into main df
    merged_df = merged_df.merge(cluster_with_none[[groupby, "pct_none"]], on=groupby, how="left")

    # Apply threshold
    filtered_df = merged_df[merged_df["pct"] > pct_threshold].copy()

    if return_series:
        # Return as Series with MultiIndex (cluster, obs_key), values = pct
        series = filtered_df.set_index([groupby, obs_key])["pct_non_none"]
        return series
    else:
        # Return DataFrame, with or without counts depending on flag
        if not include_counts:
            return filtered_df[[groupby, obs_key, "pct_non_none", "pct", "pct_none"]].reset_index(drop=True)
        return filtered_df.reset_index(drop=True)


# ###########################################################################################################
# Stat dict functions
def create_group_stats_df(
            adata: ad.AnnData,
            groups: list[str],
            stat_list: list[str] | None = None,
            groupby: str | None = None,
            save_path: str = "",
            update_adata: bool = True
        ) -> None:
    """
    Generates a DataFrame containing statistical summaries for
    groups/celltypes/leiden clusters in the provided adata object.

    The function calculates various statistical measures (e.g., mean, median,
    mode, standard deviation) for specified cells grouped by a clustering key.
    It then renames these columns to more descriptive names and adds a column
    indicating the number of cells in each cluster. The final DataFrame is saved
    to a CSV file, and optionally updates the adata object with the computed
    statistics.

    NOTE:
        This function currently assumes that specific observation keys are
        present in the adata object and that the renaming dictionary matches
        those keys.

    Args:
        adata (anndata.AnnData): Adata object.
        groups (list[str]): A list of groups/cell types to be used as keys for
            renaming columns. Each key in this list should match a column in
            ``adata.obs``.
        stat_list (list[str] | None, optional): A list of statistical measures
            to calculate. Possible options are "mean", "median", "mode", "std",
            "min", "max". If not provided, all these options will be calculated.
            Defaults to None.
        groupby (str | None, optional): Optional key to group data by, must be a
            column in adata.obs None will use the
            config["general"]["cluster_algorithm"]. Defaults to None.
        save_path (str, optional): The file path where the resulting DataFrame
            will be saved as a CSV file. Default is an empty string, meaning no
            file will be saved. Defaults to "".
        update_adata (bool, optional):
            If True, the combined DataFrame will be stored in
            ``adata.uns["stats"]["cluster_stats"]``. Defaults to True.

    Returns:
        None

    Raises:
        FileNotFoundError: If the ``save_path`` directory does not exist.
        KeyError: If the ``groupby`` or any keys in the renaming dictionary are
            not found in the input ``adata``.
        ValueError: If an unsupported statistical measure is provided in
            ``stat_list``.

    Calls:
        validate_groupby_column

    Called By:
        run_downstream
    """
    # #########################################################
    # Set default stat_list to include all possible statistics if not provided
    if stat_list is None:
        stat_list = ["mean", "median", "mode", "std", "min", "max"]
    # #########################################################
    # get the clustering algorithm key
    if groupby is None:
        groupby = adata.uns["config"]["general"]["cluster_algorithm"]
    # get the groups
    if groups is None:
        # groups = list(adata.uns[key]["names"].dtype.names)
        groups = adata.obs[groupby].cat.categories.tolist()
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True, groups=groups)
    # #########################################################
    # Initialize an empty DataFrame to store results
    cluster_stats = {}
    # #########################################################
    # Calculate statistics for each cluster and each measure in stat_list
    for stat in stat_list:
        if stat == "mean":
            cluster_stat_df = adata.obs.groupby(by=groupby).mean()
        elif stat == "median":
            cluster_stat_df = adata.obs.groupby(by=groupby).median()
        elif stat == "mode":
            cluster_stat_df = adata.obs.groupby(by=groupby).agg(lambda x: stats_mode(x)[0])
        elif stat == "std":
            cluster_stat_df = adata.obs.groupby(by=groupby).std()
        elif stat == "min":
            cluster_stat_df = adata.obs.groupby(by=groupby).min()
        elif stat == "max":
            cluster_stat_df = adata.obs.groupby(by=groupby).max()
        else:
            raise ValueError(f"Statistic '{stat}' is not supported.")

        # Rename columns according to the groups
        rename_dict = {ct: f'{stat}_{ct}' for ct in groups}
        cluster_stat_df = cluster_stat_df.rename(columns=rename_dict)

        # Add results to the final DataFrame
        cluster_stats[stat] = cluster_stat_df
    # #########################################################
    # Combine all statistical DataFrames into one
    combined_df = pd.concat(
        [x for x in cluster_stats.values() if not x.empty], axis=1)
    # #########################################################
    # Add the number of cells per cluster to the DataFrame
    combined_df["n_cells"] = adata.obs.groupby(
        by=adata.uns["config"]["general"]["cluster_algorithm"]
        ).count().iloc[:, 0].values
    combined_df = combined_df[["n_cells", *combined_df.columns[:-1]]]
    # #########################################################
    # Optionally update the adata object with the combined DataFrame
    if update_adata:
        if "stats" not in adata.uns:
            adata.uns["stats"] = {}
        adata.uns["stats"]["cluster_stats"] = combined_df
    # #########################################################
    # Save the resulting DataFrame to a CSV file, with overwrite option if file exists
    # TODO: Maybe add the overwrite option here
    if len(save_path) != 0:
        if os.path.exists(save_path):
            os.remove(save_path)
        combined_df.to_csv(save_path, index=False)


def create_filter_stat_csv(
            adata: ad.AnnData,
            save_path: str = "filter_stats.csv"
        ) -> None:
    """Summarize the adata.uns["stats"] as a CSV file.

    This function extracts statistics related to filters from the 'stats'
    dictionary within the 'adata.uns' attribute of an adata object and saves
    these statistics as a CSV file. The function looks specifically for keys
    that start with "filter" and creates a DataFrame from these entries.

    NOTE:
        This only works if the adata.uns["stats"] exists!

    Args:
        adata (anndata.AnnData): Adata object containing the data and stats.
        save_path (str, optional): The path where the CSV file will be saved.
            Defaults to "filter_stats.csv".

    Returns:
        None

    Raises:
        KeyError: If the "stats" key does not exist in the 'adata.uns'
            attribute.

    TODO:
        When the package is finished, add all the actual filtering stats keys
        here!

    Tags:
        io, stats
    """
    # #########################################################
    # Check if "stats" exists in adata.uns and handle appropriately
    if "stats" in adata.uns.keys():
        # ##########################################
        # Extract keys starting with "filter" from adata.uns["stats"]
        filter_keys = [k for k in adata.uns["stats"].keys() if k.startswith("filter")]

        # Create a dictionary with only the filtered keys and their corresponding values
        filter_dict = {k: v for k, v in adata.uns["stats"].items() if k in filter_keys}

        # Convert the dictionary into a DataFrame
        filter_df = pd.DataFrame.from_dict(filter_dict, orient="index")

        # Set column names for the DataFrame
        filter_df.columns = ["n_Cells", "n_Genes"]

        # Save the DataFrame to a CSV file
        filter_df.to_csv(save_path)
        # ##########################################
    else:
        # ##########################################
        # Log a warning if "stats" does not exist in adata.uns
        logger.warning("The stats cannot be saved because no 'stats' key exists in adata.uns.keys()")


# ###########################################################################################################
# Plotting Helpers
def calc_group_density(
            group_coords: np.ndarray,
            embedding_coords: np.ndarray
        ) -> np.ndarray:
    """Compute density values for a specific group.
    Helper function to compute density for a specific group.

    Args:
        group_coords (numpy.ndarray): Coordinates of group members.
        embedding_coords (numpy.ndarray): All embedding coordinates
            (e.g., UMAP).

    Returns:
        numpy.ndarray:
            Density values per group point.

    Tags:
        calculation
    """
    # Perform Kernel Density Estimation (KDE) for the group
    if len(group_coords) > 1:  # Ensure there are enough points for KDE
        kde = gaussian_kde(group_coords.T)
        density_values = kde(embedding_coords.T)  # Evaluate density for all coordinates
    else:
        # If there is only one point, assign zero density
        density_values = np.zeros(embedding_coords.shape[0])

    # Min-Max scale the density values between 0 and 1
    scaler = MinMaxScaler()
    density_values_scaled = scaler.fit_transform(density_values.reshape(-1, 1)).flatten()

    return density_values_scaled


def get_rows_cols_figsize(
            n_categories: int,
            ncols: int | None = None,
            nrows: int | None = None,
            base_figsize: Sequence[float | int] = (5.0, 5.0),
            padding: Sequence[float | int] = (0.0, 0.0)
        ) -> tuple[int, int, tuple[float, float]]:
    """Determine optimal subplot grid and compute total figure size.

    Args:
        n_categories: Total number of subplots or categories to display.
        ncols (int | None, optional): Desired number of columns.
            If None, calculated automatically. Defaults to None.
        nrows (int | None, optional): Desired number of rows.
            If None, calculated from ncols and n_categories. Defaults to None.
        base_figsize (Sequence[float | int], optional): Size (width, height) of
            each subplot unit. Defaults to (5.0, 5.0).
        padding (Sequence[float | int], optional):
            Padding (horizontal, vertical) between subplots, in units of
            base_figsize. Defaults to (0.0, 0.0).

    Returns:
        tuple[int, int, tuple[float, float]]:

              - ncols: Computed number of columns.
              - nrows: Computed number of rows.
              - figsize: Final figure size as (width, height).

    Called By:
        plot_density_difference, plot_embedding_density, plot_umap, plot_violin

    Tags:
        utils, visualization
    """
    if ncols is None:
        ncols = max(1, min(5, n_categories // 2))

    # Get number of cols and rows to plot
    ncols = min(n_categories, ncols)
    nrows = int(np.ceil(n_categories / ncols))
    # Calculate the resulting sizes based on the number of plots
    fig_width = max(1, ncols * (base_figsize[0] + padding[0]))
    fig_height = max(1, nrows * (base_figsize[1] + padding[1]))

    figsize = (fig_width, fig_height)

    return ncols, nrows, figsize


def bin_column(
            df: pd.DataFrame,
            input_key: str,
            bins: list[float] | int | None = None,
            labels: list[str] | None = None,
            output_key: str | None = None,
            inplace: bool = True,
            suffix: str = "_binned",
            categorized: bool = True,
        ) -> pd.DataFrame | None:
    """Bin a numerical column in ``adata.obs`` into categorical bins.

    Args:
        df: DataFrame containing the column to be binned.
        input_key (str): The column in the DataFrame to be binned.
        bins (list[float] | int | None, optional): List of bin edges or number
            of bins. If ``None``, ``pd.cut`` will try to infer bins. Defaults to None.
        labels (list[str] | None, optional): Labels for the bins. If ``None``,
            labels will be generated automatically in the format ``lower-upper``
            or ``lower+`` for open-ended bins. Defaults to None.
        output_key (str | None, optional): The name of the output column in
            the DataFrame, if None. Defaults to None.
        inplace (bool, optional): Whether to modify the DataFrame in place. If
            ``False``, returns a new DataFrame containing the original and binned
            columns. Defaults to True.
        suffix (str, optional): the suffix to append to the input_key as output
            key. Defaults to "_binned".
        categorized (str, optional): H5 doesn't support intervals as categories.
            If True, cast it as a categorical string. Defaults to True.

    Returns:
        pandas.DataFrame | None:
            - If ``inplace`` is ``True``: modifies ``df`` in place and returns ``None``.
            - If ``inplace`` is ``False``: returns a new DataFrame with the binned column added.

    .. doctest::

        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {"H-2Kb_log": [2.1, 4.0, 5.8, 6.2, 7.5, 10.0]}
        >>> df = pd.DataFrame(data)
        >>> bin_column(df, input_key="H-2Kb_log",
        ...                bins=[0, 3.5, 5.5, 6, 7, 9, np.inf],
        ...                output_key="output_key")
        >>> df["output_key"].cat.categories
        Index(['0-3.5', '3.5-5.5', '5.5-6', '6-7', '7-9', '9+'], dtype='object')

    Tags:
        annotation, obs, utils
    """
    if bins is None:
        raise ValueError("``bins`` must be provided as a list of bin edges or an integer.")

    # Determine the output key
    if output_key is None:
        output_key = f"{input_key}{suffix}"

    # Generate labels if not provided
    if labels is None and isinstance(bins, list):
        labels = [f"{bins[i]}-{bins[i+1]}" if np.isfinite(bins[i+1]) else f"{bins[i]}+"
                  for i in range(len(bins) - 1)]

    # Perform binning
    binned_data = pd.cut(
        df[input_key], bins=bins, labels=labels, include_lowest=True
        )
    if categorized:
        binned_data = binned_data.astype(str).astype("category")

    if inplace:
        df[output_key] = binned_data
    else:
        df_copy = df.copy()
        df_copy[output_key] = binned_data
        return df_copy


def set_distinct_colors(
            adata: ad.AnnData,
            key: str,
            **kwargs
        ) -> None:
    """Assigns reproducible distinct colors to categories in ``adata.obs[key]``.

    The colors are stored in ``adata.uns[f"{key}_colors"]`` and can be used for
    plotting.

    Args:
        adata (anndata.AnnData): Adata object.
        key (str): Column in ``adata.obs`` whose categories will be color-coded.
        seed (int): Random seed for reproducibility of color assignment.
            If None, colors will be randomly generated without a fixed seed.
        **kwargs: parsed to distinctipy.get_colors()

    Returns:
        None

    Raises:
        KeyError: If ``key`` is not present in ``adata.obs``.

    Calls:
        get_colors_wrapped

    Tags:
        annotation, obs
    """
    if key not in adata.obs:
        raise KeyError(f"'{key}' not found in adata.obs")
    if adata.obs[key].dtype.name != "category":
        raise AttributeError(f"'adata.obs[{key}]' is not categorical")

    categories = pd.Categorical(adata.obs[key]).categories
    n_colors = len(categories)

    colors = [distinctipy.get_hex(c) for c in get_colors_wrapped(n_colors, **kwargs)]
    adata.uns[f"{key}_colors"] = colors


def get_category_color_dict(
            adata: ad.AnnData,
            key: str,
            rng: int = 10,
            pastel_factor: float = 1,
            rerun_colors: bool = True
        ) -> dict[str, str]:
    """
    Retrieve or generate a category-to-color dictionary for a categorical obs
    key in AnnData.

    args:
        adata (anndata.AnnData): Adata object
        key (str): The categorical ``.obs`` key (e.g., 'asma_annotation_final_1')
        rng (int): Number of distinct colors to generate if missing
        pastel_factor (float): Controls color softness when generating
        rerun_colors (bool): If True, regenerate colors if mismatch is detected

    Returns:
        dict[str, str]:
            Mapping from category to hex color

    Tags:
        annotation, obs
    """
    c_key = f"{key}_colors"

    # Check if key exists in .obs
    if key not in adata.obs.keys():
        raise KeyError(f"'{key}' not found in adata.obs")

    # Check if obs[key] is categorical
    if adata.obs[key].dtype.name != "category":
        raise TypeError(f"'{key}' must be a categorical column in adata.obs")

    # Set colors if missing or if rerun is requested due to mismatch
    if c_key not in adata.uns:
        sc.pl.palettes.set_colors_for_categorical_obs(adata, key, num_colors=rng, pastel=bool(pastel_factor))
    cats_ = adata.obs[key].cat.categories.tolist()
    colors_ = adata.uns[c_key]

    if len(cats_) != len(colors_):
        if rerun_colors:
            sc.pl.palettes.set_colors_for_categorical_obs(adata, key, num_colors=rng, pastel=bool(pastel_factor))
            colors_ = adata.uns[c_key]
        else:
            raise ValueError(f"Length mismatch: {len(cats_)} categories vs. {len(colors_)} colors")

    return {k: v for k, v in zip(cats_, colors_)}


# ###########################################################################################################
# Decoupler Helpers and GSE
def our_cons(
            res: dict[str, pd.DataFrame]
        ) -> pd.DataFrame:
    """
    Applies Min-Max Scaling followed by Standard Scaling to the input dictionary
    of DataFrames, aggregates the transformed values, and returns the result.

    This function is designed to handle a dictionary of DataFrames, where each
    DataFrame is scaled using MinMaxScaler followed by StandardScaler. The
    function also handles the specific case where the DataFrame contains values
    equal to -0.0, replacing them with 0.0 to avoid scaling issues. After
    scaling, it aggregates the scaled values from all DataFrames in the
    dictionary and returns the averaged result.

    NOTE:
        The function assumes that all DataFrames in the dictionary have the
        same structure (same columns and indices). This assumption is crucial
        for the correct operation of the code.

    Args:
        res (dict[str, pandas.DataFrame]): A dictionary where keys are strings
            and values are pandas DataFrames. Each DataFrame contains numerical
            values to be scaled and aggregated.

    Returns:
        pandas.DataFrame:
            A DataFrame containing the averaged result of the
            scaled values from all DataFrames in the input dictionary.

    TODO:
        Optimize the code to avoid redundant scaling operations, especially
        when the DataFrames have identical structures.

    Tags:
        calculation, scaling
    """
    # #########################################################
    # Initialize the scalers for scaling operations
    scalar1 = MinMaxScaler()
    scalar = StandardScaler(with_mean=True)

    first = True  # Flag to identify the first iteration
    # #########################################################
    # Iterate over each DataFrame in the input dictionary
    for k in res.keys():
        # ##########################################
        # Handle the specific case where values in DataFrame are equal to -0.0
        if (res[k].values == -0.0).sum() > 1:
            saved = np.zeros_like(res[k].values, dtype=np.float32)
            empty_copy_df = pd.DataFrame(columns=res[k].columns, index=res[k].index, data=saved)
            saved[:, :] = np.where(res[k].values == -0.0, 0.0, res[k].values)
            empty_copy_df.loc[:, :] = saved
            res[k] = empty_copy_df
        # ##########################################
        # Perform scaling and aggregation for the first DataFrame
        if first:
            this_res = res[k].copy()
            this_res.loc[:, :] = scalar.fit_transform(scalar1.fit_transform(res[k].values))
            first = False  # Update the flag after the first iteration
        # ##########################################
        # Perform scaling and aggregation for subsequent DataFrames
        else:
            this_res.loc[:, :] = (this_res.values
                                  + scalar.fit_transform(
                                        scalar1.fit_transform(
                                            res[k].loc[this_res.index, this_res.columns].values)))
    # #########################################################
    # Final aggregation by averaging the results across all DataFrames
    this_res.loc[:, :] = this_res.loc[:, :] / len(res.keys())
    # #########################################################
    return this_res


def search_database(
            combined_database: pd.DataFrame,
            search_term_and: str | list[str] | None = None,
            search_term_or: str | list[str] | None = None,
            search_term_not_and: str | list[str] | None = None,
            search_term_not_or: str | list[str] | None = None,
            lower_search_space_and: bool = True,
            lower_search_space_or: bool = True,
            lower_search_space_not_and: bool = True,
            lower_search_space_not_or: bool = True,
            key: str = "cell_type",
            return_indices: bool = False
        ) -> list:
    """Searches a combined database based on specified search terms.

    Args:
        combined_database (pandas.DataFrame): The combined database to search
            within.
        search_term_and (str | list[str] | None, optional): Terms that must all
            appear in the result. Defaults to None.
        search_term_or (str | list[str] | None, optional): Terms where at least
            one must appear in the result. Defaults to None.
        search_term_not_and (str | list[str] | None, optional): Terms that must
            all be absent from the result. Defaults to None.
        search_term_not_or (str | list[str] | None, optional): Terms where at
            least one must be absent from the result. Defaults to None.
        lower_search_space_and (bool, optional): Whether to lower the search
            space for search_term_and. Defaults to True.
        lower_search_space_or (bool, optional): Whether to lower the search
            space for search_term_or. Defaults to True.
        lower_search_space_not_and (bool, optional): Whether to lower the search
            space for search_term_not_and. Defaults to True.
        lower_search_space_not_or (bool, optional): Whether to lower the search
            space for search_term_not_or. Defaults to True.
        key (str, optional): colum in the database to search within.
            Defaults to "cell_type".
        return_indices (bool, optional): Whether to return Indices instead of
            colum values. Defaults to False.

    Returns:
        list:
            List of entries from the 'key' colum that match the search criteria.

    Examples:
        >>> search_database(combined_database, "astro")
        ['Astrocytes',
         'Astrocyte',
         'Fibrous astrocyte',
         'Pan-reactive astrocyte',
         'Mature astrocyte',
         'Astroglial progenitor cell',
         'Retinal astrocyte',
         'Astrocyte progenitor cell']

        >>> search_database(combined_database, "astro", search_term_not_or=["progenitor", "astroglial"])
        ['Astrocytes',
         'Astrocyte',
         'Fibrous astrocyte',
         'Pan-reactive astrocyte',
         'Mature astrocyte',
         'Retinal astrocyte']

        >>> search_database(combined_database, "astro", search_term_not_and=["progenitor", "astroglial"])
        ['Astrocytes',
         'Astrocyte',
         'Fibrous astrocyte',
         'Pan-reactive astrocyte',
         'Mature astrocyte',
         'Retinal astrocyte',
         'Astrocyte progenitor cell']

        >>> msigdb = sc_utils.get_msigdb_df(organism="mouse")
        >>> sc_utils.search_database(
        ...     msigdb,
        ...     search_term_or=["mitophagy", "pexophagy"],
        ...     search_term_and=["gobp"],
        ...     search_term_not_and=["response"],
        ...     key="GeneSet"
        ... )
        ['GOBP_MITOPHAGY', 'GOBP_PEXOPHAGY']

    Raises:
        ValueError: If both search_term_and and search_term_or are not provided.

    Tags:
        utils
    """
    # #########################################################
    # Validate at least one of search_term_and or search_term_or is provided
    if search_term_and is None and search_term_or is None:
        raise ValueError("At least one of search_term_and or search_term_or must be provided.")
    # #########################################################
    # Initialize result with all unique entries from the specified colum
    res = combined_database[key].unique().tolist()
    # #########################################################
    # Implement search_term_and functionality
    if search_term_and is not None:
        if isinstance(search_term_and, list):
            if len(search_term_and) != 0:
                if lower_search_space_and:
                    res = [x for x in res if all(k in x.lower() for k in search_term_and)]
                else:
                    res = [x for x in res if all(k in x for k in search_term_and)]
        elif isinstance(search_term_and, str):
            if lower_search_space_and:
                res = [x for x in res if search_term_and in x.lower()]
            else:
                res = [x for x in res if search_term_and in x]
    # #########################################################
    # Implement search_term_or functionality
    if search_term_or is not None:
        if isinstance(search_term_or, list):
            if len(search_term_or) != 0:
                temp_res = []
                for k in search_term_or:
                    if lower_search_space_or:
                        temp_res.extend([x for x in combined_database[key].unique() if k in x.lower()])
                    else:
                        temp_res.extend([x for x in combined_database[key].unique() if k in x])
                res = list(set(res).intersection(temp_res))  # Only keep common entries
        elif isinstance(search_term_or, str):
            temp_res = [x for x in combined_database[key].unique() if search_term_or in x.lower()]
            if not lower_search_space_or:
                temp_res = [x for x in combined_database[key].unique() if search_term_or in x]
            res = list(set(res).intersection(temp_res))  # Only keep common entries
    # #########################################################
    # Implement search_term_not_and functionality
    if search_term_not_and is not None:
        if isinstance(search_term_not_and, list):
            if len(search_term_not_and) != 0:
                if lower_search_space_not_and:
                    res = [x for x in res if not all(k in x.lower() for k in search_term_not_and)]
                else:
                    res = [x for x in res if not all(k in x for k in search_term_not_and)]
        elif isinstance(search_term_not_and, str):
            if lower_search_space_not_and:
                res = [x for x in res if search_term_not_and not in x.lower()]
            else:
                res = [x for x in res if search_term_not_and not in x]
    # #########################################################
    # Implement search_term_not_or functionality
    if search_term_not_or is not None:
        if isinstance(search_term_not_or, list):
            if len(search_term_not_or) != 0:
                if lower_search_space_not_or:
                    res = [x for x in res if not any(k in x.lower() for k in search_term_not_or)]
                else:
                    res = [x for x in res if not any(k in x for k in search_term_not_or)]
        elif isinstance(search_term_not_or, str):
            if lower_search_space_not_or:
                res = [x for x in res if search_term_not_or not in x.lower()]
            else:
                res = [x for x in res if search_term_not_or not in x]

    if return_indices:
        return combined_database[key].isin(res)
    else:
        return res


def get_element_and_counts_with_positional_sum(
            data: dict[str, list[Any]] | pd.DataFrame,
            all_data: dict[str, list[Any]] | pd.DataFrame,
            adata: ad.AnnData = None,
            groupby: str | list = ["cell_type", "db", "tissue"],
            key: str = "genesymbol",
            return_list: bool = True,
            sort_columns: list[str] | None = None,
            ascending: list[bool] | None = None,
            gather_total_counts: bool = True,
            add_cell_types: bool = True,
            use_pct_key: str = "pct_log2p",
            min_gs_for_pct: int = 20,
        ) -> list[str] | pd.DataFrame:
    """
    Compute the total occurrences of each element and the sum of their
    positional indices, and sort elements by their total count and summed
    positional frequencies.

    Args:
        data (dict[str, list[Any]] | pd.DataFrame): The input data to process.

            - If a dictionary:
              Keys represent categories, and values are lists of elements.
            - If a pandas DataFrame:
              It must be in melted format, with a grouping column (``groupby``)
              and a column containing elements (``key``).

        all_data (pd.DataFrame or dict): Reference data to compute total counts.
        adata (anndata.AnnData, optional): adata object to add highly
            variable gene rank if provided.
        groupby (str, optional): The column name to group by if ``data``
            is a DataFrame. Defaults to "cell_type".
        key (str, optional): The key/column name containing elements
            if ``data`` is a DataFrame. Defaults to "genesymbol".
        return_list (bool, optional): If True, returns a list of sorted elements.
            If False, returns a dictionary where keys are elements
            and values are tuples (count, positional sum). Defaults to True.
        sort_columns (list | None, optional): Columns to sort by.
            Defaults to ["gs_count", "total_count"] or includes
            "highly_variable_rank" if adata provided.
        ascending (list | None, optional): Sort order per column.
            Defaults to [False, True] or [False, False, False].
        gather_total_counts (bool, optional): Whether to compute total gene
            counts in all_data. Defaults to True.
        add_cell_types (bool, optional): Whether to include a cell_types column.
            Defaults to True.
        use_pct_key (str, optional):
            If sort_columns is None and n_genesets >= min_gs_for_pct,
            this key is inserted as the first element of the default
            sort_columns list. Otherwise, it is ignored.

                - "":            Use the sort_columns only
                - "pct":         gs_count / total_count
                - "pct_log":     gs_count / log2(total_count)
                - "pct_log1p":   gs_count / log1p(total_count)
                - "pct_log2p":   gs_count / log2(total_count + 2)

            Defaults to "pct_log2p".
        min_gs_for_pct (int, optional): Minimum number of gene sets required
            to activate percentage-based sorting via use_pct_key.
            When the observed number of gene sets (n_genesets) is less than
            min_gs_for_pct, sorting defaults to raw count and total_count columns
            instead of pct-based metrics.
            Defaults to 5.

    Returns:
        list[str] | pandas.DataFrame:
            If ``return_list`` is True, returns a list of sorted elements.
            If False, returns a dictionary.
            where keys are elements and values are tuples containing:

                - The total count of occurrences of the element.
                - The sum of positional indices across all lists.

    .. doctest::

        >>> data = {
        >>>     'a': [7, 5, 9, 2],
        >>>     'b': [5, 2, 10, 6],
        >>>     'c': [8, 4, 1, 6, 5]
        >>> }
        >>> result = get_element_and_counts_with_positional_sum(data, return_list=False)
        >>> print(result)
        {5: (3, 4), 2: (2, 5), 6: (2, 7), 7: (1, 0), 8: (1, 0), 4: (1, 1), 9: (1, 2), 1: (1, 2), 10: (1, 2)}
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        >>>     'cell_type': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c'],
        >>>     'genesymbol': [7, 5, 9, 2, 5, 2, 10, 6, 8, 4, 1, 6, 5]
        >>> })
        >>> result = get_element_and_counts_with_positional_sum(df, return_list=False)
        >>> print(result)
        {5: (3, 4), 2: (2, 5), 6: (2, 7), 7: (1, 0), 8: (1, 0), 4: (1, 1), 9: (1, 2), 1: (1, 2), 10: (1, 2)}
        >>> result_list = get_element_and_counts_with_positional_sum(df, return_list=True)
        >>> print(result_list)
        [5, 2, 6, 7, 8, 4, 9, 1, 10]

    Raises:
        NotImplementedError: If ``data`` is not a dictionary or a pandas
            DataFrame.

    Calls:
        validate_groupby_column

    Tags:
        calculation, groupby, obs
    """
    # #########################################################
    # Check if groupby is properly setup
    if isinstance(groupby, str):
        groupby = [groupby]

    for g in groupby:
        validate_groupby_column(
        data, g, check_categorical=False, print_name="data")
    # #########################################################
    if isinstance(data, pd.DataFrame):
        data_dict = data.groupby(groupby)[key].apply(list).to_dict()

    elif isinstance(data, dict):
        data_dict = data
    else:
        raise NotImplementedError("Only a melted pandas DataFrame or dict is valid!")

    # Count occurrences of each element
    element_count = Counter()
    # Sum positional indices of each element
    positional_sum = defaultdict(int)

    for values in data_dict.values():
        for pos, element in enumerate(values):
            element_count[element] += 1
            positional_sum[element] += pos

    if gather_total_counts:
        if isinstance(all_data, dict):
            all_elements = [el for values in all_data.values() for el in values]
            total_occurrences = dict(pd.Series(all_elements).value_counts())
        else:
            total_occurrences = dict(all_data[key].value_counts())
    else:
        total_occurrences = {}

    # Create the dataframe with gene as index element count and positional sum as columns
    # df = pd.DataFrame([element_count, positional_sum], index=['gs_count', 'positional_sum']).T
    df = pd.DataFrame([element_count], index=['gs_count']).T
    df["positional_sum"] = [positional_sum[k] for k in df.index]
    if add_cell_types and isinstance(data, pd.DataFrame):
        grouped = (
            data.groupby(key)[groupby]
            .apply(lambda g: [
                "/".join(map(str, row)) for row in g[groupby].to_numpy().tolist()
            ])
            .to_dict()
        )

        result = {
            k: "|".join(v)
            for k, v in grouped.items()
        }
        df["cell_types"] = result
    # Add the total counts
    df["total_count"] = [total_occurrences.get(x, 0) for x in df.index]
    # Add the percentage for element count
    n_genesets = len(data[groupby].drop_duplicates())
    df["gs_count_pct"] = [round(x * 100, 2) for x in df["gs_count"] / n_genesets]
    df["gene"] = df.index.copy()
    # Reorder the columns
    if add_cell_types and isinstance(data, pd.DataFrame):
        df = df[["gene", "gs_count", "gs_count_pct", "total_count", "cell_types", "positional_sum"]]
    else:
        df = df[["gene", "gs_count", "gs_count_pct", "total_count", "positional_sum"]]
    # #########################################
    # Compute normalized metrics
    df["pct"] = df["gs_count"] / df["total_count"]
    df["pct_log"] = df["gs_count"] / np.log2(df["total_count"])
    df["pct_log1p"] = df["gs_count"] / np.log1p(df["total_count"])
    df["pct_log2p"] = df["gs_count"] / np.log2(df["total_count"] + 2)
    # #########################################
    # sort the data
    if adata is not None:
        if "highly_variable_rank" not in adata.var.columns:
            raise ValueError("Column 'highly_variable_rank' not found in adata.var. Please compute HVGs first.")
        hv_rank = adata.var["highly_variable_rank"]
        df["highly_variable_rank"] = hv_rank.reindex(df.index).values
        if min_gs_for_pct <= n_genesets and use_pct_key:
            if sort_columns is None:
                sort_columns = [use_pct_key, "gs_count", "total_count", "highly_variable_rank"]
            if ascending is None:
                ascending = [False, False, False, False]
        else:
            if sort_columns is None:
                sort_columns = ["gs_count", "total_count", "highly_variable_rank"]
            if ascending is None:
                ascending = [False, False, False]
    else:
        if min_gs_for_pct <= n_genesets and use_pct_key:
            if sort_columns is None:
                sort_columns = [use_pct_key, "gs_count", "total_count"]
            if ascending is None:
                ascending = [False, False, True]
        else:
            if sort_columns is None:
                sort_columns = ["gs_count", "total_count"]
            if ascending is None:
                ascending = [False, True]

    df = df.sort_values(sort_columns, ascending=ascending)

    if return_list:
        return df["gene"].tolist()
    else:
        return df


def mask_gse_by_significance_threshold(
            df_values: pd.DataFrame = None,
            df_sig: pd.DataFrame = None,
            value_obsm_key: str = None,
            sig_obsm_key: str = None,
            adata: ad.AnnData = None,
            threshold: float = 0.001,
            value_to_replace: float = np.nan,
            out_key: str = None,
            remove_na_cols: bool = False,
            col_min_filter: float = 0,
            inplace: bool = True,
            suffix: str = "_sig_only",
        ) -> pd.DataFrame | None:
    """
    Masks values in a DataFrame or AnnData.obsm matrix based on a significance
    threshold.

    Args:
        df_values (pandas.DataFrame, optional): DataFrame of values to mask.
            Defaults to None.
        df_sig (pandas.DataFrame, optional): DataFrame of significance values.
            Defaults to None.
        value_obsm_key (str, optional): Key in adata.obsm for values to mask.
            Defaults to None.
        sig_obsm_key (str, optional): Key in adata.obsm for significance values.
            Defaults to None.
        adata (anndata.AnnData, optional): adata object, required if using obsm
            keys. Defaults to None.
        threshold (float, optional): Significance threshold; values >= this will
            be masked. Defaults to 0.001.
        value_to_replace (float, optional): Value to insert where masking occurs
            (default: np.nan). Defaults to np.nan.
        out_key (str, optional): If provided with adata, stores result in
            adata.obsm[out_key]. Defaults to None.
        remove_na_cols (bool, optional): If True, remove columns that are
            entirely NaN after masking. Defaults to False.
        inplace (bool, optional): If the object should be modified inplace or
            not. Defaults to True.
        suffix (str, optional): The suffix to add to the sig_obsm_key
            Defaults to "_sig_only".

    Returns:
        pandas.DataFrame | None:
            Masked DataFrame if using df_values, else None if writing to adata.

    .. doctest::

        >>> update_adata_with_significance(
        >>>        adata, estimate_key="viper_estimate",
        >>>        pval_key="viper_pvals", pval=.001)

    Called By:
        add_cell_type_idxmax_from_significant, get_proximity_based_score_ranks

    Tags:
        calculation
    """
    use_df = df_values is not None and df_sig is not None
    use_adata = value_obsm_key is not None and sig_obsm_key is not None and adata is not None

    if not (use_df ^ use_adata):
        raise ValueError("Provide either (df_values and df_sig) or (value_obsm_key, sig_obsm_key, and adata)")

    if use_df:
        masked = df_values.copy()
    else:
        masked = adata.obsm[value_obsm_key].copy()
        df_sig = adata.obsm[sig_obsm_key]

    masked[df_sig >= threshold] = value_to_replace

    # Create column mask for final subsetting
    col_mask = np.ones(masked.shape[1], dtype=bool)

    if remove_na_cols:
        col_mask &= ~masked.isna().all(axis=0)
    if col_min_filter > 0:
        valid_pct = 1.0 - masked.isna().mean(axis=0)
        col_mask &= valid_pct >= col_min_filter

    # Apply final column filter
    masked = masked.loc[:, col_mask].copy()

    if use_df or not inplace:
        return masked

    adata.obsm[out_key or f"{value_obsm_key}{suffix}"] = masked


def add_cell_type_idxmax_from_significant(
            adata: ad.AnnData,
            value_obsm_key: str | None = None,
            sig_obsm_key: str | None = None,
            idx_out_key: str | bool | None = None,
            sig_out_key: str | None = None,
            keys_to_use: list | None = None,
            threshold: float = 0.001,
            celltype_min_filter: float = 0,
            suffix: str = "_sig_only",
            fillna: str | float = np.nan
        ) -> None:
    """
    Add idxmax-based cell type annotation from significant estimates to ``obs``.

    This function masks non-significant gene set estimates, then computes the
    column-wise index (cell type) with maximum value per cell (row). The result
    is stored in ``adata.obs`` under the specified key.

    Args:
        adata (anndata.AnnData): Adata object containing ``obsm`` and ``uns``.
        value_obsm_key (str | None, optional): Key in ``obsm`` for estimate
            values. Defaults to None.
        sig_obsm_key (str | None, optional): Key in ``obsm`` for p-values.
            Defaults to None.
        idx_out_key (str | bool | None, optional): Key for results in ``obs``. If
            None, defaults to "cell_type_estimation_idxmax_{value_obsm_key}".
            Defaults to None.
        sig_out_key (str | None, optional): If set, the significant key will be
            kept in the obsm. If True, the default key will be used, check the
            mask_gse_by_significance_threshold doku.
            Defaults to None and is not used if sig_obsm_key is None.
        keys_to_use (list | None, optional): If a list is provided only the
            present keys are used for the idxmax. Defaults to None.
        threshold (float, optional): Significance threshold for masking.
            Defaults to 0.001.
        celltype_min_filter (float, optional): remove cell types that have less
            than celltype_min_filter cells. Defaults to 0.
        suffix (str, optional): The suffix to add to the sig_obsm_key, only used
            if sig_out_key == True Defaults to "_sig_only".
        fillna (str | float, optional): If Nans present, replace them with fillna.
            Defaults to numpy.nan.

    Returns:
        None:
            Modifies ``adata.obs`` in place.

    Calls:
        mask_gse_by_significance_threshold

    Tags:
        annotation, calculation, obs
    """
    if value_obsm_key is None or value_obsm_key not in adata.obsm.keys():
        raise ValueError("You must provide a valid ``value_obsm_key``.")

    if sig_obsm_key:
        if sig_out_key is True:
            sig_out_key = f"{value_obsm_key}{suffix}"
        # mask the geneset by significance
        res_df = mask_gse_by_significance_threshold(
            adata=adata,
            value_obsm_key=value_obsm_key,
            sig_obsm_key=sig_obsm_key,
            threshold=threshold,
            remove_na_cols=True,
            out_key=sig_out_key,
            col_min_filter=celltype_min_filter)
        if res_df is None:
            res_df = adata.obsm[sig_out_key]
    else:
        res_df = adata.obsm[value_obsm_key]

    # The function masks the non significant values with NA and removes the columns,
    # so we must get the remaining keys
    if keys_to_use:
        remaining_keys = np.intersect1d(
            keys_to_use,
            res_df.columns.tolist()).tolist()
        if len(remaining_keys) == 0:
            logger.info("THIS IS NOT POSSIBLE!!!")
            raise ValueError("No remaining cell type keys after filtering - check your thresholds or keys_to_use.")
    else:
        remaining_keys = res_df.columns.tolist()
    # Add the idxmax to the obs
    # What we want to do:
    # adata.obs[f"cell_type_estimation_idxmax_{method}"] = adata.obsm[f"{method}_estimate_sig_only"][
    #     remaining_keys].idxmax(1)
    # For whatever reason the future version is incompetent and we have to deal with it ourselves

    df_sub = res_df[remaining_keys]

    all_na_mask = df_sub.isna().all(axis=1)

    # Prepare a Series to hold the result
    result = pd.Series(index=df_sub.index, dtype=object)

    # For non-all-NA rows, assign idxmax
    result.loc[~all_na_mask] = df_sub.loc[~all_na_mask].idxmax(axis=1)

    # For all-NA rows, assign np.nan (optional, but explicit)
    result.loc[all_na_mask] = fillna
    # print(df_sub.shape)
    adata.obs[idx_out_key or f"{value_obsm_key}_idxmax"] = result.astype("category")


def extract_gene_to_biotype(gtf_path: str) -> dict[str, str]:
    """
    Extracts a mapping from gene_id to gene_biotype from a GTF file.

    Args:
        gtf_path (str): Path to the GTF file (can be gzipped).

    Returns:
        Dict[str, str]: Mapping of gene_id -> gene_biotype.

    Raises:
        FileNotFoundError: If the file cannot be found.
        ValueError: If required fields are missing.
    """
    gene_to_biotype: dict[str, str] = {}
    open_func = gzip.open if gtf_path.endswith(".gz") else open

    try:
        with open_func(gtf_path, "rt") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue
                attr_field = fields[8]
                attr_dict = {}
                for attr in attr_field.split(";"):
                    attr = attr.strip()
                    if attr:
                        key, val = attr.split(" ", 1)
                        attr_dict[key] = val.strip('"')
                gene_id = attr_dict.get("gene_id")
                biotype = (
                    attr_dict.get("gene_biotype")
                    or attr_dict.get("gene_type")
                    or "unknown"
                )
                if gene_id:
                    gene_to_biotype[gene_id] = biotype
    except FileNotFoundError:
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    except Exception as exc:
        raise ValueError(f"Error parsing GTF: {exc}") from exc

    return gene_to_biotype


# ################################################################
# Hierarchical clustering
def get_group_hierarchy(
            adata: ad.AnnData,
            groupby: str | None = None,
            inplace: bool = True,
            layer: str | None = "log2norm_counts"
        ) -> list | None:
    """
    Calculate hierarchical clustering for groups in the groupby in an adata object.

    This function performs hierarchical clustering on the groups identified in
    an adata object, based on the mean expressions of each grouuo. It uses
    Ward's method with Euclidean distance to calculate the clustering.

    NOTE:
        The default groupby=None assumes that the adata object has a "config"
        dictionary under ``.uns`` with the required configuration details.

    Args:
        adata (anndata.AnnData): Adata object.
        key (str, optional): The key in ``adata.obs`` that defines the group
            labels. If None, it uses the key specified in
            ``adata.uns["config"]["general"]["cluster_algorithm"]``.
        inplace (bool, optional): If True, the resulting hierarchical clustering
            is stored within the adata object. If False, the function returns
            the sorted group labels. Default is True. Defaults to True.
        layer (str | None, optional): If specified, use this layer instead of
            ``adata.layers["log2norm_counts"]``. Defaults to "log2norm_counts".

    Returns:
        list | None:
            If ``inplace`` is False, returns a list of sorted cluster labels.
            Otherwise, returns None.

    Raises:
        KeyError: If the required keys are not found in the adata object.

    Calls:
        validate_groupby_column

    Called By:
        cluster_SC_scanpy_like

    Tags:
        clustering, config, groupby, obs
    """
    # #########################################################
    # Check if groupby is properly setup
    if groupby is None:
        config = adata.uns["config"]
        groupby = config["general"]["cluster_algorithm"]

    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    # #########################################################
    # Select matrix to use
    if layer is not None:
        data = adata.layers[layer]
    else:
        logger.warning(
            f"Using adata.X instead of a specified layer {layer}. This may have unintended consequences.")
        data = adata.X
    # #########################################################
    # Calculate mean expressions for each cluster
    unique_labels = adata.obs[groupby].unique()
    cluster_means = []
    for label in unique_labels:
        # ##########################################
        # Extract the rows corresponding to the current label and calculate the mean expression
        rows = adata.obs[groupby] == label
        # Compute mean for these rows; ensure result is 2D
        cluster_mean = data[rows.to_numpy()].mean(axis=0)
        cluster_means.append(np.array(cluster_mean))
    # #########################################################
    # Perform hierarchical clustering on the cluster means
    # Stack the means to form a 2D array for clustering
    cluster_means = np.vstack(cluster_means)
    linkage_ = linkage(cluster_means, method="ward", metric="euclidean")
    # #########################################################
    # Determine the order of clusters based on hierarchical clustering
    ordered_leaves = leaves_list(linkage_)
    sorted_clusters = unique_labels[ordered_leaves].astype(str).tolist()
    # #########################################################
    # Store or return the resulting cluster hierarchy based on the ``inplace`` flag
    if inplace:
        # print("Old categories:", list(adata.obs[groupby].cat.categories))
        # print("new categories:", sorted_clusters)

        # Reorder categories according to sorted_clusters
        adata.obs[groupby] = adata.obs[groupby].cat.reorder_categories(sorted_clusters, ordered=True)

        # Rename them to "0", "1", ..., preserving new order
        new_labels = [str(i) for i in range(len(sorted_clusters))]
        rename_map = dict(zip(sorted_clusters, new_labels))
        adata.obs[groupby] = adata.obs[groupby].cat.rename_categories(rename_map)
    else:
        return sorted_clusters


# ###########################################################################################################
# R Wrappers
def convert_orthologs(
            genes: list[str] | np.ndarray | pd.Series,
            input_species: str,
            output_species: str,
            return_as: str = "dataframe",
            fill_value: float | str | None = np.nan,
            save_path: str = None,
            report_converted_only: bool = False,
            **kwargs
        ) -> pd.DataFrame | list[str] | dict[str, str]:
    """
    Convert gene symbols between species using the R package `orthogene` via
    rpy2.

    This function wraps `orthogene::convert_orthologs()` and supports conversion
    of gene symbols between species such as human and mouse, using the Ensembl
    ortholog database.

    Args:
        genes (list[str] | numpy.ndarray | pandas.Series):
            A list, 1D numpy array, or pandas Series of gene symbols from
            the source species.
        input_species (str):
            Common or scientific name of the source species
            (e.g., "human" or "hsapiens").
        output_species (str):
            Common or scientific name of the target species
            (e.g., "mouse" or "mmusculus").
        return_as (str):
            Output format for the results. Must be one of:

                - "dataframe": a pandas DataFrame with columns
                  `<input_species>_genes` and `<output_species>_genes`
                - "list": a list of converted output genes aligned with
                  the input
                - "dict": a dictionary mapping input to output (only matched
                  genes included)

            Defaults to "dataframe".
        fill_value (float | str | None):
            Value to use for unmatched genes in "list" or "dataframe" mode.
            Can be `np.nan`, a string like `"NOT_FOUND"`, or `None`.
            Defaults to `np.nan`.
        save_path (str | None, default=None):
            Optional path to save the result as a CSV. If None, the result is
            not saved. Defaults to None.
        report_converted_only (bool):
            If True, only returns the matched gene mappings. Unmatched genes
            will be excluded from the result. Defaults to False.
        **kwargs:
            Additional keyword arguments passed to
            `orthogene::convert_orthologs()`.

    Returns:
        pandas.DataFrame | list[str] | dict[str, str]:
            The output in the format specified by `return_as`. See above.

    Raises:
        TypeError:
            If `genes` is not a list, numpy array, or pandas Series.
        ValueError:
            If the cleaned gene list is empty, or if `return_as` is invalid.
        Exception:
            Any R or rpy2-specific errors raised during the conversion process.

    Tags:
        rpy2
    """
    try:
        # rpy2 imports
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects import default_converter, conversion
        # Activate automatic pandas <-> R conversion
        pandas2ri.activate()
        # Import orthogene
        orthogene = importr("orthogene")
    except Exception as _:  # noqa: F841
        raise
    # ###########################################
    # Validate input types
    if isinstance(genes, (list, np.ndarray)):
        genes = pd.Series(genes)
    elif not isinstance(genes, pd.Series):
        raise TypeError("Input genes must be a list, numpy array, or pandas Series.")
    # ###########################################
    # Gene Cleaning
    genes = genes.dropna().astype(str).str.strip()
    if genes.empty:
        raise ValueError("Input gene list is empty after cleaning.")
    # ###########################################
    # Perform conversion
    converted_df = orthogene.convert_orthologs(
        genes.tolist(),
        input_species=input_species,
        output_species=output_species,
        **kwargs)
    # ###########################################
    # Convert to pure python
    with (default_converter + pandas2ri.converter).context():
        converted_df = conversion.get_conversion().rpy2py(converted_df)
    # ###########################################
    # Build a complete dataframe
    input_key = f'{input_species}_genes'
    output_key = f'{output_species}_genes'
    if report_converted_only:
        result_df = pd.DataFrame({
            input_key: converted_df["input_gene"].values,
            output_key: converted_df.index.values})
    else:
        # Build mapping: input -> output
        mapping = dict(zip(converted_df["input_gene"], converted_df.index))
        # Reconstruct output list aligned with original input
        output_genes = [mapping.get(gene, fill_value) for gene in genes]
        # Assemble final DataFrame preserving input order
        result_df = pd.DataFrame({
            input_key: genes,
            output_key: output_genes})
    # ###########################################
    # Opt. Save result
    if save_path:
        result_df.to_csv(save_path, index=False)
    # ###########################################
    # Return result as desired
    if return_as == "dataframe":
        out = result_df
    elif return_as == "dict":
        out = dict(zip(result_df[output_key], result_df[output_key]))
    elif return_as == "list":
        out = result_df[output_key].tolist()
    else:
        raise ValueError("return_as must be one of: 'dataframe', 'dict', 'list', returning dataframe now")

    return out
