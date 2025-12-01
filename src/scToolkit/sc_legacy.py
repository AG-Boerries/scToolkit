"""This Module might some day be used for legacy purposes"""
from . import (
    np, ad, Sequence, # pd, nb, csr_matrix,
    # rankdata_numba, parallel_rank,
    replace_np_array_with_list_recursive, copy, json_dump,
    sc, get_config, update_nested_dict, get_adata_sub_keys,
    min_max_scale_axis, Normalize, plt, Wedge, lk_legend, colorart, vstack,
    validate_groupby_column)


def fix_old_dict(
            adata: ad.AnnData | None = None,
            config: dict | None = None,
            inplace: bool = False,
        ) -> dict:
    """
    Updates legacy configuration dictionaries to comply with new format
    expectations.

    !!! WARNING: This function is incomplete and not safe for use.

    This function converts outdated ``config`` structures by wrapping
    ``keys_for_qc`` under the ``obs`` key and optionally generating ``var`` keys
    based on available ``adata.var`` fields. If ``inplace=True``, the updated config
    is returned; otherwise, it is written to ``adata.uns["config"]``.

    Args:
        adata (anndata.AnnData | None, optional):
            adata object containing ``.uns["config"]`` and ``.var`` attributes.
            Used for inference and updating of config keys. Defaults to None.
        config (dict | None, optional):
            Configuration dictionary to be migrated. If not provided, falls back
            to ``adata.uns["config"]`` if ``adata`` is given. Defaults to None.
        inplace (bool, optional):
            If True, returns the updated config dictionary. If False, writes the
            result back into ``adata.uns["config"]``. Defaults to False.

    Returns:
        dict:
            The updated configuration dictionary.

    Raises:
        ValueError: If neither ``adata`` nor ``config`` is provided.
        ValueError: If both ``adata`` and ``config`` are None and fallback fails.

    TODO:
        - Finalize all migration logic and safeguard assumptions about config
          structure.
        - Fix incorrect logic where ``config = adata.uns["config"].copy()`` is
          called even when ``adata`` is None.
        - Improve error messaging and handling of malformed config structure.
        - Add unit tests for edge cases and invalid input scenarios.

    Tags:
        config
    """
    # #####################################
    # Setup which config to use
    if adata is None:
        if config is None:
            config = adata.uns["config"].copy()
        else:
            raise ValueError("You have to provide a adata OR config!")
    elif config is None:
        raise ValueError("You have to provide a adata or config!")
    else:
        # Detach config from the provided one
        config = config.copy()
    # #####################################
    # Updated

    # FIx the update of the keys_for_qc
    if "obs" not in config["general"]["keys_for_qc"].keys():
        old_keys = config["general"]["keys_for_qc"].copy()
        config["general"]["keys_for_qc"] = {}
        config["general"]["keys_for_qc"]["obs"] = old_keys
        if adata is not None:
            config["general"]["keys_for_qc"]["var"] = np.intersect1d(
                        ["n_cells", "n_counts", "n_unique",
                         # , "means" "variances", "variances_norm"],  # Not so informative
                         ],
                        adata.var.keys())
    # #####################################
    # Overwrite the adata if provided, else return config
    if inplace:
        return config

    adata.uns["config"] = config


def save_h5ad_old(
            adata: ad.AnnData,
            path_to_h5ad: str,
            path_to_config: str,
            compression: str = "gzip",
            save_umap: str = ""
        ) -> None:
    """Saves a adata object with experiment_dict.

    DEPRECATED: This is only for legacy purposes (which only should be used to
        have continue old projects, which none of you have except for Ariane and me!!!)

    NOTE:
        At the moment this also creates a config.json because h5, doesn't
        support None values and lists. it would remove the dict keys that are
        equals to None and convert the lists to numpy.ndarray's so we use json
        for now.

    Args:
        adata (anndata.AnnData): Adata object with with experiment_dict in
            adata.uns["config"]
        path_to_h5ad (str): Path to save the adata object.
        path_to_config (str): Path to save the config as json.
        compression (str, optional): The Compression to use for h5ad, Defaults
            to "gzip" Defaults to "gzip".

    Returns:
        None

    Calls:
        replace_np_array_with_list_recursive

    TODO:
        Check for each version of h5 if they support this finally,
        otherwise write a wrapper.

    Tags:
        config, io
    """
    # ########################################################################
    # Here we save and remove the umap object before saving the adata object
    umap_key = adata.uns["config"]["general"]["save_umap_object"]["key"]
    if (
            len(save_umap) != 0
            and umap_key in adata.uns.keys()):
        import pickle
        import lzma

        with lzma.open(save_umap, "wb") as f:
            pickle.dump(adata.uns[umap_key], f)
    # ########################################################################
    # Backup the config object, to replace it again after the changes for saving
    adata.uns["config"] = replace_np_array_with_list_recursive(
            adata.uns["config"])
    this_experiment_dict = copy(adata.uns["config"])
    # ###################
    # To save the object convert the config to a normal dictionary
    adata.uns["config"] = {k: v for k, v in adata.uns["config"].items()}
    # ###################
    # save the config as json, this ensures no changes made to the config
    with open(path_to_config, 'w') as fp:
        json_dump(adata.uns["config"], fp)
    # ###################
    # then save the object
    adata.write_h5ad(path_to_h5ad, compression=compression)
    # ####################
    # Reset the config to experiment_dict again
    adata.uns["config"] = this_experiment_dict


def load_h5ad_old(
            path_to_h5ad: str,
            path_to_config: str
        ) -> ad.AnnData:
    """Loads a adata object with experiment_dict.

    DEPRECATED: This is only for legacy purposes to be able to load the .json
        configs

    Args:
        path_to_h5ad (str): Path to load the adata object from.
        path_to_config (str): Path to load the config json from.

    Returns:
        anndata.AnnData:
            Adata object with with experiment_dict in
            adata.uns["config"]

    Calls:
        get_config, update_nested_dict

    Tags:
        config, io
    """
    from json import load
    # To load it use
    adata = sc.read_h5ad(path_to_h5ad)
    # Create a new sc_experiment config with the same params as the old one.
    config = get_config(
            param_setup=adata.uns["config"]["param_setup"],
            dataset_name=adata.uns["config"]["dataset_name"],
            save_path=adata.uns["config"]["general"]["save_path"],
            use_GPU=adata.uns["config"]["general"]["use_GPU"])
    # ########################
    # Load the correct config
    with open(path_to_config, 'r') as fp:
        config_old = load(fp)
    # ########################
    # Update the experiment_dict
    config = update_nested_dict(config, config_old)
    # config.update({k: v for k, v in adata.uns["config"].items()})
    # ########################
    # Replace the config with the same but experiment_dict
    adata.uns["config"] = config
    # ########################
    # Return the object
    return adata


def plot_split_dotplot_mpl(
            adata: ad.AnnData,
            keys: list[str],
            groupby: str,
            condition_obs_key: str,
            vs: tuple[str, str],
            expression_cutoff: float = 0.0,
            dot_min: float = 0.0,
            dot_max: float = 0.5,
            smallest_dot: float = 0.0,
            largest_dot: float = 0.4,
            cmap: str = "turbo",
            figsize:  Sequence[float | int] = (12, 8),
            title: str | None = None,
            obsm_keys: list[str] | None = None,
            varm_keys: list[str] | None = None,
            minmax_scale_axis: int | None = None,
            **extract_kwargs
        ) -> None:
    """Plots a split dot plot from an adata object.

    Each dot represents a gene expression value in a specific group:
    - The dot size indicates the fraction of cells expressing the gene.
    - The dot color represents the mean expression across expressing cells.
    - Left half-circle = Condition 1
    - Right half-circle = Condition 2

    Args:
        adata (anndata.AnnData): Adata object.
        keys (list[str]): List of marker genes to visualize.
        groupby (str): Column(s) in ``adata.obs`` defining the grouping.
        condition_obs_key (str):
            Column in ``adata.obs`` specifying the experimental condition.
        vs (tuple[str):
            A pair of conditions to compare (e.g., ``("condition1",
            "condition2")``).
        expression_cutoff (float, optional):
            Minimum expression level to consider a gene as expressed.
            Defaults to 0.0.
        dot_min (float, optional):
            Minimum fraction of cells expressing a gene to be displayed.
            Defaults to 0.0.
        dot_max (float, optional):
            Maximum fraction of cells expressing a gene to be displayed.
            Defaults to 0.5.
        smallest_dot (float, optional):
            Minimum size of dots in the plot. Defaults to 0.0.
        largest_dot (float, optional):
            Maximum size of dots in the plot. Defaults to 0.4.
        cmap (str, optional):
            Colormap for the mean expression values. Defaults to "turbo".
        figsize (Sequence[float | int], optional):
            Size of the figure ``(width, height)``. Defaults to (12, 8).
        title (str or None, optional):
            Title of the plot. If None, it defaults to ``"cond1 vs cond2"``.
            Defaults to None.
        obsm_keys (list of str or None, optional):
            Keys in ``adata.obsm`` containing precomputed expression values.
            Defaults to None.
        varm_keys (list of str or None, optional):
            Keys in ``adata.varm`` containing precomputed expression values.
            Defaults to None.
        minmax_scale_axis (int or None, optional):
            If any axis should be scaled between 0 and 1, 0 = columns. Defaults
            to None.
        **extract_kwargs: Additional keyword arguments forwarded to
            ``sc_utils.get_adata_sub_keys``.

    Returns:
        None:
            The function generates and displays a split dot plot.

    Raises:
        AssertionError: If required columns are missing from ``adata.obs``.
        AssertionError: If ``vs`` does not contain exactly two categories.
        AssertionError: If specified conditions are not found in
            ``condition_obs_key``.

    Calls:
        get_adata_sub_keys, min_max_scale_axis, validate_groupby_column

    Tags:
        groupby, obs, var, visualization
    """
    # #########################################################
    # Check if groupby is properly setup
    validate_groupby_column(
        adata.obs, groupby, check_categorical=True)
    validate_groupby_column(
        adata.obs, condition_obs_key, check_categorical=True, groups=vs)
    # Check if it is realy 2
    if not isinstance(vs, tuple) or len(vs) != 2:
        raise ValueError(
            "'vs' must be a tuple of exactly two condition labels.")
    # #########################################################
    # setup the rest
    assert len(vs) == 2, "Argument 'vs' must specify exactly two categories."
    cond1, cond2 = vs
    if title is None:
        title = f"{cond1} vs {cond2}"

    if len(keys) == 0:
        raise AttributeError("The keys are empty!")

    df_expr = get_adata_sub_keys(
        adata, keys, obsm_keys, varm_keys,
        groupby=[condition_obs_key] + [groupby],
        **extract_kwargs)
    # #########################################################
    assert cond1 in df_expr[condition_obs_key].unique(), f"'{cond1}' not found in column '{condition_obs_key}'."
    assert cond2 in df_expr[condition_obs_key].unique(), f"'{cond2}' not found in column '{condition_obs_key}'."

    df_c1 = df_expr[df_expr[condition_obs_key] == cond1].copy()
    df_c2 = df_expr[df_expr[condition_obs_key] == cond2].copy()

    frac_c1 = df_c1.groupby(groupby, observed=True)[keys].agg(lambda x: (x > expression_cutoff).sum() / len(x))
    frac_c2 = df_c2.groupby(groupby, observed=True)[keys].agg(lambda x: (x > expression_cutoff).sum() / len(x))

    mean_c1 = df_c1.groupby(groupby, observed=True)[keys].agg(lambda x: x.mask(x <= expression_cutoff).mean())
    mean_c2 = df_c2.groupby(groupby, observed=True)[keys].agg(lambda x: x.mask(x <= expression_cutoff).mean())

    if minmax_scale_axis is not None:
        min_max_scale_axis(data=None, data_list=[mean_c1, mean_c2], axis=minmax_scale_axis, inplace=True)

    group_names = frac_c1.index.tolist()

    norm = Normalize(vmin=min(mean_c1.min().min(), mean_c2.min().min()),
                     vmax=max(mean_c1.max().max(), mean_c2.max().max()))
    color_map = plt.colormaps.get_cmap(cmap)

    if dot_max is None:
        dot_max = max(frac_c1.max().max(), frac_c2.max().max())

    frac_c1 = np.clip(frac_c1, dot_min, dot_max)
    frac_c2 = np.clip(frac_c2, dot_min, dot_max)

    frac_c1 = (frac_c1 - dot_min) / (dot_max - dot_min)
    frac_c2 = (frac_c2 - dot_min) / (dot_max - dot_min)

    size_c1 = frac_c1 * (largest_dot - smallest_dot) + smallest_dot
    size_c2 = frac_c2 * (largest_dot - smallest_dot) + smallest_dot

    _, ax = plt.subplots(figsize=figsize)

    y_positions = []
    for col_idx, feature in enumerate(keys):
        for row_idx, group in enumerate(group_names):
            size1 = size_c1.loc[group, feature]
            size2 = size_c2.loc[group, feature]
            color1 = color_map(norm(mean_c1.loc[group, feature]))[:3]
            color2 = color_map(norm(mean_c2.loc[group, feature]))[:3]

            size1 = max(size1, smallest_dot)
            size2 = max(size2, smallest_dot)

            x_pos = col_idx
            y_pos = row_idx
            y_positions.append(y_pos)

            radius1 = size1
            radius2 = size2

            ax.add_patch(Wedge((x_pos, y_pos), radius1, 90, 270, facecolor=color1, edgecolor="black", linewidth=0.5))
            ax.add_patch(Wedge((x_pos, y_pos), radius2, 270, 90, facecolor=color2, edgecolor="black", linewidth=0.5))

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)

    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)

    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)

    ax.set_aspect("equal")
    if title is not False:
        ax.set_title(title)

    legend_fractions = np.linspace(dot_min, dot_max, 5)
    legend_sizes = legend_fractions * (largest_dot - smallest_dot) + smallest_dot
    legend_radii = [x * 100 for x in legend_sizes]

    size_legend = lk_legend(
        legend_items=[("circle", f"{int(f * 100)}%", {"markersize": r})
                      for f, r in zip(np.linspace(0, 1, 5), legend_radii)],
        title="Fraction of cells",
        alignment="left",
        handletextpad=1)

    color_legend = colorart(
        ax=ax, norm=norm, cmap=cmap,
        title="Mean Expression")
    ax.grid(False)
    vstack([size_legend, color_legend], spacing=20,
           alignment="left", frameon=True, ax=ax, padding=2,
           loc="upper left", bbox_to_anchor=(1.02, 1), bbox_transform=ax.transAxes)

    plt.show()
