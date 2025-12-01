'''Replace inefficient scannpy function.

# Faster and more memory efficient.

%%time
%memit bla = scr.score_genes_efficient(adata, combined_geneset['Tumor_astrocytes'], inplace=False)
peak memory: 1595.01 MiB, increment: 0.00 MiB
CPU times: user 198 ms, sys: 36.9 ms, total: 235 ms
Wall time: 354 ms

%%time
%memit bli = sc.tl.score_genes(adata, combined_geneset['Tumor_astrocytes'], copy=True)

peak memory: 1679.72 MiB, increment: 90.78 MiB
CPU times: user 228 ms, sys: 83.1 ms, total: 311 ms
Wall time: 428 ms

np.allclose(bla.to_numpy(), bli.obs["score"].to_numpy())
True
'''

# #########################################################
# Libraries
from . import (
    np, issparse, pd, spmatrix, ad,
    get_logger)
# #########################################################
# from local files
logger = get_logger(name="sc_replacements")


def _sparse_nanmean(
            data: spmatrix,
            axis: int
        ) -> np.ndarray:
    """np.nanmean equivalent for sparse matrices

    Args:
        data (spmatrix): Sparse input matrix.
        axis (int): Axis along which the mean is computed.

    Returns:
        numpy.ndarray: 
            Dense array with the mean along the specified axis.

    Called By:
        score_genes_efficient

    Tags:
        sparse, utils
    """
    if not issparse(data):
        raise TypeError("data must be a sparse matrix")

    # count the number of nan elements per row/column (dep. on axis)
    nan_mask = data.copy()
    nan_mask.data = np.isnan(nan_mask.data)
    nan_mask.eliminate_zeros()
    n_nan_elements = nan_mask.shape[axis] - nan_mask.sum(axis)

    # set the nans to 0, so that a normal .sum() works
    data_nan_removed = data.copy()
    data_nan_removed.data[np.isnan(data_nan_removed.data)] = 0
    data_nan_removed.eliminate_zeros()

    # the average
    # float64 for score_genes function compatibility)
    s = data_nan_removed.sum(axis, dtype="float64")
    m = s / n_nan_elements

    return m


def score_genes_efficient(
            adata: ad.AnnData,
            gene_list: list[str],
            ctrl_size: int = 50,
            gene_pool: list[str] | None = None,
            n_bins: int = 25,
            score_name: str = "score",
            random_state: int = 0,
            inplace: bool = True,
            dtype: str = "float16"
        ) -> pd.Series | None:
    """Efficiently computes gene set scores by comparing to matched control genes.

    Scores each cell by the average expression of genes in ``gene_list``,
    subtracted by the average expression of a matched control set of genes with
    similar overall expression levels. This approach emulates the Seurat scoring
    strategy using binning of average expression values.

    Args:
        adata (anndata.AnnData): Adata object.
        gene_list (list[str]):
            List of gene names to use for scoring. Must be present in
            ``adata.var_names``.
        ctrl_size (int, optional):
            Number of control genes to sample per expression bin.
            Defaults to 50.
        gene_pool (list[str] | None, optional):
            Pool of genes to choose control genes from. If None, uses all genes
            in ``adata``. Defaults to None.
        n_bins (int, optional):
            Number of expression bins for matching control genes.
            Defaults to 25.
        score_name (str, optional):
            Column name to store scores in ``adata.obs`` if ``inplace=True``.
            Defaults to "score".
        random_state (int, optional):
            Seed for reproducible control gene sampling. Defaults to 0.
        inplace (bool, optional):
            Whether to add the result directly to ``adata.obs``. If False, returns
            a Series. Defaults to True.
        dtype (str, optional):
            Desired NumPy dtype for the output scores. Defaults to "float16".

    Returns:
        pandas.Series | None: 
            If ``inplace=False``, returns a pandas Series with scores indexed by
            ``adata.obs_names``. Otherwise, updates ``adata.obs[score_name]``
            and returns None.

    Raises:
        ValueError: If ``gene_list`` contains no valid genes present in
            ``adata.var_names``.
        ValueError: If no valid control genes are found in ``gene_pool``.

    Calls:
        _sparse_nanmean

    Called By:
        score_genes, score_genes_parallel, score_genes_parallel.func

    Tags:
        annotation, calculation, obs, var
    """
    if random_state is not None:
        np.random.seed(random_state)

    gene_list_in_var = []
    var_names = adata.var_names
    genes_to_ignore = []
    for gene in gene_list:
        if gene in var_names:
            gene_list_in_var.append(gene)
        else:
            genes_to_ignore.append(gene)
    if len(genes_to_ignore) > 0:
        logger.warning(f"genes are not in var_names and ignored: {genes_to_ignore}")
    gene_list = set(gene_list_in_var)

    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")

    if gene_pool is None:
        gene_pool = list(var_names)
    else:
        gene_pool = [x for x in gene_pool if x in var_names]
    if not gene_pool:
        raise ValueError("No valid genes were passed for reference set.")

    # Trying here to match the Seurat approach in scoring cells.
    # Basically we need to compare genes against random genes in a matched
    # interval of expression.

    _adata_subset = (
        adata[:, gene_pool] if len(gene_pool) < len(adata.var_names) else adata)
    if issparse(_adata_subset.X):
        obs_avg = pd.Series(
            np.array(_sparse_nanmean(_adata_subset.X, axis=0)).flatten(),
            index=gene_pool,
        )  # average expression of genes
    else:
        obs_avg = pd.Series(
            np.nanmean(_adata_subset.X, axis=0), index=gene_pool
        )  # average expression of genes

    obs_avg = obs_avg[
        np.isfinite(obs_avg)
    ]  # Sometimes (and I don't know how) missing data may be there, with nansfor

    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    control_genes = set()

    # now pick ``ctrl_size`` genes from every cut
    for cut in np.unique(obs_cut.loc[list(gene_list)]):
        r_genes = np.array(obs_cut[obs_cut == cut].index)
        np.random.shuffle(r_genes)
        # uses full r_genes if ctrl_size > len(r_genes)
        control_genes.update(set(r_genes[:ctrl_size]))

    # To index, we need a list – indexing implies an order.
    control_genes = list(control_genes - gene_list)
    gene_list = list(gene_list)

    X_list = adata[:, gene_list].X
    if issparse(X_list):
        X_list = np.array(_sparse_nanmean(X_list, axis=1)).flatten()
    else:
        X_list = np.nanmean(X_list, axis=1, dtype="float64")

    X_control = adata[:, control_genes].X
    if issparse(X_control):
        X_control = np.array(_sparse_nanmean(X_control, axis=1)).flatten()
    else:
        X_control = np.nanmean(X_control, axis=1, dtype="float64")

    score = X_list - X_control
    pds = pd.Series(
            np.array(score).ravel(), index=adata.obs_names, dtype=dtype,
            name=score_name)
    if inplace:
        adata.obs[score_name] = pds
    else:
        return pds
