
'''
Method VIPER.
Code to run the Virtual Inference of Protein-activity by Enriched Regulon (VIPER) method.
'''


from . import (
    pd, np, nb, csr_matrix, ad, warnings,
    rankdata_numba, parallel_rank, get_logger)


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*IProgress not found.*")
    from tqdm.auto import tqdm  # or any code that triggers the warning
    from decoupler.pre import extract, match, rename_net, get_net_mat, filt_min_n, return_data

from numba_stats import norm

# from tqdm.auto import tqdm

logger = get_logger(name="sc_replacements")


def get_tmp_idxs(pval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts upper triangle p-value pairs and their indices, excluding NaNs.

    Filters the upper triangle of a symmetric p-value matrix to obtain paired
    values (pval[i, j], pval[j, i]) where neither is NaN, and returns both
    the valid pairs and their corresponding (i, j) indices.

    Args:
        pval (numpy.ndarray): A square 2D array of p-values.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple containing:

                - tmp: Array of shape (n_valid, 2) with valid p-value pairs.
                - idxs: Array of shape (n_valid, 2) with corresponding index pairs.

    Raises:
        ValueError: If ``pval`` is not a 2D square array.

    Called By:
        shadow_regulon

    Tags:
        calculation
    """
    # Get indices for upper triangle including the diagonal
    i, j = np.triu_indices_from(pval)

    # Filter out NaN values in both pval[i, j] and pval[j, i]
    mask = ~np.isnan(pval[i, j]) & ~np.isnan(pval[j, i])

    # Collect values and indices
    tmp = np.column_stack((pval[i[mask], j[mask]], pval[j[mask], i[mask]]))
    idxs = np.column_stack((i[mask], j[mask]))

    return tmp, idxs


# ###############################################################################################
@nb.njit(nb.types.Tuple((nb.f4[:, :], nb.i8[:]))(nb.f4[:, :], nb.i8[:, :], nb.f4[:], nb.i8[:], nb.i8), cache=True)
def get_wts_posidxs(
            wts: np.ndarray,
            idxs: np.ndarray,
            pval1: np.ndarray,
            table: np.ndarray,
            penalty: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjusts weights and extracts position indices based on p-value signs and penalties.

    For each index pair, this function determines a direction based on the sign
    of the corresponding p-value. It applies a scaling to one of the vectors in
    ``wts`` according to a penalized adjustment factor. The modified weights
    matrix and selected position indices are returned.

    Args:
        wts (numpy.ndarray):
            A 2D weight matrix of shape (n_samples, n_features), to be updated
            in-place.
        idxs (numpy.ndarray):
            A 2D array of index pairs with shape (n_pairs, 2) indicating feature
            columns.
        pval1 (numpy.ndarray):
            A 1D array of signed p-value differences used to determine update
            direction.
        table (numpy.ndarray):
            A 1D array representing feature usage counts used for normalization.
        penalty (float):
            A scalar penalty term controlling the strength of weight
            adjustments.

    Returns:
        tuple[np.ndarray, np.ndarray]:

            - Updated weight matrix of the same shape as ``wts``.
            - 1D array of selected position indices derived from index pairs.

    Raises:
        IndexError: If ``table`` is indexed out of bounds due to values in ``idxs``.
        ValueError: If shapes of input arrays are incompatible.

    Called By:
        shadow_regulon

    TODO:
        This function modifies ``wts`` in-place, which may lead to unintended side
        effects. Consider making a defensive copy of ``wts`` before performing
        updates.

    Tags:
        calculation
    """
    pos_idxs = np.zeros(idxs.shape[0], dtype=nb.i8)
    for j in nb.prange(idxs.shape[0]):
        p = pval1[j]
        if p > 0:
            x_idx, y_idx = idxs[j]
        else:
            y_idx, x_idx = idxs[j]
        pos_idxs[j] = x_idx

        x = wts[:, x_idx]
        y = wts[:, y_idx]
        x_msk, y_msk = x != 0, y != 0
        msk = x_msk * y_msk
        x[msk] = x[msk] / (1 + np.abs(p))**(penalty/table[x_idx])
        wts[:, x_idx] = x

    return wts, pos_idxs


@nb.njit(nb.f4[:](nb.i8, nb.f4[:, :], nb.i8, nb.f4[:]), cache=True)
def fill_pval_mat(
            j: int,
            reg: np.ndarray,
            n_targets: int,
            s2: np.ndarray
        ) -> np.ndarray:
    """
    Computes contribution scores for all regulators except self,
    for p-value matrix construction.

    For a given target index ``j``, this function computes a per-feature score
    that reflects the weighted contribution of other features (columns in ``reg``)
    to the observed signal ``s2``. Only features with at least ``n_targets``
    non-zero entries are considered. The resulting scores are used to build a
    row in the pairwise p-value matrix.

    Args:
        j (int):
            Index of the current feature to exclude from comparison.
        reg (numpy.ndarray):
            A 2D array of shape (n_samples, n_features) representing a
            regularization matrix.
        n_targets (int):
            Minimum number of non-zero entries required in a column to be
            considered valid.
        s2 (numpy.ndarray):
            A 1D array of length ``n_samples`` representing transformed input
            scores.

    Returns:
        numpy.ndarray:
            A 1D float32 array of length ``n_features`` containing computed
            scores for each regulator, with NaNs for the excluded or invalid
            entries.

    Raises:
        ValueError: If ``reg`` and ``s2`` have mismatched dimensions.

    Called By:
        get_inter_pvals, get_inter_pvals_old

    TODO:
        Handle division by zero in ``np.max(np.abs(reg[:, k]))`` gracefully to
            avoid NaNs or infs.

    Tags:
        calculation
    """
    n_fsets = reg.shape[1]
    col = np.full(n_fsets, np.nan, dtype=nb.f4)
    for k in nb.prange(n_fsets):
        if k != j:
            k_msk = reg[:, k] != 0
            if k_msk.sum() >= n_targets:
                sum1 = np.sum(reg[:, k] * s2)
                ss = np.sign(sum1)
                if ss == 0:
                    ss = 1
                ww = np.abs(reg[:, k]) / np.max(np.abs(reg[:, k]))
                col[k] = np.abs(sum1) / np.sum(np.abs(reg[:, k])) * ss * np.sqrt(np.sum(ww**2))
    return col


def get_inter_pvals(
            nes_i: np.ndarray,
            ss_i: np.ndarray,
            sub_net: np.ndarray,
            n_targets: int
        ) -> np.ndarray:
    """
    Computes pairwise p-values between regulators based on enrichment and scores.

    Constructs a symmetric p-value matrix between regulators by evaluating their
    mutual relationships through a transformed scoring framework. Uses a
    regularization matrix and inverse normal transformations to derive p-values.

    Args:
        nes_i (numpy.ndarray):
            A 1D array of normalized enrichment scores (NES) for each regulator.
        ss_i (numpy.ndarray):
            A 1D array of input sample scores (e.g., expression or signal
            values).
        sub_net (numpy.ndarray):
            A binary or weighted connectivity matrix (n_genes x n_regulators).
        n_targets (int):
            Minimum required number of non-zero targets for a regulator.

    Returns:
        numpy.ndarray:
            A 2D symmetric matrix of p-values (float32) indicating regulatory
            interaction strengths.

    Calls:
        fill_pval_mat, rankdata_numba

    Called By:
        shadow_regulon

    Tags:
        calculation
    """
    n_cols = sub_net.shape[1]
    pval = np.full((n_cols, n_cols), np.nan, dtype=np.float32)

    for j in range(n_cols):
        trgt_msk = sub_net[:, j] != 0

        # Handle regularization matrix
        reg = (sub_net[trgt_msk] != 0) * sub_net[trgt_msk, j][:, None]

        # Ranking and normalization
        s2 = ss_i[trgt_msk].astype(np.float32)
        ranks = rankdata_numba(s2) / (s2.shape[0] + 1) * 2 - 1

        s1 = np.abs(ranks) * 2 - 1
        s1 += (1 - np.max(s1)) / 2
        s1 = norm.ppf(s1 / 2 + 0.5, loc=0, scale=1)

        tmp = np.sign(nes_i[j]) if nes_i[j] != 0 else 1
        s2 = norm.ppf(ranks / 2 + 0.5, loc=0, scale=1) * tmp

        # Fill p-value matrix
        pval[j] = fill_pval_mat(j, reg.astype(np.float32), n_targets, s2.astype(np.float32))

    pval = 1 - norm.cdf(pval, loc=0, scale=1)
    return pval.astype(np.float32)


def get_inter_pvals_old(
            nes_i: np.ndarray,
            ss_i: np.ndarray,
            sub_net: np.ndarray,
            n_targets: int
        ) -> np.ndarray:
    """Legacy version of ``get_inter_pvals`` using a slightly different ranking
    pipeline.

    Computes a matrix of inter-regulator p-values based on transformed
    enrichment and score data. This version uses older logic for ranking and
    transformation.

    Args:
        nes_i (numpy.ndarray):
            A 1D array of enrichment scores per regulator.
        ss_i (numpy.ndarray):
            A 1D array of sample-specific scores.
        sub_net (numpy.ndarray):
            A network matrix (n_genes x n_regulators) of binary or real-valued
            connections.
        n_targets (int):
            Minimum number of connected genes required to compute statistics.

    Returns:
        numpy.ndarray:
            A 2D array of float32 p-values between regulators.

    Calls:
        fill_pval_mat, rankdata_numba

    Tags:
        calculation
    """
    pval = np.full((sub_net.shape[1], sub_net.shape[1]), np.nan, dtype=np.float32)
    for j in range(sub_net.shape[1]):
        trgt_msk = sub_net[:, j] != 0

        reg = ((sub_net[trgt_msk] != 0) * sub_net[trgt_msk, j].reshape(-1, 1)).astype(np.float32)

        s2 = ss_i[trgt_msk]
        s2 = rankdata_numba(s2) / (s2.shape[0]+1) * 2 - 1
        s1 = np.abs(s2) * 2 - 1
        s1 = s1 + (1 - np.max(s1)) / 2
        s1 = norm.ppf(s1/2 + 0.5, loc=0, scale=1)

        tmp = np.sign(nes_i[j])
        if tmp == 0:
            tmp = 1
        s2 = (norm.ppf(s2/2 + 0.5, loc=0, scale=1) * tmp).astype(np.float32)

        pval[j] = fill_pval_mat(j, reg, n_targets, s2)

    pval = 1 - norm.cdf(pval, loc=0, scale=1)

    return pval.astype(np.float32)


def shadow_regulon(
            nes_i: np.ndarray,
            ss_i: np.ndarray,
            net: np.ndarray,
            reg_sign: float = 1.96,
            n_targets: int = 10,
            penalty: int = 20
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Computes a 'shadow regulon' by refining regulatory interactions.

    Identifies significant regulators based on NES and computes interactions
    between them. Refines these interactions using signed p-values and penalties
    to suppress indirect or weak influences.

    Args:
        nes_i (numpy.ndarray):
            A 1D array of enrichment scores per regulator.
        ss_i (numpy.ndarray):
            A 1D array of input scores per gene.
        net (numpy.ndarray):
            Original binary or weighted regulatory network.
        reg_sign (float, optional):
            Threshold for significance in NES values. Defaults to 1.96.
        n_targets (int, optional):
            Minimum number of targets required per regulator. Defaults to 10.
        penalty (int, optional):
            Penalty term controlling update magnitude. Defaults to 20.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None:
            If successful, returns a tuple containing:

                  - sub_net (np.ndarray): Refined sub-network of regulators.
                  - wts (np.ndarray): Corresponding weight matrix.
                  - idxs (np.ndarray): Indices of selected regulators.

              Returns None if fewer than 2 regulators are retained.

    Calls:
        get_inter_pvals, get_tmp_idxs, get_wts_posidxs

    Called By:
        viper

    Tags:
        calculation
    """
    # Find significant activities
    msk_sign = np.abs(nes_i) > reg_sign

    # Filter by significance
    nes_i = nes_i[msk_sign]
    sub_net = net[:, msk_sign]

    # Init likelihood mat
    wts = np.zeros(sub_net.shape, dtype=np.float32)
    wts[sub_net != 0] = 1.0

    if wts.shape[1] < 2:
        return None

    # Get significant interatcions between regulators
    pval = get_inter_pvals(nes_i, ss_i, sub_net, n_targets=n_targets)

    # Get pairs of regulators
    tmp, idxs = get_tmp_idxs(pval)

    if tmp.size == 0:
        return None

    pval1 = np.log10(tmp[:, 1]) - np.log10(tmp[:, 0])
    unique, counts = np.unique(idxs.flatten(), return_counts=True)

    table = np.zeros(unique.max()+1, dtype=np.int64)
    table[unique] = counts

    # Modify interactions based on sign of pval1
    wts, pos_idxs = get_wts_posidxs(wts, idxs, pval1, table, penalty)

    # Select only regulators with positive pval1
    pos_idxs = np.unique(pos_idxs)
    sub_net = sub_net[:, pos_idxs]
    wts = wts[:, pos_idxs]
    idxs = np.where(msk_sign)[0][pos_idxs]

    return sub_net, wts, idxs


def aREA(
            mat: np.ndarray,
            net: np.ndarray,
            wts: np.ndarray | None = None
        ) -> np.ndarray:
    """Compute a normalized enrichment score (NES) based on input data arrays.

    This function calculates a normalized enrichment score (NES) by integrating
    information from the input data (``mat``), a network (``net``), and optional
    weights (``wts``). The process involves normalizing input arrays, ranking
    data, transforming probabilities via the normal quantile function, and
    finally computing the NES.

    Args:
        mat (numpy.ndarray):
            A 1D or 2D numeric array of values to be ranked and transformed.
            If ``mat`` is 1D, it is directly ranked. If 2D, ranking is done
            along the second dimension (i.e., columns).
        net (numpy.ndarray):
            A 2D numeric array representing a network or connectivity matrix.
            Typically, ``net`` should have the same number of columns as ``mat``.
        wts (numpy.ndarray | None, optional):
            A 2D numeric array of weights corresponding to the ``net`` matrix.
            If not provided, a default binary mask will be created
            where non-zero entries in ``net`` are assigned a weight of 1.
            Must have the same shape as ``net`` when provided. Defaults to None.

    Returns:
        numpy.ndarray:
            A 1D array of normalized enrichment scores (NES) for each row subset
            of the input data, considering the weighted network structure.

    Calls:
        parallel_rank, rankdata_numba

    Called By:
        viper

    Tags:
        calculation
    """
    # If no weights are provided, create a default binary weight mask based on ``net``
    if wts is None:
        wts = np.zeros(net.shape)
        wts[net != 0] = 1

    # Normalize ``net`` between -1 and 1, column-wise
    net = net / np.max(np.abs(net), axis=0)
    # Normalize ``wts`` column-wise
    wts = wts / np.max(wts, axis=0)
    # Compute norm factor for ``wts``
    nes = np.sqrt(np.sum(wts**2, axis=0))
    # Normalize ``wts`` by their column-wise sum
    wts = wts / np.sum(wts, axis=0)
    # Rank the values in ``mat``. If ``mat`` is 1D, use ``rankdata_numba``
    # if 2D, use ``parallel_rank``. The ranks are then normalized by ``mat.shape[1] + 1``.
    if len(mat.shape) == 1:
        mat = rankdata_numba(mat) / (mat.shape[1] + 1)
    else:
        mat = parallel_rank(mat, axis=1) / (mat.shape[1] + 1)

    # Transform ranks to a symmetric score around 0.5
    t1 = np.abs(mat - 0.5) * 2
    t1 = t1 + (1 - np.max(t1)) / 2

    # Mask rows based on whether the corresponding row in ``net`` has at least one non-zero entry
    msk = np.sum(net != 0, axis=1) >= 1
    t1, mat = t1[:, msk], mat[:, msk]
    net, wts = net[msk], wts[msk]

    # Convert ranks to normally distributed values via the inverse CDF (ppf)
    t1 = norm.ppf(t1, loc=0, scale=1)
    mat = norm.ppf(mat, loc=0, scale=1)

    # Compute weighted sums
    sum1 = mat.dot(wts * net)
    sum2 = t1.dot((1 - np.abs(net)) * wts)

    # Combine the results to form the NES
    tmp = (np.abs(sum1) + sum2 * (sum2 > 0)) * np.sign(sum1)
    nes = tmp * nes

    return nes


def viper(
            mat: np.ndarray | csr_matrix,
            net: np.ndarray,
            pleiotropy: bool = True,
            reg_sign: float = 0.05,
            n_targets: int = 10,
            penalty: int = 20,
            batch_size: int = 10000,
            verbose: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs Virtual Inference of Protein-activity by Enriched Regulon (VIPER) analysis.

    Computes normalized enrichment scores (NES) for input samples using a
    regulatory network. Optionally applies a pleiotropy correction step to
    refine regulator activity estimates.

    Args:
        mat (numpy.ndarray | csr_matrix):
            Expression or activity matrix of shape (n_samples, n_features).
        net (numpy.ndarray):
            Regulatory network of shape (n_features, n_regulators).
        pleiotropy (bool, optional):
            Whether to apply shadow regulon correction. Defaults to True.
        reg_sign (float, optional):
            Significance threshold for identifying regulators. Defaults to 0.05.
        n_targets (int, optional):
            Minimum number of target genes per regulator. Defaults to 10.
        penalty (int, optional):
            Penalty parameter for adjusting weights. Defaults to 20.
        batch_size (int, optional):
            Number of samples per batch if input is sparse. Defaults to 10000.
        verbose (bool, optional):
            Whether to print progress messages. Defaults to False.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:

              - nes (np.ndarray): Matrix of normalized enrichment scores
               (n_samples x n_regulators).
              - pvals (np.ndarray): Two-tailed p-value matrix of the same shape
                as ``nes``.

    Raises:
        ValueError: If input dimensions are inconsistent.

    Calls:
        aREA, shadow_regulon

    Called By:
        run_viper

    Tags:
        annotation, calculation
    """
    # Get number of batches
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape
    n_batches = int(np.ceil(n_samples / batch_size))

    if verbose:
        print('Infering activities on {0} batches.'.format(n_batches))

    if isinstance(mat, csr_matrix):
        # Init empty acts
        n_batches = int(np.ceil(n_samples / batch_size))
        nes = np.zeros((n_samples, n_fsets), dtype=np.float32)
        # for i in tqdm(range(n_batches), disable=not verbose):
        for i in range(n_batches):
            # Subset batch
            srt, end = i*batch_size, i*batch_size+batch_size
            tmp = mat[srt:end].toarray()

            # Compute VIPER for batch
            nes[srt:end] = aREA(tmp, net)
    else:
        # Compute VIPER for all
        nes = aREA(mat, net)

    if pleiotropy:
        if verbose:
            print('Computing pleiotropy correction.')

        reg_sign = norm.ppf(1 - (reg_sign / 2), loc=0, scale=1)

        # First loop: compute shadow regulons
        shadows = [None] * nes.shape[0]
        ss_values = [None] * nes.shape[0]

        # for i in tqdm(range(nes.shape[0]), disable=not verbose):
        for i in range(nes.shape[0]):
            # Extract per sample
            if isinstance(mat, csr_matrix):
                ss_i = mat[i].toarray()[0]
            else:
                ss_i = mat[i]
            ss_values[i] = ss_i
            nes_i = nes[i]

            # Compute shadow regulons
            shadow = shadow_regulon(nes_i, ss_i, net, reg_sign=reg_sign, n_targets=n_targets, penalty=penalty)
            shadows[i] = shadow

        # Second loop: recompute activity with shadow regulons and update nes
        # for i in tqdm(range(nes.shape[0]), disable=not verbose):
        for i in range(nes.shape[0]):
            shadow = shadows[i]
            if shadow is None:
                continue
            else:
                sub_net, wts, idxs = shadow
                ss_i = ss_values[i]

                # Recompute activity and update nes
                tmp = aREA(ss_i.reshape(1, -1), sub_net, wts=wts)[0]
                nes[i, idxs] = tmp

    # Get pvalues
    pvals = norm.cdf(-np.abs(nes), loc=0, scale=1) * 2

    return nes, pvals


def run_viper(
            mat: pd.DataFrame,
            net: pd.DataFrame,
            source: str = 'source',
            target: str = 'target',
            weight: str = 'weight',
            pleiotropy: bool = True,
            reg_sign: float = 0.05,
            n_targets: int = 10,
            penalty: int = 20,
            batch_size: int = 10000,
            min_n: int = 5,
            verbose: bool = False,
            use_raw: bool = False
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Virtual Inference of Protein-activity by Enriched Regulon (VIPER).

    VIPER (Alvarez et al., 2016) estimates biological activities by performing a
    three-tailed enrichment score calculation. For further information check the
    supplementary information of the decoupler manuscript or the original
    publication.

    Alvarez M.J.et al. (2016) Functional characterization of somatic mutations
    in cancer using network-based inference of protein activity. Nat. Genet.,
    48, 838â€“847.

    Estimates biological activities using a three-tailed enrichment score
    calculation. If ``mat`` is adata, results are stored in
    ``.obsm['viper_estimate']`` and ``.obsm['viper_pvals']``.

    Args:
        mat (pandas.DataFrame): Input matrix, list of [features, matrix],
            or adata object.
        net (pandas.DataFrame): Network in long format with regulatory
            interactions.
        source (str, optional): Column in ``net`` for source nodes.
            Defaults to 'source'.
        target (str, optional): Column in ``net`` for target nodes.
            Defaults to 'target'.
        weight (str, optional): Column in ``net`` for edge weights.
            Defaults to 'weight'.
        pleiotropy (bool, optional): Whether to apply correction for pleiotropic
            regulators. Defaults to True.
        reg_sign (float, optional): P-value threshold for significant
            regulators. Defaults to 0.05.
        n_targets (int, optional): Minimum number of overlapping targets
            required. Defaults to 10.
        penalty (int, optional): Penalty factor for pleiotropic interactions (>1
            applies penalty). Defaults to 20.
        batch_size (int, optional): Size of matrix batches for computation.
            Defaults to 10000.
        min_n (int, optional): Minimum targets per source for inclusion.
            Defaults to 5.
        verbose (bool, optional): Whether to show progress messages.
            Defaults to False.
        use_raw (bool, optional): Whether to use ``.raw.X`` in adata, if
            available. Defaults to False.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:
              tuple[pd.DataFrame, pd.DataFrame] (estimate, p-values) matrices.

    Calls:
        viper

    Called By:
        run_decoupler

    Tags:
        annotation, calculation
    """
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        print('Running viper on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0],
                                                                                              len(c), net.shape[1]))

    # Run VIPER
    estimate, pvals = viper(m, net, pleiotropy=pleiotropy, reg_sign=reg_sign, n_targets=n_targets, penalty=penalty,
                            batch_size=batch_size, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'viper_estimate'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'viper_pvals'

    return return_data(mat=mat, results=(estimate, pvals))


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
    Runs the Decoupler tool to analyze and visualize cell type-specific
    marker expression.

    The function applies various decoupling methods on single-cell data provided
    in ``adata``, using predefined ref/markers and the specified run type. It
    allows for several customization options, including visualization, adjusting
    for specific layers, and appending cell type information to the dataset.

    NOTE:
        The parameters are set up for human T-cells, and you would have to
        change this manually, because the databases didn't provide automatic
        subsetting.

    Args:
        adata (anndata.AnnData): Adata object.
        ref (dict[str): Marker genes to use in the decoupling analysis.
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
        run_viper(
            mat=adata, net=ref, source=source, target=target, weight=weight,
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
