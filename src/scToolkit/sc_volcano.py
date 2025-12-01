

from scToolkit import (
    np, pd, plt, cdist, Any,
    linear_sum_assignment, get_rows_cols_figsize)

##############################################################################################
# Pure helpers
def _validate_deg_input(deg_df: pd.DataFrame) -> None:
    """
    Validate that the differential expression DataFrame contains
    all required columns for volcano plotting.

    Args:
        deg_df (pandas.DataFrame): Differential expression DataFrame
            containing DEG statistics.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    # Verify the presence of all mandatory columns.
    required_columns = {
        "group", "reference", "names", "scores",
        "logfoldchanges", "pvals", "pvals_adj",
        "pct_nz_group", "pct_nz_reference",
    }
    missing_columns = required_columns - set(deg_df.columns)
    if missing_columns:
        msg = f"deg_df missing required columns: {missing_columns}"
        raise ValueError(msg)


def _resolve_groups(
            deg_df: pd.DataFrame,
            group: str | list[str] | None
        ) -> list[str]:
    """
    Resolve which groups from the DEG DataFrame should be plotted.

    Args:
        deg_df (pandas.DataFrame): Differential expression DataFrame.
        group (str | list[str] | None): Name(s) of group(s) to include.
            If None, all groups are used.

    Returns:
        list[str]: List of resolved group names to plot.

    Raises:
        TypeError: If group is not None, str, or list-like.
    """
    # Normalize the input group argument to a string list.
    if group is None:
        return deg_df["group"].unique().tolist()
    if isinstance(group, str):
        return [group]
    if isinstance(group, (list, tuple, set)):
        return list(group)
    msg = "group must be None, str, or list[str]"
    raise TypeError(msg)


def _subset_by_genes(
            deg_df: pd.DataFrame,
            genes: str | list[str] | None
        ) -> pd.DataFrame:
    """
    Subset a DEG DataFrame to include only specified genes.

    Args:
        deg_df (pandas.DataFrame): Differential expression DataFrame.
        genes (str | list[str] | None): Gene name(s) to include.
            If None, the full DataFrame is returned.

    Returns:
        pd.DataFrame: Subset of deg_df containing only selected genes.
    """
    # Filter rows based on gene names, if provided.
    if genes is None:
        return deg_df
    if isinstance(genes, str):
        genes = [genes]
    return deg_df[deg_df["names"].isin(genes)].copy()

import pandas as pd


def _resolve_sort_column(
            deg_df: pd.DataFrame,
            sort_by: str
        ) -> pd.Series:
    """
    Resolve and return the Series used for sorting, supporting
    an optional 'abs_' prefix to sort by absolute values.

    Args:
        deg_df (pandas.DataFrame): Differential expression DataFrame.
        sort_by (str): Column name to sort by. May include the prefix
            'abs_' to sort by absolute values.

    Returns:
        pd.Series: Column data prepared for sorting.

    Raises:
        ValueError: If the specified column does not exist in deg_df.
    """
    # Check if sorting requires absolute values and extract column.
    use_absolute = sort_by.startswith("abs_")
    column_name = sort_by[4:] if use_absolute else sort_by

    if column_name not in deg_df.columns:
        msg = f"Invalid sort column '{column_name}' not found in DataFrame."
        raise ValueError(msg)

    sort_series = deg_df[column_name].astype(float)
    return sort_series.abs() if use_absolute else sort_series


def _select_top_genes(
            deg_df: pd.DataFrame,
            sort_by: str,
            top_n: int | None,
            balanced: bool
        ) -> pd.DataFrame:
    """
    Select top genes from a DEG DataFrame using sorting and balance options.

    Args:
        deg_df (pandas.DataFrame): Differential expression DataFrame.
        sort_by (str): Column name to sort by (supports 'abs_' prefix).
        top_n (int | None): Number of top genes to return. If None, all are
            returned in sorted order.
        balanced (bool): If True, selects top_n/2 smallest and largest
            values symmetrically.

    Returns:
        pd.DataFrame: Subset of deg_df with top-ranked genes.

    Raises:
        ValueError: If sort_by column does not exist.
    """
    # #########################################################
    # Prepare sort column and determine ascending direction.
    df_sorted = deg_df.copy()
    df_sorted["sort_col"] = _resolve_sort_column(df_sorted, sort_by)
    ascending = sort_by in {"pvals_adj", "pct_nz_reference", "pct_nz_group"}
    df_sorted = df_sorted.sort_values("sort_col", ascending=ascending)
    # #########################################################
    # Return early if no top_n limit specified.
    if top_n is None:
        return df_sorted
    # #########################################################
    # Select either balanced or unbalanced top subsets.
    if balanced:
        half_n = top_n // 2
        top_half = df_sorted.head(half_n)
        bottom_half = df_sorted.tail(half_n)
        return pd.concat([top_half, bottom_half])

    return df_sorted.head(top_n)


##############################################################################################
# Lame functions
def lame_param_quadrant(
            theta: np.ndarray,
            radius_x: float,
            radius_y: float,
            exponent_p: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Lamé (superellipse) coordinates for the first quadrant.

    The function evaluates the Lamé curve defined by:
        |x / radius_x|**p + |y / radius_y|**p = 1
    for θ in [0, π/2]. It avoids sign-based numerical instability by
    restricting evaluation to positive sine and cosine values.

    Args:
        theta (numpy.ndarray): Angles in radians. Expected range [0, π/2].
        radius_x (float): Semi-axis length in the x-direction.
        radius_y (float): Semi-axis length in the y-direction.
        exponent_p (float): Shape exponent. 2=circle, >2=squared shape.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates.

    Raises:
        ValueError: If any of the input parameters are non-positive.
    """
    if radius_x <= 0 or radius_y <= 0 or exponent_p <= 0:
        msg = "radius_x, radius_y, and exponent_p must be positive values."
        raise ValueError(msg)

    exponent_factor = 2 / exponent_p
    x_coords = radius_x * (np.cos(theta) ** exponent_factor)
    y_coords = radius_y * (np.sin(theta) ** exponent_factor)
    return x_coords, y_coords


def stack_angles_lame_even_y(
            genes: list[tuple[int, str]],
            radius_x: float,
            radius_y: float,
            exponent_p: float,
            side: str,
            axis: plt.Axes,
            renderer: Any,
            font_kwargs: dict[str, Any],
            spacing_factor: float = 1.05
        ) -> np.ndarray:
    """
    Compute Lamé curve anchor angles with even vertical spacing.

    The function determines equally spaced anchor points along a Lamé
    curve in the vertical (y) direction, starting just below the curve
    apex and proceeding downward by one label height per step.

    Args:
        genes (list[tuple[int, str]]): List of (index, gene_name) tuples.
        radius_x (float): Semi-axis length in x-direction.
        radius_y (float): Semi-axis length in y-direction.
        exponent_p (float): Lamé exponent controlling curvature shape.
        side (str): Either "L" or "R" to mirror left/right.
        axis (plt.Axes): Matplotlib axis for text sizing and transforms.
        renderer (Any): Renderer instance for text size calculation.
        font_kwargs (dict[str, Any]): Font parameters for measuring label size.
        spacing_factor (float, optional): Label vertical spacing multiplier.
            Defaults to 1.05.

    Returns:
        np.ndarray: Array of θ angles corresponding to anchor positions.

    Raises:
        ValueError: If radius_x, radius_y, or exponent_p are non-positive.
    """
    if radius_x <= 0 or radius_y <= 0 or exponent_p <= 0:
        msg = "radius_x, radius_y, and exponent_p must be positive."
        raise ValueError(msg)
    if not genes:
        return np.array([])

    num_labels = len(genes)
    exponent_factor = 2 / exponent_p

    # Dense first-quadrant sampling
    theta_dense = np.linspace(0, np.pi / 2, 4000)
    x_dense = radius_x * (np.cos(theta_dense) ** exponent_factor)
    y_dense = radius_y * (np.sin(theta_dense) ** exponent_factor)

    # Measure label height in display pixels
    temp_labels = [axis.text(0, 0, g[1], **font_kwargs) for g in genes]
    heights_px = np.array([t.get_window_extent(renderer=renderer).height
                           for t in temp_labels])
    for t in temp_labels:
        t.remove()
    label_height_px = np.median(heights_px) * spacing_factor

    # Convert pixel height to data-space Δy
    (_, y0_px) = axis.transData.transform((0, 0))
    (_, y1_px) = axis.transData.transform((0, 1))
    px_per_data = abs(y1_px - y0_px)
    delta_y_data = label_height_px / px_per_data

    # Start slightly below the top of the Lamé curve
    start_y = radius_y - delta_y_data
    y_targets = start_y - np.arange(num_labels) * delta_y_data
    y_targets = np.clip(y_targets, 0, radius_y)

    # Map target y-values to nearest sampled Lamé points
    theta_values = np.zeros_like(y_targets)
    for i, y_t in enumerate(y_targets):
        idx = np.argmin(np.abs(y_dense - y_t))
        theta_values[i] = theta_dense[idx]

    if side == "L":
        theta_values = np.pi - theta_values
    return theta_values


def lame_coords(
            angles: np.ndarray,
            radius_x: float,
            radius_y: float,
            exponent_p: float,
            side: str
        ) -> np.ndarray:
    """
    Compute mirrored Lamé (superellipse) coordinates for both sides
    of the curve using the stable first-quadrant parameterization.

    Args:
        angles (np.ndarray): Array of Lamé angles (radians).
        radius_x (float): Semi-axis length in x-direction.
        radius_y (float): Semi-axis length in y-direction.
        exponent_p (float): Shape exponent p (2=circle, >2=squared shape).
        side (str): "R" for right or "L" for left mirrored half.

    Returns:
        np.ndarray: Array of (x, y) coordinates on the Lamé curve.

    Raises:
        ValueError: If parameters are invalid or side is not 'L' or 'R'.
    """
    # #########################################################
    # Validate inputs to ensure numeric stability and side flag.
    if radius_x <= 0 or radius_y <= 0 or exponent_p <= 0:
        msg = "radius_x, radius_y, and exponent_p must be positive."
        raise ValueError(msg)
    if side not in {"L", "R"}:
        raise ValueError("side must be either 'L' or 'R'.")
    # #########################################################
    # Map generic angle set to first quadrant, then mirror if needed.
    if side == "R":
        x_coords, y_coords = lame_param_quadrant(
            np.pi / 2 - (np.pi / 2 - angles % (np.pi / 2)),
            radius_x,
            radius_y,
            exponent_p,
        )
    else:
        x_coords, y_coords = lame_param_quadrant(
            np.pi / 2 - (np.pi / 2 - (np.pi - angles) % (np.pi / 2)),
            radius_x,
            radius_y,
            exponent_p,
        )
        x_coords = -x_coords
    # #########################################################
    # Stack coordinate pairs for return.
    return np.column_stack((x_coords, y_coords))


##############################################################################################
# The actual plot
def plot_volcano(
            deg_df: pd.DataFrame,
            group: str | list[str] | None = None,
            genes: str | list[str] | None = None,
            top_n: int | None = 10,
            sort_by: str = "scores",
            balanced: bool = True,
            spacing_factor: float = 1.5,
            label_offset_px: float = 20.0,
            text_shift_px: float = 10.0,
            font_kwargs: dict[str, Any] | None = None,
            line_kwargs: dict[str, Any] | None = None,
            bbox_margin_factor: float = .1,
            base_figsize: tuple[float, float] = (7.0, 5.0),
            ncols: int | None = None,
            scatter_alpha: float = 0.6,
            title_prefix: str = "Volcano Plot",
            title: str | None = None,
            show: bool = True,
            axis: plt.Axes | np.ndarray | None = None,
            save_path: str | None = None,
    	    lfc_thresh: float = 1.0,
    	    pval_thresh: float = 0.001,
    	    max_label_len: int = 15,
            ) -> None | np.ndarray[plt.Axes]:
    """
    Generate an enhanced volcano plot of differential gene expression (DEG) results.

    This function plots DEG data using Lamé (superellipse) curves to position
    labels evenly and minimize overlap, adds reference lines for |log2FC| and
    -log10(padj) thresholds, and automatically clips long gene names with an
    ellipsis to maintain readability and balanced visual structure.

    Note:
        - The possible `sort_by` options include:
          {"scores", "logfoldchanges", "pvals", "pvals_adj",
          "pct_nz_group", "pct_nz_reference", "abs_scores",
          "abs_logfoldchanges"}.
        - You cannot use `top_n` and `genes` simultaneously.
          Specify only one selection method per plot.
        - Gene names longer than `max_label_len` are truncated
          and end with a '|' character.

    Args:
        deg_df (pd.DataFrame): Differential expression results table.
        group (str | list[str] | None, optional): Group(s) to plot.
            Defaults to None.
        genes (str | list[str] | None, optional): Specific genes to annotate.
            Defaults to None.
        top_n (int | None, optional): Number of top genes to label.
            Defaults to 10.
        sort_by (str, optional): Column to sort genes by. Defaults to "scores".
        balanced (bool, optional): Balance top_n genes across both tails.
            Defaults to True.
        spacing_factor (float, optional): Vertical spacing multiplier.
            Defaults to 1.5.
        label_offset_px (float, optional): Horizontal label offset in pixels.
            Defaults to 20.0.
        text_shift_px (float, optional): Tangential text shift in pixels.
            Defaults to 10.0.
        font_kwargs (dict[str, Any] | None, optional): Font styling for labels.
            Defaults to None.
        line_kwargs (dict[str, Any] | None, optional): Arrow line properties.
            Defaults to None.
        bbox_margin_factor (float, optional): Plot margin scaling factor.
            Defaults to 0.1.
        base_figsize (tuple[float, float], optional): Base figure size.
            Defaults to (7.0, 5.0).
        ncols (int | None, optional): Number of subplot columns. Defaults to None.
        scatter_alpha (float, optional): Scatter transparency. Defaults to 0.6.
        title_prefix (str, optional): Prefix for plot title.
            Defaults to "Volcano Plot".
        title (str | None, optional): Custom title. Defaults to None.
        show (bool, optional): Whether to display the figure. Defaults to True.
        axis (matplotlib.pyplot.Axes | np.ndarray | None, optional):
            Axis or collection of axes on which to draw the plot.

                - If a single `matplotlib.pyplot.Axes` is provided, all
                  content is plotted onto that axis.
                - If an `np.ndarray` of Axes objects is provided, each element can be used
                  for multi-panel layouts. The array can be arranged with any number of
                  rows and columns, and should be flattened using `.ravel()` before being
                  passed to the function.
                - If `None`, a new figure and axis are automatically created.

            Defaults to None.
        save_path (str | None, optional): If provided, saves the figure to this
            path. Defaults to None.
        lfc_thresh (float, optional): Absolute log2FC threshold line.
            Defaults to 1.0.
        pval_thresh (float, optional): Adjusted p-value threshold. Defaults to 0.05.
        max_label_len (int, optional): Maximum gene name length before clipping.
            Defaults to 15.

    Returns:
         None | np.ndarray[matplotlib.pyplot.Axes]:
            Displays or returns matplotlib Axes array.
    """
    # #########################################################
    # IO checks
    _validate_deg_input(deg_df)
    groups = _resolve_groups(deg_df, group)
    # #########################################################
    # Set default font and arrow properties.
    font_kwargs = font_kwargs or {}
    font_kwargs.setdefault("fontsize", 8)
    font_kwargs.setdefault("fontweight", "bold")

    line_kwargs = line_kwargs or {}
    line_kwargs.setdefault("lw", 0.6)
    line_kwargs.setdefault("line_color", "black")
    # #########################################################
    # Prepare figure layout and axes.
    num_plots = len(groups)
    ncols, nrows, figsize = get_rows_cols_figsize(
        n_categories=num_plots, ncols=ncols, base_figsize=base_figsize)
    if axis is not None:
        axes = np.atleast_1d(axis).ravel()
        fig = axes[0].get_figure()
        if len(axes) < num_plots:
            raise ValueError("Number of provided axes < number of groups.")
    else:
        ncols, nrows, figsize = get_rows_cols_figsize(
            n_categories=num_plots, ncols=ncols, base_figsize=base_figsize
        )
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True
        )
        axes = np.ravel(axes)
    axes = np.ravel(axes)
    # #########################################################
    # Iterate over all groups and create plots.
    for ax_i, grp in enumerate(groups):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]

        grp_df_full = deg_df[deg_df["group"] == grp].copy()
        if grp_df_full.empty:
            continue
        ref = grp_df_full["reference"].iloc[0]
        # #########################################################
        # Subset gene DataFrame as per user input.
        if genes is not None:
            grp_df = _subset_by_genes(grp_df_full, genes)
        elif top_n is not None:
            grp_df = _select_top_genes(grp_df_full, sort_by, top_n, balanced)
        else:
            grp_df = pd.DataFrame(columns=grp_df_full.columns)
        # #########################################################
        # Prepare base volcano scatter plot.
        all_logfc = grp_df_full["logfoldchanges"].to_numpy()
        all_neglogp = -np.log10(grp_df_full["pvals_adj"].clip(lower=1e-300)).to_numpy()
        colors = np.where(all_logfc > 0, "red", "blue")
        ax.scatter(all_logfc, all_neglogp, c=colors, s=8, alpha=scatter_alpha)
        ax.axvline(0, color="grey", lw=0.8)
        ax.axvline(lfc_thresh, color="red", ls="--", lw=1.0)
        ax.axvline(-lfc_thresh, color="blue", ls="--", lw=1.0)
        ax.axhline(-np.log10(pval_thresh), color="black", ls="--", lw=0.8)
        ax.set_xlabel("log2FC")
        ax.set_ylabel("-log10(padj)")
        ax.set_title(title if title else f"{title_prefix}: {grp} vs {ref}")
        if grp_df.empty:
            continue

        logfc = grp_df["logfoldchanges"].to_numpy()
        neglogp = -np.log10(grp_df["pvals_adj"].clip(lower=1e-300)).to_numpy()
        names = [n if len(n) <= max_label_len else f"{n[:max_label_len-1]}|" for n in grp_df["names"]]
        top_idx = np.arange(len(grp_df))

        radius_x = np.abs(all_logfc).max() * 1.05
        radius_y = np.abs(all_neglogp).max() * 1.05
        exponent_p = 4.0
        renderer = fig.canvas.get_renderer()
        # #########################################################
        # Split genes by sign and compute Lamé curve anchor points.
        left_genes = [(i, names[i]) for i in top_idx if logfc[i] < 0]
        right_genes = [(i, names[i]) for i in top_idx if logfc[i] >= 0]

        angles_left = stack_angles_lame_even_y(
            left_genes, radius_x, radius_y, exponent_p, "L",
            ax, renderer, font_kwargs, spacing_factor)
        angles_right = stack_angles_lame_even_y(
            right_genes, radius_x, radius_y, exponent_p, "R",
            ax, renderer, font_kwargs, spacing_factor)

        label_coords_left = lame_coords(angles_left, radius_x, radius_y, exponent_p, "L")
        label_coords_right = lame_coords(angles_right, radius_x, radius_y, exponent_p, "R")
        # #########################################################
        # Define nested label placement helper.
        def place_labels(genes, coords, side, offset_px, text_shift_px):
            bboxes = []
            for (idx, name), (x_anchor, y_anchor) in zip(genes, coords):
                gx, gy = logfc[idx], neglogp[idx]
                base_disp = ax.transData.transform((x_anchor, y_anchor))
                new_disp = (
                    base_disp + np.array([-offset_px, text_shift_px])
                    if side == "L"
                    else base_disp + np.array([offset_px, text_shift_px]))
                lx, ly = ax.transData.inverted().transform(new_disp)
                ann = ax.annotate(
                    text=name,
                    xy=(gx, gy),
                    xytext=(lx, ly),
                    **font_kwargs,
                    ha="right" if side == "L" else "left",
                    va="bottom",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=line_kwargs["line_color"],
                        **{k: v for k, v in line_kwargs.items() if k != "line_color"}))
                bboxes.append(ann.get_window_extent(renderer=renderer))
            return bboxes
        # #########################################################
        # Match gene positions to nearest label anchors.
        def optimal_match(data_coords, label_coords):
            cost = cdist(data_coords, label_coords)
            rows, cols = linear_sum_assignment(cost)
            return rows, cols

        data_coords_left = np.column_stack(
            [[logfc[i] for i, _ in left_genes], [neglogp[i] for i, _ in left_genes]])
        data_coords_right = np.column_stack(
            [[logfc[i] for i, _ in right_genes], [neglogp[i] for i, _ in right_genes]])

        if len(left_genes) and len(angles_left):
            rows, cols = optimal_match(data_coords_left, label_coords_left)
            left_genes = [left_genes[i] for i in rows]
            label_coords_left = label_coords_left[cols]
        if len(right_genes) and len(angles_right):
            rows, cols = optimal_match(data_coords_right, label_coords_right)
            right_genes = [right_genes[i] for i in rows]
            label_coords_right = label_coords_right[cols]
        # #########################################################
        # Place labels on both sides.
        bboxes = []
        bboxes += place_labels(
            left_genes, label_coords_left, "L", label_offset_px,
            text_shift_px)
        bboxes += place_labels(
            right_genes, label_coords_right, "R", label_offset_px,
            text_shift_px)
        # #########################################################
        # Draw Lamé reference curve for visual alignment.
        theta = np.linspace(0, np.pi / 2, 400)
        x_half, y_half = lame_param_quadrant(theta, radius_x, radius_y, exponent_p)
        # ax.plot(x_half, y_half, "r--", lw=1.0, alpha=0.5)
        # ax.plot(-x_half, y_half, "b--", lw=1.0, alpha=0.5)
        # #########################################################
        # Adjust axis bounds to fit labels.
        if bboxes:
            all_extents = np.array([[b.x0, b.y0, b.x1, b.y1] for b in bboxes])
            (x0_disp, y0_disp, x1_disp, y1_disp) = (
                all_extents[:, 0].min(),
                all_extents[:, 1].min(),
                all_extents[:, 2].max(),
                all_extents[:, 3].max())
            (lx0, ly0) = ax.transData.inverted().transform((x0_disp, y0_disp))
            (lx1, ly1) = ax.transData.inverted().transform((x1_disp, y1_disp))
            sx0, sx1 = all_logfc.min(), all_logfc.max()
            sy0, sy1 = all_neglogp.min(), all_neglogp.max()
            x0, x1 = min(lx0, sx0), max(lx1, sx1)
            y0, y1 = min(ly0, sy0), max(ly1, sy1)
            dx, dy = (
                (x1 - x0) * bbox_margin_factor,
                (y1 - y0) * bbox_margin_factor)
            ax.set_xlim(x0 - dx, x1 + dx)
            ax.set_ylim(y0 - dy, y1 + dy)

    # #########################################################
    # Save if save path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # #########################################################
    # Remove unused axes if fewer groups than subplot slots
    # only if they were not provided
    if axis is None:
        for ax in axes[ax_i + 1:]:
            ax.remove()
    # #########################################################
    # Show or return axes
    if show and axis is None:
    # only show if self-created
        plt.show()
    else:
        return axes
