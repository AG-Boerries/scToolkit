# If a version with git hash was stored, use that instead
# from . import version
# from .version import __version__
# flake8: skip-file
# std Libraries
# #########################################################
# Set the number of cores here directly
import os

slurm_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
if slurm_cpus is not None:
    num_cores = int(slurm_cpus)
else:
    num_cores = os.cpu_count()  # fallback if not under Slurm

# Keep 5% of cores free to not overload the system
# Setthing a hard max, because more then 50 cores never improved the performance
if num_cores > 10:
    num_cores = min(num_cores - int(num_cores * .05), 50)

os.environ["OMP_NUM_THREADS"] = f'{num_cores}'
import numba as nb
# Set numba threads
nb.set_num_threads(num_cores)
# #########################################################
# Standard Library Imports
import ast
import gzip
import lzma
import pickle
import re
import sys
import warnings
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy as copy
from itertools import chain, combinations, product
from joblib import Parallel, delayed, parallel_config
from json import dump as json_dump
from json import load as json_load
from json import loads as json_loads
from os import path as os_path
from os import remove
from pathlib import Path
from re import search, compile, sub, split  # , findall
from sys import modules, stderr, stdout
from time import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Literal, Callable, NamedTuple, Iterator, Sequence
import inspect
import textwrap
from functools import partial
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*IProgress not found.*")
    from tqdm.auto import tqdm  # or any code that triggers the warning
# #########################################################
# Pandas and NumPy for data manipulation and analysis
import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv, unique as pd_unique
from pandas.errors import PerformanceWarning
from pandas.api.types import is_categorical_dtype
# #########################################################
# Scipy for scientific and technical computing
from scipy.cluster import hierarchy
from scipy.sparse import (
    csc_matrix, csr_matrix, issparse, isspmatrix_csr, spmatrix, isspmatrix_csc,
    SparseEfficiencyWarning, triu, tril)
from scipy.stats import mode as stats_mode
from scipy.stats import gaussian_kde, iqr, normaltest  # , zscore
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, set_link_color_palette
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.utils.sparsefuncs import inplace_column_scale, inplace_row_scale
from sklearn.metrics import jaccard_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_extra.cluster import KMedoids
from KDEpy import FFTKDE
# #########################################################
# Matplotlib for plotting and visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import (
    LinearSegmentedColormap, ListedColormap, Normalize,
    is_color_like, to_rgba, BoundaryNorm, Colormap, TwoSlopeNorm)
from matplotlib.pyplot import cm, colormaps, rc_context
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle as mpl_rectangle
from matplotlib.path import Path as mpl_path
# #########################################################
# Marsilea
from marsilea.plotter import SizedMesh, Labels, Title
from marsilea import WhiteBoard, Heatmap, SizedHeatmap
# #########################################################
# Seaborn for statistical data visualization
from seaborn import (
    clustermap, heatmap, violinplot, stripplot, barplot,
    move_legend, scatterplot, color_palette, axes_style)
# #########################################################
# Legendkit, hopefully it will be constantly updated, otherwise copy the working code here
from legendkit import colorbar, vstack, colorart
from legendkit import legend as lk_legend
# #########################################################
# Color tools
import distinctipy
# #########################################################
# Scanpy and Anndata for single-cell data analysis
import anndata as ad
import scanpy as sc
# #########################################################
# Muon for multi-omics data analysis
# TODO: remove the muon dependancy, they pack so much crap there and introduce too many bugs per release
# from muon import prot as pt
# #################################################################################################
# Handdling for the buggy CUDA error message code
# Function to suppress stderr output

cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
if cuda_devices is not None and cuda_devices.strip():
    print("CUDA devices visible:", cuda_devices)
    try:
        import rapids_singlecell as rsf
    except ImportError:
        print("NO GPU FUNCTIONS AVAILABLE, USE CPU ONLY!")
        rsf = None
    except Exception as e:
        print(f"NO GPU FUNCTIONS AVAILABLE, USE CPU ONLY! The error was {e.__class__.__name__}")
        rsf = None
else:
    print("NO GPUs AVAILABLE, USE CPU ONLY!")
    rsf = None
# #################################################################################################
# Import the functions fron scToolkit
from .common_logger import get_logger, clear_multi_loggers
from .paths import (
    ALL_PATHS, GENESET_FILENAMES, create_code_structure, path_to_repo)

from .sc_config import (
        get_config, get_stats, update_nested_dict,
        update_nested, get_nested_dict_keys_structure,
        update_key_in_config)

# from . import utils

from .sc_replacements import score_genes_efficient

from .utils import (
    get_print_highlighter, df_split_col, replace_special_chars,
    create_acronym, ensure_unique_acronyms, parallel_rank, rankdata_numba,
    replace_np_array_with_list_recursive, clip_dataframe, min_max_scaler,
    inplace_max_scale_csr, set_sparse_subset_to_zero, min_max_scale_axis,
    categorically_group_equal_numba, get_colors_wrapped,
    bilinear_interpolate_numpy, get_fft_grid, save_dataframe,
    determine_clip_threshold, get_categorical_columns,
    validate_groupby_column, _compute_ratio_csr, clean_byte_literal_strings)

from .sc_utils import (
    check_group_overlap,
    flag_gene_family, get_n_unique, convert_to_mem_efficient,
    setup_image_folder, get_save_path,
    # get_highly_variables_sorted,
    get_adata_by_mod, subset_degs,
    get_DEGs_per_group,
    create_group_stats_df, convert_mixed_types,
    get_shape_diff, filter_genes_multicall, filter_cells_multicall,
    get_highly_variable, filter_genes_n_families_regex,
    filter_genes_n_families, get_desired_cpu_gpu_object,
    mark_highly_variable_genes, initialize_cuda,
    get_valide_ref_dicts, get_all_markers_from_ref_dict,
    get_DEG_gene_csvs, save_genesets_to_csv,
    calc_highly_variable_genes_unique_based,
    map_geneset_to_degs, count_genesets_per_group,
    get_msigdb_df, get_adata_sub_keys, replace_small_counts,
    ref_dict_long_value_split, ref_dict_sort_values_hvg_like,
    calc_group_density, get_rows_cols_figsize,
    get_random_generator, get_deg_df,
    calc_DEGs, interpolate_density_to_embedding,
    subset_adata_random, mask_gse_by_significance_threshold,
    deduplicate_ref_dict, get_group_hierarchy,
    _validate_thresholds, get_thresholds, calc_top_n_ratio)


# scToolkit.sc_plots
from .sc_plots import (
    plot_ref_dotplots, create_gene_dendrogram, plot_mediods_heatmap,
    plot_per_group_DEG_dotplot_n_gene_dendrogram,
    continuos_umap_helper, discrete_umap_helper,
    plot_per_group_stacked_violins, plot_ref_stacked_violins,
    plot_per_group_DEG_umaps, plot_hierarchical_heatmap,
    plot_hallmark_group_heatmap, plot_jaccard_heatmap_cluster_comparison,
    plot_embedding_density, get_colormap, plot_umap,
    plot_umap_cat_splitting, calc_dotplot_figsize, plot_spatial)

from .sc_code import cluster_SC_scanpy_like, save_h5ad, load_h5ad

# scToolkit.sc_advanced
__version__ = "0.1.0"
