"""This was used for storing markers, now the automatic csv handling via scToolkit/databases"""

from . import (
    np, plt,
    read_csv, DataFrame,
    get_logger, save_dataframe)

logger = get_logger(name="markers")


def create_the_hg_vcf_genesets(
            path_to_vcf_extracted_categories: str
        ) -> None:
    """This function is totally linked to update_rtracklayer_database.

    update_rtracklayer_database creates the "../databases/Genesets.csv" which
    will be converted to "../databases/gene_to_function.zip"

    This function processes the gene annotations from the provided CSV file and
    generates gene sets grouped by biotype. Pseudogene categories are aggregated
    and filtered to avoid overlap with protein-coding genes. The resulting sets
    are saved as a CSV file for downstream use.

    NOTE:
        - This is tightly coupled with ``update_rtracklayer_database``, which
          creates the input data required by this function.
        - This was actually, now you can use extract_gene_to_biotype() to
          extract the gene biotype.

    Args:
        path_to_vcf_extracted_categories (str): Path to CSV file containing
            extracted gene biotype annotations from VCF-derived data.

    Returns:
        None:
            Writes the curated gene sets to "vcf_genesets.csv" in the current
            directory.

    Raises:
        AssertionError: If any pseudogene overlaps with protein-coding genes
            (should not occur).

    Calls:
        save_dataframe

    Tags:
        annotation, io
    """
    # #################################################
    # Load the data
    ctg = read_csv(path_to_vcf_extracted_categories)
    # convert the row wise to biotype grouped gene lists
    res = ctg.groupby("gene_biotype")["gene_name"].apply(list)
    # #################################################
    # Create a reasonable dict out of the data
    dicts_to_use = {k: v for k, v in res.items()}
    # Collect all the pseudogene categories and remove the protein coding again.
    all_pseudogenes = []

    for k in [k for k in res.keys() if "pseudo" in k.lower()]:
        all_pseudogenes.extend(res[k])
        del dicts_to_use[k]

    all_pseudogenes = np.unique(all_pseudogenes)

    # Check if any pesudo gene is overlapping with the proteincoding ones
    assert len(np.intersect1d(all_pseudogenes, res["protein_coding"])) == 0
    dicts_to_use["pseudogenes_hg"] = all_pseudogenes

    # Remove the protein coding genes from the lncRna
    if len(np.intersect1d(res["lncRNA"], res["protein_coding"])) > 0:
        dicts_to_use["lncRNA"] = np.setxor1d(dicts_to_use["lncRNA"],
                                             np.intersect1d(res["lncRNA"], res["protein_coding"]))
    # Remove the protein coding genes from the miRNA
    if len(np.intersect1d(res["miRNA"], res["protein_coding"])) > 0:
        dicts_to_use["miRNA"] = np.setxor1d(dicts_to_use["miRNA"],
                                            np.intersect1d(res["miRNA"], res["protein_coding"]))
    # #################################################
    # Convert the dict to dataframe and save it as csv
    max_len = max([len(v) for v in dicts_to_use.values()])
    # adjust the value length to convert it to pandas dataframe
    dicts_to_use = {k: list(v) + ([""] * (max_len - len(v))) for k, v in dicts_to_use.items()}
    df = DataFrame(dicts_to_use)
    save_dataframe(df, "vcg_genesets.csv", index=False)


# ###################################################################################################
# Provided Databases
# ## Renaming dict for multi database use
# FIll them if you want to use the functionality.
replacement_dict = {}
replacement_dict_databases = {}
ref_dict_rna = {}
ref_dict_prot = {}

all_celltypes_to_use = []
all_celltypes_to_use.extend(replacement_dict.keys())
for v in replacement_dict.values():
    all_celltypes_to_use.extend(v)

all_celltypes_to_use_databases = ["FC_T_cells", "MAIT"]
all_celltypes_to_use_databases.extend(replacement_dict_databases.keys())
for v in replacement_dict_databases.values():
    all_celltypes_to_use_databases.extend(v)

# ## Create a palette for the cell types and append a dummy for unset
marker_gene_palette = {}
for i, cell_name in zip(np.linspace(0, 1, 1 + len(ref_dict_prot.keys())), list(ref_dict_prot.keys()) + ["unset"]):
    marker_gene_palette[cell_name] = plt.get_cmap('gist_rainbow')(i)
# ###########################################################################################
ref_dict_comb = {}
# ## Create dictionaries for the marker genes and proteins
for k in np.unique([*ref_dict_rna.keys(), *ref_dict_prot.keys()]):
    if k in ref_dict_rna.keys():
        if k in ref_dict_prot.keys():
            ref_dict_comb[k] = list(np.concatenate([ref_dict_rna[k], ref_dict_prot[k]]))
        else:
            ref_dict_comb[k] = ref_dict_rna[k]
    else:
        ref_dict_comb[k] = ref_dict_prot[k]
# ## Create list of all marker genes and proteins
all_markers_rna = []
for marker in ref_dict_rna.values():
    all_markers_rna.extend(marker)

all_markers_prot = []
for marker in ref_dict_prot.values():
    all_markers_prot.extend(marker)

all_markers_comb = all_markers_prot + all_markers_rna
# print(all_markers_comb)

# all_markers_rna_not_to_delete = all_markers_rna.extend(x for x in all_markers_comb if x not in all_markers_rna)
