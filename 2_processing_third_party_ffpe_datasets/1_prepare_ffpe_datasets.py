from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from util import Dataset


DCIS_EXPR_DATA_PATH = Path(
    "/path/to/metadata/expression_counts.Jan2023_1_2_and_2_2.rds"
)
DCIS_EXPR_DATA_MASTER_PATH = Path(
    "/path/to/metadata/master.Jan2023.RNA-Only.rda"
)

DCIS_META_PATH = Path("/path/to/metadata/ship1_2_tbl.Jan2023.rds")

GENE_REF = Path("/path/to/db/ensembl_release104/GRCh38/Homo_sapiens.GRCh38.104.gtf")

GENE_REF_COL_NAMES = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]

COLUMNS_TO_KEEP = [
    "protein_coding",
    "lncRNA",
    "unprocessed_pseudogene",
    "processed_pseudogene",
    "transcribed_processed_pseudogene",
    "transcribed_unitary_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "TEC",
    "unitary_pseudogene",
    "polymorphic_pseudogene",
    "snRNA",
    "miRNA",
    "misc_RNA",
    "snoRNA",
    "scaRNA",
    "rRNA_pseudogene",
    "pseudogene",
    "rRNA",
    "IG_V_pseudogene",
    "scRNA",
    "IG_V_gene",
    "IG_C_gene",
    "IG_J_gene",
    "sRNA",
    "ribozyme",
    "translated_processed_pseudogene",
    "vault_RNA",
    "TR_V_gene",
    "TR_C_gene",
    "TR_J_gene",
    "TR_V_pseudogene",
    "TR_D_gene",
    "translated_unprocessed_pseudogene",
    "IG_C_pseudogene",
    "TR_J_pseudogene",
    "IG_J_pseudogene",
    "IG_D_gene",
    "IG_pseudogene",
    "Mt_tRNA",
    "Mt_rRNA",
]

FFPE_DATASET_DIR = Path("/path/to/third_party_ffpe/")
GDC_BC = FFPE_DATASET_DIR / "gdc_download.CMI_BC_Only"
FFPE_FORMATTED_DIR = Path("/path/to/Formatted_Third_Party_FFPE/")
BC_DCIS_PRIVATE_DIR = Path("/path/to/Formatted_Private_DCIS_Datasets")
ENSEMBLE_TO_REFSEQ_TABLE_PATH = (
    "/path/to/metadata/ensemble_to_refseq_gene_name_table.rds"
)

DATASET_PATHS = [
    "GSE120795/GSE120795_total_norms_raw_counts.tsv",
    "GSE146889/GSE146889_GeneCount.tsv",
    "GSE167977/GSE167977_Raw_Counts.txt",
    "GSE181466/GSE181466_rsem_genes_matrix-97.txt",
    "GSE209998/GSE209998_GeneCount.rds",
    "GSE47462/GSE47462_Raw_counts_Refseq_genes.txt",
]


def parse_attributes(attribute_str):
    attributes = {}

    for attribute in attribute_str.split(";"):
        if attribute.strip():
            key, value = attribute.strip().split(" ", 1)
            attributes[key] = value.strip('"')

    return attributes


def parse_all_attributes(gtf_dfx):
    attr_data = []

    for row in gtf_dfx["attribute"]:
        a = parse_attributes(row)
        attr_data.append(a)

    return pd.DataFrame(attr_data)


def process_attr_df(attr_dfx, columns_to_keep: list):
    return (
        attr_dfx[attr_dfx["transcript_biotype"].isin(columns_to_keep)][
            ["gene_id", "gene_name"]
        ]
        .dropna()
        .drop_duplicates()
    )


def filter_columns(columns, condition):
    return list(filter(condition, columns))


def get_gdc_samples(top_dir):
    samples = []

    for s in top_dir.glob("**/*gene_counts.tsv"):
        sample = Dataset(data_path=s)
        sample.is_path_exist()
        samples.append(sample)

    return samples


def process_all_gdc_samples(samples):
    def process_sample(path: Path, idx: str):
        return (
            pd.read_csv(
                path,
                sep="\t",
                skiprows=1,
            )
            .dropna()
            .query("gene_type == 'protein_coding'")[["gene_id", "unstranded"]]
            .assign(gene_id_stem=lambda x: x["gene_id"].str.split(".", expand=True)[0])
            .drop(["gene_id"], axis=1)
            .set_index("gene_id_stem")
            .T.rename(index={"unstranded": idx})
        )

    samples_processed = []

    for s in samples:
        sam = process_sample(path=s.data_path, idx=s.get_id())
        sam.index.name = "sample_id"
        sam.columns.name = "gene_id"
        samples_processed.append(sam)

    all_samples = pd.concat(samples_processed)
    all_samples = all_samples.astype("int32")

    return all_samples


def remove_genes_with_zero_counts(adata):
    return adata[:, adata.X.any(axis=0)]  # Remove genes with all zero counts


def filter_with_common_genes(
    gene_expr_table, common_genes, index_col=None, use_index=False, is_ensemble=True
):
    indx = gene_expr_table[index_col] if use_index else gene_expr_table.index

    if is_ensemble:
        return gene_expr_table[indx.isin(common_genes["gene_id"])]
    else:
        return gene_expr_table[indx.isin(common_genes["gene_name"])]


def convert_to_anndata(
    gene_expression_table,
    use_index=False,
    index_col=None,
    obs_subtype="NaN",
    obs_type="NaN",
):
    def pipe(gene_expr):
        adata = ad.AnnData(
            gene_expr.astype("int32").values,
            dtype=gene_expr.astype("int32").values.dtype,
        )
        adata.obs_names = list(gene_expr.index)
        adata.var_names = list(gene_expr.columns)

        adata.obs["batch"] = 1
        adata.obs["subtype"] = obs_subtype
        adata.obs["type"] = obs_type

        adata.var["gene"] = gene_expr.columns.values

        return adata

    if use_index:
        gene_expression_table = gene_expression_table.set_index(index_col)

    gene_expression_table = gene_expression_table.dropna(axis=1)

    return pipe(gene_expression_table.T)


def sqrt_transform(adata):
    adata.layers["sqrt_transformed"] = np.sqrt(adata.X)
    return adata


def make_correlations(adata):
    sample_sample_dist = np.corrcoef(adata.layers["sqrt_transformed"])
    gene_gene_dist = np.corrcoef(adata.layers["sqrt_transformed"].T)
    return sample_sample_dist.astype("float16"), gene_gene_dist.astype("float16")


def convert_to_binary(arr, threshold):
    arr = arr.copy()
    arr[arr >= threshold] = 1
    arr[arr < threshold] = 0


def top_k_indices_symmetric_matrix(matrix, k):
    # Get the upper triangle of the matrix, including the diagonal
    upper_triangle_indices = np.triu_indices(matrix.shape[0])
    upper_triangle_values = matrix[upper_triangle_indices]

    # Get the indices of top k values from the upper triangle
    top_k_1d_indices = np.argsort(upper_triangle_values)[-k:]

    # Convert the 1D indices to 2D indices
    top_k_2d_indices = (
        upper_triangle_indices[0][top_k_1d_indices],
        upper_triangle_indices[1][top_k_1d_indices],
    )

    return list(zip(top_k_2d_indices[0], top_k_2d_indices[1]))


def create_top_k_adjacency_matrix(distance_matrix, k):
    dist_top_k_indices = top_k_indices_symmetric_matrix(distance_matrix, k)

    top_k_adjacency_matrix = np.zeros(distance_matrix.shape)

    for x, y in dist_top_k_indices:
        top_k_adjacency_matrix[x, y] = distance_matrix[x, y]

    np.fill_diagonal(top_k_adjacency_matrix, 1)

    return csc_matrix(top_k_adjacency_matrix, dtype="float16")


def create_top_k_adjacencies(adata, sample_sample_dist, gene_gene_dist, k=10):
    adata.obsm["sample_sample_cor_top10"] = create_top_k_adjacency_matrix(
        sample_sample_dist, k
    )
    adata.varm["gene_gene_cor_top10"] = create_top_k_adjacency_matrix(gene_gene_dist, k)

    return adata


def create_adjacency_matrix(distance_matrix, threshold):
    convert_to_binary(distance_matrix, threshold)
    np.fill_diagonal(distance_matrix, 1)

    return csc_matrix(distance_matrix, dtype="int8")


def create_adjacencies(adata, sample_sample_dist, gene_gene_dist, threshold=0.8):
    adata.obsm["sample_sample_adj"] = create_adjacency_matrix(
        sample_sample_dist, threshold
    )
    adata.varm["gene_gene_adj"] = create_adjacency_matrix(gene_gene_dist, threshold)

    return adata


def compile_anndata(
    dataset,
    common_genes,
    set_column_as_index=False,
    index_col=None,
    is_ensemble=True,
    obs_subtype="NaN",
    obs_type="NaN",
):
    #dataset = filter_with_common_genes(
    #    dataset, common_genes, index_col, set_column_as_index, is_ensemble
    #)
    adata = convert_to_anndata(
        dataset,
        set_column_as_index,
        index_col,
        obs_subtype=obs_subtype,
        obs_type=obs_type,
    )
    adata = remove_genes_with_zero_counts(adata)

    adata = sqrt_transform(adata)
    sample_sample_dist, gene_gene_dist = make_correlations(adata)
    # adata.obsm["sample_sample_cor"] = sample_sample_dist
    # adata.varm["gene_gene_cor"] = gene_gene_dist
    adata = create_adjacencies(adata, sample_sample_dist, gene_gene_dist)
    # adata = create_top_k_adjacencies(adata, sample_sample_dist, gene_gene_dist)

    return adata


def get_dataset_stats(adata):
    """Get Statistics for the Datasets"""

    data = adata.X

    # Calculate the mean
    mean_value = data.sum() / (adata.n_obs * adata.n_vars)

    # Calculate the squared deviations from the mean
    squared_deviations = (data - mean_value) ** 2

    # Calculate the variance
    variance = squared_deviations.sum() / (adata.n_obs * adata.n_vars)

    # Calculate the standard deviation
    standard_deviation = np.sqrt(variance)


    return {
        "Average Library Size": adata.X.sum() // adata.n_obs,
        "Fraction of Zeros": np.round(
            np.count_nonzero(adata.X == 0) / (adata.n_obs*adata.n_vars) , decimals=3
        ),
        "Average Mean Expression": np.round(
            adata.X.sum() / (adata.n_obs * adata.n_vars), decimals=3
        ),
        "Stdev Ave Mean Expression": np.round(
            standard_deviation, decimals=3
        ),
    }


def prepare_attr_df():
    if Path("attributes.csv").exists():
        return pd.read_csv("attributes.csv")

    only_protein_coding = lambda col: "protein_coding" in col.lower()
    # no_pseudogene = lambda col: "pseudogene" not in col.lower()
    # no_rna = lambda col: "rna" not in col.lower()
    # columns_to_keep_no_pseudogene = filter_columns(COLUMNS_TO_KEEP, no_pseudogene)
    # columns_to_keep_no_rna = filter_columns(COLUMNS_TO_KEEP, no_rna)

    columns_to_keep_only_protein_coding = filter_columns(
        COLUMNS_TO_KEEP, only_protein_coding
    )

    gtf_df = pd.read_csv(GENE_REF, sep="\t", comment="#", names=GENE_REF_COL_NAMES)

    attr_df = parse_all_attributes(gtf_df)
    attr_df_protein_coding = process_attr_df(
        attr_df, columns_to_keep_only_protein_coding
    )

    attr_df_protein_coding.reset_index(inplace=True, drop=True)
    attr_df_protein_coding.to_csv("attributes.csv", index=False)

    return attr_df_protein_coding


def compile_third_party_datasets(data_map, common_genes, max_workers=6):
    cols_gse_14 = ["GeneName"] + [
        c for c in data_map["GSE146889"].columns.values if "tumor" in c and "count" in c
    ]

    gse_18_copy = data_map["GSE181466"].copy()
    gse_18_copy["gene_name"] = data_map["GSE181466"]["Unnamed: 0"].str.split(
        "|", expand=True
    )[0]
    gse_18_copy = gse_18_copy[gse_18_copy.gene_name != "?"]
    gse_18_copy = gse_18_copy.drop(columns=["Unnamed: 0"])
    cols_gse_20 = [c for c in data_map["GSE209998"].columns.values if "TT" in c]
    cols_gse_47_normal = [
        c for c in data_map["GSE47462"].columns.values if "normal" in c or "symbol" in c
    ]
    cols_gse_47_tumor = [
        c for c in data_map["GSE47462"].columns.values if "normal" not in c
    ]
    gse_47_tumor_types = [c.split("_")[-1] for c in cols_gse_47_tumor if c != "symbol"]
    gdc_bc = process_all_gdc_samples(get_gdc_samples(GDC_BC))

    with ProcessPoolExecutor(max_workers) as executor:
        future_gse_12 = executor.submit(
            compile_anndata,
            dataset=data_map["GSE120795"],
            common_genes=common_genes,
            set_column_as_index=False,
            index_col=None,
            is_ensemble=True,
        )

        future_gse_14 = executor.submit(
            compile_anndata,
            dataset=data_map["GSE146889"][cols_gse_14],
            common_genes=common_genes,
            set_column_as_index=True,
            index_col="GeneName",
            is_ensemble=False,
        )

        future_gse_16 = executor.submit(
            compile_anndata,
            dataset=data_map["GSE167977"],
            common_genes=common_genes,
            set_column_as_index=True,
            index_col="ensembl_gene_id",
            is_ensemble=True,
        )

        future_gse_18 = executor.submit(
            compile_anndata,
            dataset=gse_18_copy,
            common_genes=common_genes,
            set_column_as_index=True,
            index_col="gene_name",
            is_ensemble=False,
        )

        future_gse_20 = executor.submit(
            compile_anndata,
            dataset=data_map["GSE209998"][cols_gse_20],
            common_genes=common_genes,
            set_column_as_index=False,
            index_col=None,
            is_ensemble=False,
        )

        future_gse_47_normal = executor.submit(
            compile_anndata,
            dataset=data_map["GSE47462"][cols_gse_47_normal],
            common_genes=common_genes,
            set_column_as_index=True,
            index_col="symbol",
            is_ensemble=False,
        )
        future_gse_47_tumor = executor.submit(
            compile_anndata,
            dataset=data_map["GSE47462"][cols_gse_47_tumor],
            common_genes=common_genes,
            set_column_as_index=True,
            index_col="symbol",
            is_ensemble=False,
            obs_type=gse_47_tumor_types,
        )
        future_gdc_bc_anndata = executor.submit(
            compile_anndata,
            dataset=gdc_bc.T,
            common_genes=common_genes,
            set_column_as_index=False,
            index_col=None,
            is_ensemble=True,
        )

        gse_12 = future_gse_12.result()
        gse_14 = future_gse_14.result()
        gse_16 = future_gse_16.result()
        gse_18 = future_gse_18.result()
        gse_20 = future_gse_20.result()
        gse_47_normal = future_gse_47_normal.result()
        gse_47_tumor = future_gse_47_tumor.result()
        gdc_bc_anndata = future_gdc_bc_anndata.result()

    for d in (
        gse_12,
        gse_14,
        gse_16,
        gse_18,
        gse_20,
        gse_47_normal,
        gse_47_tumor,
        gdc_bc_anndata,
    ):
        print(d)
        print(get_dataset_stats(d), "\n")

    gse_12.write(FFPE_FORMATTED_DIR / "GSE120795_19K.tau_1.h5ad")

    gse_14.var_names_make_unique()
    gse_14.write(FFPE_FORMATTED_DIR / "GSE146889_19K_tumor.tau_1.h5ad")

    gse_16.write(FFPE_FORMATTED_DIR / "GSE167977_19K.tau_1.h5ad")
    gse_18.write(FFPE_FORMATTED_DIR / "GSE181466_19K.tau_1.h5ad")
    gse_20.write(FFPE_FORMATTED_DIR / "GSE209998_19K_tumor.tau_1.h5ad")
    gse_47_normal.write(FFPE_FORMATTED_DIR / "GSE47462_19K_normal.tau_1.h5ad")
    gse_47_tumor.write(FFPE_FORMATTED_DIR / "GSE47462_19K_tumor.tau_2.h5ad")
    gdc_bc_anndata.write(FFPE_FORMATTED_DIR / "gdc_download.CMI_BC_Only_19K.tau_1.h5ad")


def compile_inhouse_datasets(common_genes):
    dcis_master_dataset = Dataset(DCIS_EXPR_DATA_MASTER_PATH)
    dcis_data_master = dcis_master_dataset.read_data()
    qc_ok_obs = {
        idx
        for idx in dcis_data_master[dcis_data_master.blacklist == False]["observation"]
    }

    dcis_dataset = Dataset(DCIS_EXPR_DATA_PATH)
    dcis_data = dcis_dataset.read_data()

    dcis_data_obs_ids = {int(col.split("_")[3]) for col in dcis_data.columns}
    dcis_data_obs_ids_qc_ok = dcis_data_obs_ids.intersection(qc_ok_obs)
    dcis_data_obs_ids_qc_ok = dcis_data_obs_ids.intersection(qc_ok_obs)

    tumor_annot = {
        "primary": "D",
        "normal": "N",
        "stroma": "S",
    }

    cols = [
        c for c in dcis_data.columns if int(c.split("_")[3]) in dcis_data_obs_ids_qc_ok
    ]

    dcis_data_qc_ok = dcis_data.copy()[cols]

    dcis_data_qc_ok_primary = dcis_data_qc_ok.filter(regex=f"_{tumor_annot['primary']}")
    dcis_data_qc_ok_stroma = dcis_data_qc_ok.filter(regex=f"_{tumor_annot['stroma']}")
    dcis_data_qc_ok_normal = dcis_data_qc_ok.filter(regex=f"_{tumor_annot['normal']}")

    dcis_meta_dataset = Dataset(DCIS_META_PATH)
    dcis_meta = dcis_meta_dataset.read_data()

    dcis_meta_subtype_info = dcis_meta[["sample_name", "subtype"]]

    subtypes = []

    for col in dcis_data_qc_ok_primary.columns:
        dcis_meta_subtype_info_slice = dcis_meta_subtype_info[
            dcis_meta_subtype_info["sample_name"] == col
        ]["subtype"].values.tolist()

        if dcis_meta_subtype_info_slice:
            subtypes.append(dcis_meta_subtype_info_slice[0])
        else:
            subtypes.append("NaN")

    with ProcessPoolExecutor(max_workers=6) as executor:
        future_dcis_data_qc_ok_primary = executor.submit(
            compile_anndata,
            dataset=dcis_data_qc_ok_primary,
            common_genes=common_genes,
            set_column_as_index=False,
            index_col=None,
            is_ensemble=True,
            obs_subtype=subtypes,
        )

        future_dcis_data_qc_ok_stroma = executor.submit(
            compile_anndata,
            dataset=dcis_data_qc_ok_stroma,
            common_genes=common_genes,
            set_column_as_index=False,
            index_col=None,
            is_ensemble=True,
        )

        future_dcis_data_qc_ok_normal = executor.submit(
            compile_anndata,
            dataset=dcis_data_qc_ok_normal,
            common_genes=common_genes,
            set_column_as_index=False,
            index_col=None,
            is_ensemble=True,
        )

        dcis_primary = future_dcis_data_qc_ok_primary.result()
        dcis_stroma = future_dcis_data_qc_ok_stroma.result()
        dcis_normal = future_dcis_data_qc_ok_normal.result()

    for d in (dcis_primary, dcis_stroma, dcis_normal):
        print(d)
        print(get_dataset_stats(d), "\n")

    dcis_primary.write(
        BC_DCIS_PRIVATE_DIR / "expression_counts.Jan2023_1_2_and_2_2_19K.tau_1.h5ad"
    )
    dcis_stroma.write(
        BC_DCIS_PRIVATE_DIR / "expression_counts.Jan2023_1_2_and_2_2_19K.tau_2.h5ad"
    )
    dcis_normal.write(
        BC_DCIS_PRIVATE_DIR / "expression_counts.Jan2023_1_2_and_2_2_19K.tau_3.h5ad"
    )


if __name__ == "__main__":
    datasets = [Dataset(FFPE_DATASET_DIR / d) for d in DATASET_PATHS]
    data_map = {d.get_id_stem(): d.read_data() for d in datasets}
    attr_df_protein_coding = prepare_attr_df()

    compile_third_party_datasets(data_map, attr_df_protein_coding)
    #compile_inhouse_datasets(attr_df_protein_coding)
