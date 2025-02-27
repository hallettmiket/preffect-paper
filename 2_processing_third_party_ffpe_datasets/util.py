from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyreadr


@dataclass
class Dataset:
    data_path: Path

    def is_path_exist(self):
        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)

    def get_id(self):
        return self.data_path.stem.split(".")[0]

    def get_id_stem(self):
        return self.data_path.stem.split("_")[0]

    def read_data(self):
        return (
            pyreadr.read_r(self.data_path)[None]
            if self.data_path.suffix in (".rds", ".rda")
            else pd.read_csv(self.data_path, sep="\t")
        )


def make_dataframe(genes):
    return pd.DataFrame(genes, columns=["gene_id"])


def ensemble_id_to_gene_name(ensemble_ids, lookup_table):
    return set(pd.merge(ensemble_ids, lookup_table)["gene_name"].unique())


def create_gene_conversion_table(ensemble_to_refseq_table_path):
    gene_convert = pyreadr.read_r(ensemble_to_refseq_table_path)[None]
    gene_conversion_table = gene_convert[["gene_id", "gene_name"]]

    return gene_conversion_table[
        ~gene_conversion_table["gene_name"].str.startswith("ENSG")
    ]


def find_common_genes(data_mapx):
    gse_12x = data_mapx["GSE120795"].index.values
    gse_14x = set(data_mapx["GSE146889"].GeneName.values)
    gse_16x = data_mapx["GSE167977"].ensembl_gene_id.values
    gse_18x = set(
        data_mapx["GSE181466"]["Unnamed: 0"].str.split("|", expand=True)[0].values
    )
    gse_20x = set(data_mapx["GSE209998"].index.values)
    gse_47x = set(data_mapx["GSE47462"].symbol.values)

    gene_conversion_table = create_gene_conversion_table()

    gse_12_translated = ensemble_id_to_gene_name(
        make_dataframe(gse_12x), gene_conversion_table
    )
    gse_16_translated = ensemble_id_to_gene_name(
        make_dataframe(gse_16x), gene_conversion_table
    )
    # gdc_bc_translated = ensemble_id_to_gene_name(
    #     make_dataframe(gdc_bc.columns.values), gene_conversion_table
    # )

    common_gene_names = (
        gse_12_translated
        & gse_14x
        & gse_16_translated
        & gse_18x
        & gse_20x
        & gse_47x
        # & gdc_bc_translated
    )

    return gene_conversion_table[
        gene_conversion_table["gene_name"].isin(common_gene_names)
    ]
