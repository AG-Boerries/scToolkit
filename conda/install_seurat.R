remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE, upgrade="never")
# install.packages("Seurat", Ncpus=7, quiet=True, repos="https://ftp.gwdg.de/pub/misc/cran/")  # this is the 38 mirror to goettingen
# For the R kernel
# BiocManager::install("IRkernel", Ncpus=7, lib=LIB)
# IRkernel::installspec()
IRkernel::installspec(name = 'ir432', displayname = 'R-4.3.2')
# Install latest genesymbol converter
install.packages("HGNChelper", repos = "https://cloud.r-project.org")