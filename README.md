# Investigating biomarkers in Parkinson's disease using machine learning

## Overview

This repository contains the implementation of five different approaches for predicting cases and controls in Parkinson's Disease (PD) patients, as described in our presented work.

## Libraries and Dependencies

- Python version: 3.9
- Additional libraries and their versions:

  - NumPy: 1.19.5
  - Pandas: 1.2.1
  - Scikit-learn: 0.24.1
  - Matplotlib: 3.3.4

 ## Setting Up Environment

To run this project, you need to set up your environment with the necessary dependencies. Follow these steps:

`pip install -r requirements.txt`

## Research Objectives

Genome-Wide Association Studies (GWAS) identify genetic variations in individuals affected with diseases such as Parkinson's disease (PD), which are not present in individuals free of this disease. GWAS data can be used to identify genetic variations associated with the disease of interest. However, GWAS datasets are extensive and contain many more Single Nucleotide Polymorphisms (SNPs pronounced “snips”) than individual samples. To address these challenges, we used a recently developed feature selection algorithm (SVFS) and applied it to PD GWAS datasets. We discovered a group of SNPs that are potentially novel PD biomarkers as we found indirect links between them and PD in the literature but have not directly been associated with PD before. These SNPs open new potential lines of investigation. Some of these SNPs are directly related to PD, while others have an indirect relationship.

## Dataset Description

We are using five different datasets obtained from the database of Genotype and Phenotype (dbGaP) [84].

1. **Phs000126 (Familial) dataset:** This dataset combines the results of two major National Institutes of Health (NIH)-funded genetic research projects aimed at discovering new genes that influence the risk of PD. PROGENI (PI: Tatiana Foroud; R01NS037167) and GenePD (PI: Richard Myers; R01NS036711) have been analyzing and recruiting families with two or more PD affected members for over eight years. There are almost 1,000 PD families in the total sample.

2. **Phs000394 (Autopsy)-Confirmed Parkinson DiseaseGWAS Consortium (APDGC):** This consortium was established to perform a genome-wide association research in people with neuropathologically diagnosed PD and healthy controls. The study's hypothesis is that by enrolling only cases and controls with neuropathologically proven illness status, diagnostic misclassification will be reduced and power to identify novel genetic connections will be increased.

3. **Phs000089 (NINDS) repository:** This repository was created in 2001 with the intention of creating standardized, widely applicable diagnostic and other clinical data as well as a collection of DNA and cell line samples to enhance the field of neurological illness gene discovery. All samples, phenotypic information, and genotypic information are accessible. The collection also includes well-described neurologically healthy control subjects. This collection served as the foundation for both the expanded investigation by Simon-Sanchez et al. and the first stage study by Fung et al. The laboratories of Dr. Andrew Singleton of the National Institute on Aging (NIA) and Dr. John Hardy of the NIA produced and submitted the genotyping data (NIH Intramural, funding from NIA and NINDS). NINDS dataset is divided into NINDS1 and NINDS2.

4. **Genome-Wide Association scan phs000048 (Tier 1):** The dbGaP team at NCBI calculated this Genome-Wide Association scan between genotype and PD status. 443 sibling pairs that were at odds for PD served as the samples. Between June 1996 and May 2004, the sibling pairs were drawn from the Mayo Clinic's Rochester, Minnesota, Department of Neurology's clinical practice. Drs. Maraganore and Rocca used three Perlegen DNA chips per person and 85k SNP markers to give genotype data.

## Data Collection

All used data are gathered from the National Center for Biotechnology Information (NCBI) website.

The table below shows a summary of the used datasets. The provided information is extracted from the ped.log of each dataset.

| Dataset ID          | Samples | Missing Phenotype | Cases | Controls | SNPs    |
|---------------------|---------|-------------------|-------|----------|---------|
| phs000394 (Autopsy)| 1001    | 24                | 642   | 335      | 1134514 |
| phs000126 (Familial)| 2082    | 315               | 900   | 867      | 344301  |
| phs000089 (NINDS1)  | 1741    | 0                 | 940   | 801      | 545066  |
| phs000089 (NINDS2)  | 526     | 0                 | 263   | 263      | 241847  |
| phs000048 (Tier1)    | 886     | 0                 | 443   | 443      | 198345  |

## Data Gathering

Before starting to preprocess the datasets, we need to decrypt and convert the datasets into PED format. PED format is a standard format among genomic datasets. After downloading the whole datasets, one would perform the following steps:

### SRA Toolkit Installation

1. **Install SRA toolkit on the server:**
   - Download the installation file from: [SRA Toolkit Installation](https://github.com/ncbi/sra-tools/wiki/02.-Installing-SRA-Toolkit)
   - After downloading the toolkit, extract it with this command:

     ```bash
     tar -vxzf sratoolkit.tar.gz
     ```

   - For convenience (and to show where the binaries are), append the path to the binaries to the PATH environment variable:

     ```bash
     export PATH=$PATH:$PWD/sratoolkit.2.4.0-1.mac64/bin
     ```

   - Verify that the shell will find the binaries:

     ```bash
     which fastq-dump
     ```

     This should produce output similar to:

     ```
     /Users/JoeUser/sratoolkit.2.4.0-1.mac64/bin/fastq-dump
     ```

   - Proceed to configure the toolkit: [Quick Toolkit Configuration](https://github.com/ncbi/sra-tools/wiki/Quick-Toolkit-Configuration)

   - Test that the toolkit is functional:

     ```bash
     fastq-dump --stdout SRR390728 | head -n 8
     ```

     Within a few seconds, the command should produce the exact output as specified.

## Dataset Decryption

We need to decrypt the “matrixfmt” file for our research. To do that, we performed the following steps:

- **Decrypt the file named:** `phg000233.v1.CIDR AutopsyPD.genotype-calls matrixfmt.c1.ARU.tar.ncbi.enc`. As an example, the commands are showing the steps for the Autopsy dataset. This file is encrypted with a `.ngc` key value.

- **Read this guideline before applying the decryption command on the datasets:** [Decryption Guideline](https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?view=toolkit_doc&f=vdb-decrypt)

- **Run this command for decryption:**

```bash
/gpfs/home/aameli/sratoolkit.2.11.0-ubuntu64/bin/vdb-decrypt --ngc /
```

## Dataset Conversion to PED Format with PLINK

Make sure PLINK [106] is installed on the server properly. If it is not installed on one's operating system, download and install it: [PLINK Installation](https://zzz.bwh.harvard.edu/plink/)

- Create a text file named `allfiles.txt` which contains the following:

CIDR AutopsyPD Top sample level.bed
CIDR AutopsyPD Top sample level.bim
CIDR AutopsyPD Top sample level.fam


Note: Sometimes, these three files are zipped. So, extract them before going to the next step. The file names mentioned are from the Autopsy dataset.

- Use PLINK to convert these three files into PED format:

```bash
plink --bfile dbGaP_AutopsyPD_filter --merge-list allFiles.txt --recode --allele1234 --out PD_5 --noweb
```

After running the mentioned command, there will be some outputs. The outputs which we are looking for are:

PD_5.map

PD_5.ped

PD_5.fam

Now, we can run the preprocessing R® script on PED and FAM files to convert them into CSV format.

This section provides instructions on using PLINK to convert dataset files into the PED format, which is essential for further preprocessing. It includes guidance on creating a text file for PLINK, running the conversion command, and identifying the output files required for subsequent processing.


## Approaches

### Approach 0: Baseline Approach

In this baseline approach, we employed the SVFS feature selection algorithm on each dataset separately to extract the most important features. The SVFS feature selection algorithm underwent 50 rounds (5-fold cross-validation for 10 times) for each dataset. The common Single Nucleotide Polymorphisms (SNPs) were obtained for each dataset, and using cross-validation, we assessed the classification performance of a model generated with the common SNPs as features and a Random Forest classifier.

#### Implementation Steps

1. **Feature Selection Algorithm:**
   - Run the feature selection algorithm on each dataset separately.
   - Set the path to each dataset in the code (hardcoded).
   - File: `Approach0/Part_A/Autopsy/SVFs/loop_SVFS.py`
   - Command: `python3 loop_SVFS.py`

2. **Cross Validation on Important Features:**
   - After running the feature selection algorithm on each dataset:
     - File: `Approach0/Part_A/Autopsy/CV/mostCommon.py`
     - Command: `python3 mostCommon.py`
   
3. **Repeat for Other Datasets:**
   - Repeat the same steps for all four datasets.

### Approach 1: Feature Selection from Familial Dataset

In approach 1, we selected the important features from the Familial dataset and performed cross-validation to assess the classification performance of a model generated using each of the other datasets. We ran the SVFS feature selection algorithm on the Familial dataset and extracted the most common SNPs. Our features were the selected most common SNPs from the Familial dataset, obtained the most common SNPs in common with each of the other datasets, and carried out 10-fold cross-validation on each of the other datasets. As with approach 0, we extended the common SNPs with Linkage Disequilibrium (LD) and repeated the same steps in part B.

#### Implementation Steps

1. **Most Common SNPs Selection from Familial Dataset:**
   - Most common SNPs were previously selected from the Familial dataset using the SVFS feature selection.
   - File: `Approach1/Autopsy-FinalAnalyze/MostCommonSNPsFromFamily.py`
   - Command: `python3 MostCommonSNPsFromFamily.py`

2. **Cross Validation on Different Datasets with Familial Features:**
   - After finding the most important features from the Familial dataset:
     - Files: `Approach1/Autopsy-FinalAnalyze/`, `Approach1/NINDS1-FinalAnalyze`, ...
     - Command: `python3 MostCommonSNPsFromFamily.py`
   
3. **For LD Usage:**
   - For using LD, navigate to:
     - File: `Approach1/Autopsy-FinalAnalyze/MostCommonSNPsFromFamilyandLD.py`
     - Command: `python3 MostCommonSNPsFromFamilyandLD.py`

### Approach 2: Intersection of SNPs between Familial and Other Datasets

In approach 2, we first obtained the intersection of SNPs between the Familial dataset and each of the other four datasets. SNPs not in the intersection were removed from the datasets. We then followed the same steps as in approach 1 by extracting the most common SNPs from the condensed version of the Familial dataset and performing cross-validation on the other datasets. As before, for part B, SNPs were extended with Linkage Disequilibrium (LD) as well.

#### Implementation Steps

1. **Run Intersection Code:**
   - Run the intersection code on the Familial dataset and all other 4 datasets to get the common columns between Familial and Autopsy, Familial and NINDS1, Familial and NINDS2, Familial and Tier1.
   - File: `Intersection.py`
   - Command: `python3 Intersection.py`

2. **Feature Selection Algorithm on Filtered Datasets:**
   - After getting the intersection, run the SVFS feature selection algorithm on the filtered datasets separately:
     - File: `Approach2/Part_A/FamilyAndAutopsy/SVFs/loop_SVFS.py`
     - Command: `python3 loop_SVFS.py`

3. **Cross Validation on Important Features for Each Filtered Dataset:**
   - After finding the most important features for each of the 4 filtered datasets, run cross-validation on each of them according to the most important features:
     - File: `Approach2/Part_A/FamilyAndAutopsy/CV/mostCommon.py`
     - Command: `python3 mostCommon.py`
### Approach 3: Merging Datasets for Increased Instances

In approach 3, we increased the number of instances by merging datasets before doing feature selection. Four merged datasets were created: Familial and Autopsy, Familial and NINDS1, Familial and NINDS2, and Familial and Tier1. We got the SNPs in the intersection between the Familial dataset and the other 4 datasets. We ran the SVFS feature selection algorithm on each of the four merged datasets and extracted the most common SNPs. Then, we performed cross-validation to assess the classification performance of a model generated using each of the other datasets.

#### Implementation Steps

1. **Merge Datasets and Add Dataset ID:**
   - Merge Familial datasets with each of the other 4 datasets on common columns.
   - Keep only the common columns between Familial and all other 4 datasets and then merge them together to have more instances.
   - File: `MergeAndID.py`
   - Command: `python3 MergeAndID.py`

2. **Feature Selection Algorithm on Merged Datasets:**
   - Repeat the same steps by running SVFs on the merged version of Familial and Autopsy, Familial and NINDS1, Familial and NINDS2, Familial and Tier1:
     - File: `Approach3/Part_A/FamilyAndAutopsy/SVFs/loop_SVFS.py`
     - Command: `python3 loop_SVFS.py`

3. **Cross Validation on Merged Datasets after Feature Selection:**
   - Run cross-validation on the same merged datasets after finding the most important features from the previous step.
     - File: `Approach3/Part_A/FamilyAndAutopsy/CV/mostCommon.py`
     - Command: `python3 mostCommon.py`

### Approach 4: Balanced Merging of Datasets

In approach 4, the same as approach 3, datasets were merged, but an equal number of instances per dataset were merged. The number of cases and controls taken from each dataset is the same.

#### Implementation Steps

1. **Merge Datasets with Balancing and Add Dataset ID:**
   - Merge Familial datasets with each of the other 4 datasets on common columns, ensuring an equal number of cases and controls.
   - Keep only the common columns between Familial and all other 4 datasets and then merge them together to have more instances, with an equal number of cases and controls.
   - File: `Approach4/MergeAndBalancingAndID.py`
   - Command: `python3 MergeAndBalancingAndID.py`

2. **Feature Selection Algorithm on Balanced Merged Datasets:**
   - Repeat the same steps by running SVFs on the merged version of Familial and Autopsy, Familial and NINDS1, Familial and NINDS2, Familial and Tier1 with the same number of cases and controls (0,1):
     - File: `Approach4/Part_A/FamilyAndAutopsy/SVFs/loop_SVFS.py`
     - Command: `python3 loop_SVFS.py`

3. **Cross Validation on Balanced Merged Datasets after Feature Selection:**
   - Run cross-validation on the same merged datasets after finding the most important features from the previous step.
     - File: `Approach4/Part_A/FamilyAndAutopsy/CV/mostCommon.py`
     - Command: `python3 mostCommon.py`

## Usage Guidelines

- Run the code according to the provided guidelines for each approach and part.
- Optionally, you can choose to run part A (without LD) and part B (with LD) separately based on your preference for using Linkage Disequilibrium (LD) in your analysis.
- In each part, the SNPs in LD with our SNPs are available in a text file in each directory.

## Additional Notes

- Ensure that all required settings specified in the code are properly maintained.
- Download each dataset separately with permission from the NCBI Website.
- Make sure to add the correct path to each file, dataset, and setting in your system before running the code.
- If you encounter any issues or have questions, feel free to reach out for assistance via email at aameli@mun.ca.


**Thank you for using our Parkinson's Disease Prediction project!**
