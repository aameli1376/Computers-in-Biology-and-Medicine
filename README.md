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
