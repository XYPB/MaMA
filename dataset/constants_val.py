import os
from pathlib import Path


DATA_BASE_DIR = "<path-to-your-data-folder>"
DATA_BASE_DIR = os.path.expanduser(DATA_BASE_DIR)
MY_API_TOKEN = "<replace-with-your-hf-api-token>"
HF_CKPT_CACHE_DIR = "~/palmer_scratch/hugging-face-cache"
HF_CKPT_CACHE_DIR = os.path.expanduser(HF_CKPT_CACHE_DIR)


# #############################################
# EMBED constants
# #############################################
EMBED_DATA_DIR = DATA_BASE_DIR + "/Embed"
EMBED_DATA_PATH = EMBED_DATA_DIR + "/images"
EMBED_TRAIN_META_CSV = (
    EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_train.csv"
)
EMBED_TEST_META_CSV = (
    EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_test.csv"
)
EMBED_VALID_META_CSV = (
    EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_valid.csv"
)
# Read the full annotation for calcification information
EMBED_ANNO_CSV_REDUCED = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical_reduced.csv"
EMBED_ANNO_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical.csv"
EMBED_LEGENDS_CSV = EMBED_DATA_DIR + "/tables/AWS_Open_Data_Clinical_Legend.csv"
EMBED_INTER_VIEW_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_view.pkl"
EMBED_INTER_SIDE_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_side.pkl"
EMBED_TRAIN_PATH2DENSITY = EMBED_DATA_DIR + "/train_path2density.pickle"
EMBED_VALID_PATH2DENSITY = EMBED_DATA_DIR + "/valid_path2density.pickle"
EMBED_TEST_PATH2DENSITY = EMBED_DATA_DIR + "/test_path2density.pickle"
EMBED_10PCT_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2label.pickle"
EMBED_10PCT_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2density.pickle"


EMBED_IMAGE_TYPE_COL = "FinalImageType"
EMBED_PATH_COL = "anon_dicom_path"
EMBED_PID_COL = "empi_anon"
EMBED_SID_COL = "acc_anon"
EMBED_SIDE_COL = "ImageLateralityFinal"
EMBED_FINDING_SIDE_COL = "side"
EMBED_VIEW_COL = "ViewPosition"
EMBED_DENSITY_COL = "tissueden"
EMBED_BIRADS_COL = "asses"
EMBED_PROCEDURE_COL = "StudyDescription"
EMBED_MASS_SHAPE_COL = "massshape"
EMBED_MASS_DENSITY_COL = "massdens"
EMBED_CALC_FIND_COL = "calcfind"
EMBED_CALC_DIST_COL = "calcdistri"
EMBED_AGE_COL = "age_at_study"
EMBED_ROI_COORD = "ROI_coords"
EMBED_RACE_COL = "RACE_DESC"
EMBED_ETHNIC_COL = "ETHNIC_GROUP_DESC"
EMBED_PATH_TRANS_FUNC = lambda x: x.replace(
    "/mnt/NAS2/mammo/anon_dicom", EMBED_DATA_PATH
)
EMBED_PROCEDURE2REASON_FUNC = lambda x: (
    "screening"
    if "screen" in x.lower()
    else "diagnostic" if "diag" in x.lower() else ""
)
# Normal caption constants
BREAST_BASE_CAPTION = "This is a breast 2D full-field digital mammogram of a patient "
BREAST_SIDE_CAPTION = "on side "  # Make the caption more grammarly correct
BREAST_VIEW_CAPTION = "with view "
BREAST_DENSITY_CAPTION = "with breast tissue density "
BREAST_BIRADS_CAPTION = "with BIRADS score "
# TODO: Add more findings according to the EMBED dataset structure
# Natural Captions
EMBED_NATURE_BASE_CAPTION = (
    "This is a breast 2D full-field digital {{REASON}} mammogram of a patient. "
)
EMBED_NATURE_IMAGE_CAPTION = (
    "This mammogram is for {{SIDE}} breast with {{VIEW}} view. "
)
# Structural Captions
EMBED_PROCEDURE = "Procedure reported: "  # EMBED_PROCEDURE_COL
EMBED_REASON = (
    "Reason for procedure: "  # Screening / Diagnostic, maybe add more details later
)
EMBED_PATIENT = "Patient info: "  # AGE + RACE + ETHNIC
EMBED_IMAGE = "Image info: "  # EMBED_IMAGE_TYPE_COL + EMBED_SIDE_COL + EMBED_VIEW_COL
EMBED_DENSITY = "Breast composition: "  # EMBED_DENSITY_COL + extra description
EMBED_FINDINGS = (
    "Findings: "  # EMBED_MASS info + EMBED_CALC_FIND_COL + extra description
)
EMBED_IMPRESSIONS = "Impressions: "  # EMBED_BIRADS_COL + extra description
EMBED_ASSESSMENT = "Overall Assessment: "  # EMBED_BIRADS_COL number

EMBED_PATIENT_INFO_CAPTION = (
    "This patient is {{RACE}}, {{ETHNIC}}, and {{AGE}} years old. "
)
EMBED_IMAGE_INFO_CAPTION = "This is a {{IMAGE_TYPE}} full-field digital mammogram of the {{SIDE}} breast with {{VIEW}} view. "
EMBED_BREAST_COMPOSITION_CAPTION = "The breast is {{DENSITY}}. "
EMBED_DENSITY_EXTRA_CAPTION = {
    3: "This may lower the sensitivity of mammography. ",
    4: "This may lower the sensitivity of mammography. ",
}
EMBED_FINDS_CAPTION = "The mammogram shows that "
EMBED_MASS_CAPTION = {
    "A": "an additional imaging is recommended. ",
    "N": "no significant masses, calcification, or other abnormalities are present. ",
    "B": "a benign finding is present. ",
    "P": "a probably benign finding is present. ",
    "S": "a suspicious abnormality is present. ",
    "M": "a highly suggestive of malignancy is present, a biopsy is recommended. ",
    "K": "a known biopsy-proven malignant mass is present. ",
}
EMBED_MASS_EXTRA_CAPTION = "The mass is {{SHAPE}} and {{DENSITY}}. "
EMBED_CALC_FINDS_CAPTION = "A {{DISTRI}} {{SHAPE}} calcification is present. "
EMBED_IMPRESSION_CAPTION = "BI-RADS Category {{BIRADS}}: {{BIRADS_DESC}}. "
EMBED_ASSESSMENT_CAPTION = {
    "A": "Additional imaging is recommended. ",
    "N": "Negative. ",
    "B": "Benign. ",
    "P": "Probably benign. ",
    "S": "Suspicious abnormality. ",
    "M": "Highly suggestive of malignancy. ",
    "K": "Known biopsy-proven malignancy. ",
}
EMBED_SIDES_DESC = {
    "L": "left",
    "R": "right",
    "B": "bilateral",
}
EMBED_DENSITY_DESC = {
    1: "almost entirely fat",
    2: "scattered fibroglandular densities",
    3: "heterogeneously dense",
    4: "extremely dense",
    5: "normal male dense",
}
EMBED_LETTER_TO_BIRADS = {
    "A": 0,
    "N": 1,
    "B": 2,
    "P": 3,
    "S": 4,
    "M": 5,
    "K": 6,
}
EMBED_BIRADS_DESC = {
    "A": "additional imaging required",
    "N": "negative",
    "B": "benign finding",
    "P": "probably benign finding",
    "S": "suspicious abnormality",
    "M": "highly suggestive of malignancy",
    "K": "known biopsy-proven malignancy",
}
GET_JPEG_PATH_FUNC = lambda x: x.replace("Embed", "EMBED_1080_JPG").replace(
    ".dcm", "_resized.jpg"
)


# #############################################
# VinDr constants
# #############################################

VINDR_DATA_PATH = DATA_BASE_DIR + "/VinDr/vindr-1.0.0"
VINDR_IMAGE_DIR = VINDR_DATA_PATH + "/images"
VINDR_CSV_DIR = VINDR_DATA_PATH + "/breast-level_annotations.csv"
VINDR_DET_CSV_DIR = VINDR_DATA_PATH + "/finding_annotations.csv"
VINDR_DENSITY_LETTER2DIGIT = {"A": 1, "B": 2, "C": 3, "D": 4}


# #############################################
# RSNA constants
# #############################################
RSNA_MAMMO_DATA_PATH = DATA_BASE_DIR + "/rsna-breast-cancer-detection"
RSNA_MAMMO_JPEG_DIR = RSNA_MAMMO_DATA_PATH + "/RSNA_MAMMO_1080_JPG"
RSNA_MAMMO_TRAIN_CSV = RSNA_MAMMO_DATA_PATH + "/rsna_mammo_train.csv"
RSNA_MAMMO_TEST_CSV = RSNA_MAMMO_DATA_PATH + "/rsna_mammo_test.csv"
RSNA_MAMMO_BALANCE_TEST_CSV = RSNA_MAMMO_DATA_PATH + "/rsna_mammo_balanced_test.csv"
RSNA_MAMMO_CANCER_DESC = {
    0: "Cancer negative: overall healthy or just benign finding",
    1: "Cancer positive: screening image with known biopsy-proven malignancy or suspicious abnormality found",
}
RSNA_MAMMO_BIRADS_DESC = {
    0: ("N or B", "Negative or Benign"),
    1: (
        "A",
        "Additional imaging required with biopsy-proven malignancy or suspicious abnormality found",
    ),
}
