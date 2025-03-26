import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from .dataset.constants_val import *

df_anno = pd.read_csv("data/tables/EMBED_OpenData_clinical_reduced.csv")
df_anno_patho = pd.read_csv("data/tables/EMBED_OpenData_clinical.csv")
df_meta = pd.read_csv("data/tables/EMBED_OpenData_metadata_reduced.csv")
df_meta = df_meta.drop("Unnamed: 0", axis=1)


# Find inter-view/inter-side images
img_path2same_case = {}
img_path2same_side = {}
same_case_cnt = []
same_side_cnt = []
for i, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
    sid = row[EMBED_SID_COL]
    side = row[EMBED_SIDE_COL]
    cur_path = EMBED_PATH_TRANS_FUNC(row[EMBED_PATH_COL])

    same_study_df = df_meta[df_meta[EMBED_SID_COL] == sid]
    same_study_p = [
        EMBED_PATH_TRANS_FUNC(p) for p in same_study_df[EMBED_PATH_COL].tolist()
    ]
    assert cur_path in same_study_p
    same_study_p.remove(cur_path)
    img_path2same_case[cur_path] = same_study_p
    same_case_cnt.append(len(same_study_p))

    same_side_df = same_study_df[same_study_df[EMBED_SIDE_COL] == side]
    # print(same_side_df[EMBED_VIEW_COL], row[EMBED_VIEW_COL])
    same_side_p = [
        EMBED_PATH_TRANS_FUNC(p) for p in same_side_df[EMBED_PATH_COL].tolist()
    ]
    assert cur_path in same_side_p
    same_side_p.remove(cur_path)
    # print(cur_path, same_side_p)
    img_path2same_side[cur_path] = same_side_p
    same_side_cnt.append(len(same_side_p))

pickle.dump(img_path2same_case, open("data/tables/img_path2inter_side.pkl", "wb"))
pickle.dump(img_path2same_side, open("data/tables/img_path2inter_view.pkl", "wb"))

# Only consider 2D images
df_meta = df_meta[df_meta[EMBED_IMAGE_TYPE_COL] == "2D"]

unique_patients = list(set(df_meta["empi_anon"].tolist()))
unique_exams_cnt_all = Counter(df_meta["acc_anon"].tolist())
patient2exam_all = {
    patient: df_meta[df_meta["empi_anon"] == patient]["acc_anon"].tolist()
    for patient in unique_patients
}
exam2birads_all = {
    exam: df_anno[df_anno["acc_anon"] == exam]["asses"].tolist()
    for exam in unique_exams_cnt_all.keys()
}

exam2patho_all = {
    exam: (
        df_anno_patho[df_anno_patho["acc_anon"] == exam]["path_severity"].tolist(),
        df_anno_patho[df_anno_patho["acc_anon"] == exam]["side"].tolist(),
    )
    for exam in unique_exams_cnt_all.keys()
}
exam2patho_all = {k: [] if np.isnan(v[0][0]) else v for k, v in exam2patho_all.items()}

unique_patients = list(set(df_meta["empi_anon"].tolist()))
unique_exams_cnt = Counter(df_meta["acc_anon"].tolist())
unique_images = list(set(df_meta["anon_dicom_path"].tolist()))
patient2exam = {
    patient: df_meta[df_meta["empi_anon"] == patient]["acc_anon"].tolist()
    for patient in unique_patients
}
patient2img = {
    patient: df_meta[df_meta["empi_anon"] == patient]["anon_dicom_path"].tolist()
    for patient in unique_patients
}
exam2view = {
    exam: df_meta[df_meta["acc_anon"] == exam]["ViewPosition"].tolist()
    for exam in unique_exams_cnt.keys()
}
exam2side = {
    exam: df_meta[df_meta["acc_anon"] == exam]["ImageLateralityFinal"].tolist()
    for exam in unique_exams_cnt.keys()
}
exam2density = {
    exam: df_anno[df_anno["acc_anon"] == exam]["tissueden"].tolist()
    for exam in unique_exams_cnt.keys()
}
exam2birads = {
    exam: df_anno[df_anno["acc_anon"] == exam]["asses"].tolist()
    for exam in unique_exams_cnt.keys()
}

total_rows = len(unique_patients)
first_split = int(total_rows * 0.7)
second_split = first_split + int(total_rows * 0.2)
shuffle_patient = random.sample(unique_patients, len(unique_patients))
patient_train = shuffle_patient[:first_split]
patient_test = shuffle_patient[first_split:second_split]
patient_val = shuffle_patient[second_split:]

df_train = df_meta[df_meta["empi_anon"].isin(patient_train)]
df_test = df_meta[df_meta["empi_anon"].isin(patient_test)]
df_val = df_meta[df_meta["empi_anon"].isin(patient_val)]
print(len(df_train), len(df_test), len(df_val))

df_train.to_csv("data/tables/EMBED_OpenData_metadata_reduced_train.csv", index=False)
df_test.to_csv("data/tables/EMBED_OpenData_metadata_reduced_test.csv", index=False)
df_val.to_csv("data/tables/EMBED_OpenData_metadata_reduced_valid.csv", index=False)

# plot stats
density_values = list(range(4))
density_freq = (
    np.array([27675, 103944, 107166, 14037])
    + np.array([3571, 14988, 15001, 2058])
    + np.array([7387, 31035, 30438, 4179])
)
plt.figure(figsize=(8, 4))
bars = plt.bar(density_values, density_freq, color="lightskyblue")
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 1),
        ha="center",
        va="bottom",
    )
# density == 5 ->  male
plt.xlabel("EMBED Density", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(density_values)
print(density_values, density_freq)
plt.savefig("tmp/density_freq.png", dpi=300)

birads_letter2num = EMBED_LETTER_TO_BIRADS
exam2birads_all = {
    k: [birads_letter2num[vv] for vv in list(set(v)) if vv in birads_letter2num.keys()]
    for k, v in exam2birads_all.items()
}
exam_density_cnt, exam_density_freq = np.unique(
    [len(v) for v in exam2density.values()], return_counts=True
)
overall_birads = []
for v in exam2birads_all.values():
    overall_birads.extend(v)
birads_values, birads_freq = np.unique(overall_birads, return_counts=True)
plt.figure(figsize=(8, 4))
bars = plt.bar(birads_values, birads_freq, color="lightskyblue")
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 1),
        ha="center",
        va="bottom",
    )
plt.xlabel("EMBED BIRADS", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(birads_values)
plt.savefig("tmp/birads_freq.png", dpi=300)
print(birads_values, birads_freq)
