import pandas as pd

# input files
pc1_file = "gp_combined_scores/gp_fit_PC1.csv"
pc2_file = "gp_combined_scores/gp_fit_PC2.csv"
pc3_file = "gp_combined_scores/gp_fit_PC3.csv"

# output file
out_file = "Combined_scores_GP.csv"

# load only the GP mean columns
df1 = pd.read_csv(pc1_file)[["MJD", "PC1_gp_mean"]].rename(columns={"PC1_gp_mean": "PC1"})
df2 = pd.read_csv(pc2_file)[["MJD", "PC2_gp_mean"]].rename(columns={"PC2_gp_mean": "PC2"})
df3 = pd.read_csv(pc3_file)[["MJD", "PC3_gp_mean"]].rename(columns={"PC3_gp_mean": "PC3"})

# merge on MJD
df = df1.merge(df2, on="MJD", how="inner").merge(df3, on="MJD", how="inner")

# add dataset column so format matches Combined_scores.csv more closely
df["dataset"] = "GP"

# sort just in case
df = df.sort_values("MJD").reset_index(drop=True)

# save
df.to_csv(out_file, index=False)

print(f"Saved merged GP scores to: {out_file}")
print(df.head())