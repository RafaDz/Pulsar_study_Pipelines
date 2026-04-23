import pandas as pd
import matplotlib.pyplot as plt

# file path
csv_path = "mjd_nudot_err_gp_daily_with_glitches.csv"
csv_path2 = "spin_down_no_F2_and_glitches.csv"
csv_path3 = "spin_down.csv"

# column to plot
nudot_col = "nudot"   # change to "nudot_with_glitches" if needed

# load
df = pd.read_csv(csv_path)
df2 = pd.read_csv(csv_path2)
df3 = pd.read_csv(csv_path3)
# sort by MJD just in case
df = df.sort_values("MJD").reset_index(drop=True)
df2 = df2.sort_values("MJD").reset_index(drop=True)
df3 = df3.sort_values("MJD").reset_index(drop=True)

# plot
plt.figure(figsize=(10, 5))
plt.plot(df["MJD"], df[nudot_col], "-", linewidth=1.0)
plt.plot(df2["MJD"], df2[nudot_col], "-", linewidth=1.0, alpha=0.7)
plt.plot(df3["MJD"], df3[nudot_col], "-", linewidth=1.0, alpha=0.7)
plt.xlabel("MJD")
plt.ylabel(nudot_col)
plt.title(f"{nudot_col} vs MJD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("nudot_comparison.png", format='png', dpi=400)
plt.show()