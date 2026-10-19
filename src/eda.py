# %% [markdown]
# # Heart Disease UCI - Exploratory Data Analysis
# Percent-format script: run with `python -m src.eda`, or open as a notebook
# in VS Code / Jupyter (via jupytext).

# %%
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove when running interactively
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import CATEGORICAL_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, TARGET  # noqa: E402
from src.data_processing import ensure_clean_data  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# %% [markdown]
# ## 1. Overview & missing values
# %%
df = ensure_clean_data()
print(df.shape)
print(df.describe().T.round(2))
missing = df.isna().sum()
print("\nMissing values:\n", missing[missing > 0])

# %% [markdown]
# ## 2. Class balance
# %%
fig, ax = plt.subplots(figsize=(5, 4))
counts = df[TARGET].value_counts().sort_index()
sns.barplot(x=["No disease (0)", "Disease (1)"], y=counts.values, ax=ax)
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, f"{v} ({v / len(df):.1%})", ha="center")
ax.set_title("Class balance")
ax.set_ylabel("Patients")
save(fig, "class_balance.png")

# %% [markdown]
# ## 3. Numeric feature distributions
# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
    sns.histplot(data=df, x=col, hue=TARGET, kde=True, bins=25, ax=ax, element="step")
    ax.set_title(f"Distribution of {col}")
axes.ravel()[-1].axis("off")
fig.suptitle("Numeric features by target", y=1.02, fontsize=14)
fig.tight_layout()
save(fig, "histograms_numeric.png")

# %% [markdown]
# ## 4. Outliers (box plots)
# %%
fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4))
for ax, col in zip(axes, NUMERIC_FEATURES):
    sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
    ax.set_title(col)
fig.tight_layout()
save(fig, "boxplots_numeric.png")

# %% [markdown]
# ## 5. Categorical features vs target
# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
    sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
    ax.set_title(col)
fig.tight_layout()
save(fig, "countplots_categorical.png")

# %% [markdown]
# ## 6. Correlation heatmap
# %%
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
ax.set_title("Correlation heatmap")
save(fig, "correlation_heatmap.png")

target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3)
print("\nCorrelation with target:\n", target_corr)
