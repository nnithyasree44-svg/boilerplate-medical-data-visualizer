import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def draw_cat_plot():
    # 1. Import data
    df = pd.read_csv("medical_examination.csv")

    # 2. Add overweight column
    df["overweight"] = (
        df["weight"] / ((df["height"] / 100) ** 2) > 25
    ).astype(int)

    # 3. Normalize cholesterol and glucose
    df["cholesterol"] = df["cholesterol"].apply(
        lambda x: 0 if x == 1 else 1
    )

    df["gluc"] = df["gluc"].apply(
        lambda x: 0 if x == 1 else 1
    )

    # 5. Create DataFrame for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # 6. Group and reformat data
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7. Create categorical plot
    cat_plot = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    )

    # 8. Get figure
    fig = cat_plot.fig

    # 9. Do not modify
    fig.savefig("catplot.png")

    return fig


def draw_heat_map():
    # 1. Import data
    df = pd.read_csv("medical_examination.csv")

    # Add overweight column
    df["overweight"] = (
        df["weight"] / ((df["height"] / 100) ** 2) > 25
    ).astype(int)

    # Normalize cholesterol and glucose
    df["cholesterol"] = df["cholesterol"].apply(
        lambda x: 0 if x == 1 else 1
    )

    df["gluc"] = df["gluc"].apply(
        lambda x: 0 if x == 1 else 1
    )

    # 11. Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Plot heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax
    )

    # 16. Do not modify
    fig.savefig("heatmap.png")

    return fig