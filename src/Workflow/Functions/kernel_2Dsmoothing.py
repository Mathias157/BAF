"""
TITLE

Description

Created on 15.07.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
# %% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import click

# %% ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###


def format_supply_curve(supply_curve: dict):
    """Formats a supply curve for Gaussian smoothing

    Args:
        supply_curve (dict): The supply curve for a commodity and region

    Returns:
        pd.DataFrame: A dataframe that 'do_kernel_smoothing' accepts
    """

    df = pd.DataFrame(supply_curve).T
    df.loc[:, "price"] = df["price"].apply(max)
    df.loc[:, "capacity"] = df["capacity"].apply(max)

    return df


def format_supply_curves_full(supply_curves: dict):
    """Formats all supply curves

    Args:
        supply_curves (dict): Supply curves for all commodities and regions

    Returns:
        pd.DataFrame: A dataframe that 'do_kernel_smoothing' accepts
    """

    commodities = supply_curves.keys()

    # Collect data
    data = []
    for commodity in commodities:
        regions = supply_curves[commodity].keys()

        for region in regions:
            # Get parameters in the index
            temp_df = pd.DataFrame(supply_curves[commodity][region]).T

            # Apply indices
            temp_df["commodity"] = commodity
            temp_df["region"] = region

            # Append
            data.append(temp_df)

    # Get the total dataframe
    df = pd.concat(data)

    # Simply supply curves to max values, as they will be because this assumes using all timeslices
    df.loc[:, ["price"]] = df["price"].apply(max)
    df.loc[:, ["capacity"]] = df["capacity"].apply(max)

    return df


def gaussian_kernel_smooth_2d(
    x1_obs, x2_obs, y_obs, x1_new, x2_new, bandwidth_x, bandwidth_y
):
    """
    2D Gaussian kernel smoothing

    Parameters:
    - x1_obs, x2_obs: 1D arrays of observed x1 and x2 coordinates
    - y_obs: 1D array of observed y values
    - x1_new, x2_new: 1D arrays of new x1 and x2 coordinates where you want predictions
    - bandwidth_x, bandwidth_y: bandwidths for x1 and x2 dimensions

    Returns:
    - y_smooth: 1D array of smoothed y values at (x1_new, x2_new) points
    """
    y_smooth = np.zeros(len(x1_new))

    for i, (x1_target, x2_target) in enumerate(zip(x1_new, x2_new)):
        # Calculate 2D Gaussian weights
        weights = np.exp(
            -0.5
            * (
                ((x1_obs - x1_target) / bandwidth_x) ** 2
                + ((x2_obs - x2_target) / bandwidth_y) ** 2
            )
        )
        weights = weights / np.sum(weights)
        y_smooth[i] = np.sum(weights * y_obs)

    return y_smooth


def do_kernel_smoothing(
    df: pd.DataFrame,
    parameter_x: str,
    parameter_y: str,
    parameter_z: str,
    bandwidth_x: float = 0.1,
    bandwidth_y: float = 0.1,
    plot: bool = False,
    plot_name: str = "output.png",
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, plt.Figure, plt.Axes]
):
    x_obs, x_abs_min, x_abs_max = normalise(getattr(df, parameter_x).astype(float))
    y_obs, y_abs_min, y_abs_max = normalise(getattr(df, parameter_y).astype(float))
    z_obs = getattr(df, parameter_z).astype(float)

    # Create 2D grid for surface plot
    x_smooth = np.linspace(0, 1, 100)
    y_smooth = np.linspace(0, 1, 100)

    # Create meshgrid
    X_grid, Y_grid = np.meshgrid(x_smooth, y_smooth)

    # Flatten for the 2D smoothing function
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    smoothed = gaussian_kernel_smooth_2d(
        x_obs, y_obs, z_obs, X_flat, Y_flat, bandwidth_x, bandwidth_y
    )

    if plot:
        Z_grid = smoothed.reshape(X_grid.shape)  # Reshape back to grid
        # plt.style.use('dark_background')
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor='none')
        ax.scatter(x_obs * x_abs_max + x_abs_min, y_obs * y_abs_max + y_abs_min, z_obs)
        ax.set_xlabel(parameter_x)
        ax.set_ylabel(parameter_y)
        ax.set_zlabel(parameter_z)
        ax.plot_surface(
            X_grid * x_abs_max + x_abs_min,
            Y_grid * y_abs_max + y_abs_min,
            Z_grid,
            alpha=0.7,
            cmap="viridis",
        )
        ax.set_title(
            f"Bandwidth {parameter_x}: {bandwidth_x} Bandwidth {parameter_y}: {bandwidth_y}"
        )

        # if parameter_z == 'price':
        #     ax.set_zlim([0, 500])

        ax.legend((parameter_z, "smoothed"))
        fig.subplots_adjust(hspace=0.7)
        fig.savefig(f"Workflow/MetaResults/{parameter_z}_{plot_name}", transparent=True)
        with open(f"Workflow/MetaResults/{parameter_z}_{plot_name}.pkl", "wb") as f:
            pkl.dump(fig, f)

        return smoothed, X_flat * x_abs_max + x_abs_min, Y_flat * y_abs_max + y_abs_min

    else:
        return smoothed, X_flat * x_abs_max + x_abs_min, Y_flat * y_abs_max + y_abs_min


def normalise(series: pd.Series):
    abs_max = series.max()
    abs_min = series.min()
    series = (series - abs_min) / abs_max

    return series, abs_min, abs_max


### ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###


@click.command()
def main():
    commodity = "HYDROGEN"
    region = "FR"
    output = "capacity"
    parameter_x = "exo_demand"
    parameter_y = "VRE_availability"
    bandwidth_x, bandwidth_y = 5e5, 1e4

    with open("supply_curves.pkl", "rb") as f:
        supply_curves = pkl.load(f)

    df = format_supply_curves_full(supply_curves).query(
        f'commodity == "{commodity}" and region == "{region}"'
    )

    do_kernel_smoothing(df, parameter_x, parameter_y, output, plot=True)


if __name__ == "__main__":
    main()
