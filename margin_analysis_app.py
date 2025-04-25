import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

class ItemNotFoundError(Exception):
    """Raised when the specified item is not in the DataFrame."""
    pass

def plot_price_cost_margin_figure(
    data: pd.DataFrame,
    item_name: str,
    date_col: str = "Month",
    price_col: str = "avg_unit_price",
    cost_col: str = "avg_unit_cost",
) -> plt.Figure:
    df_item = data.xs(item_name, level="Item", drop_level=False)
    if df_item.empty:
        raise ItemNotFoundError(f"Item '{item_name}' not found")

    x = np.arange(len(df_item))
    width = 0.35

    # colors (edit here)
    bar_color_price, bar_color_cost = "#9FC5E8", "#CACAD3"
    line_color_margin, line_color_trend = "#2ca02c", "#674EA7"

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width/2, df_item[price_col], width, label="Unit Price", color=bar_color_price)
    ax1.bar(x + width/2, df_item[cost_col],  width, label="Unit Cost",  color=bar_color_cost)
    ax1.set_xlabel("Month",               fontweight="bold")
    ax1.set_ylabel("Price/Cost (US$)",    fontweight="bold")
    ax1.set_title(item_name,              fontweight="bold")
    dates = df_item.index.get_level_values(date_col).strftime("%Y-%m")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=45)
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(x, df_item["gross_unit_margin_pct"], marker="o",
             label="Margin %", color=line_color_margin)
    ax2.set_ylabel("Margin %", fontweight="bold")
    ax2.set_ylim(0, None)

    coeffs = np.polyfit(x, df_item["gross_unit_margin_pct"], 1)
    ax2.plot(x, np.poly1d(coeffs)(x), linestyle="--",
             label="Linear Trend", color=line_color_trend)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    return fig


def main():

    # Initialize session state for selected item
    if 'selected_item' not in st.session_state:
        st.session_state.selected_item = None

    st.set_page_config(page_title="Top 100 Margin Analysis", layout="wide")
    st.header("Top 100 Parts: Margin Analysis")

    grouped_items = pd.read_parquet('top_100_items.parquet')

    # only display select columns
    display_df = grouped_items.copy().reset_index(drop=False)
    display_df = display_df[['manufacturer', 'Item', 'level_1_category', 'level_2_category']]

    # remove monthly data from display_df to eliminate all the extra rows
    display_df = display_df.groupby('Item').agg(
        Manufacturer=('manufacturer', 'first'),
        Level_1 = ('level_1_category', 'first'),
        Level_2 = ('level_2_category', 'first'),
    )

    display_df.reset_index(inplace=True)

    # Define desired column order
    display_df = display_df.loc[:, ['Manufacturer', 'Level_1', 'Level_2', 'Item']]

    # Sidebar filters
    st.sidebar.header("Filters")
    mfs = display_df["Manufacturer"].dropna().unique()
    l1s = display_df["Level_1"].dropna().unique()

    mfg = st.sidebar.selectbox("Manufacturer", options=[None, *mfs])
    l1  = st.sidebar.selectbox("Level 1 Category", options=[None, *l1s])

    if l1:
        l2s = display_df.query("Level_1 == @l1")["Level_2"].dropna().unique()
    else:
        l2s = display_df["Level_2"].dropna().unique()

    l2 = st.sidebar.selectbox("Level 2 Category", options=[None, *l2s])

    # Apply filters to DataFrame
    df = display_df
    if mfg: df = df[df["Manufacturer"] == mfg]
    if l1:  df = df[df["Level_1"] == l1]
    if l2:  df = df[df["Level_2"] == l2]

    # Get all available years from the data
    all_months = grouped_items.index.get_level_values('Month')
    available_years = sorted(list(set([month.year for month in all_months])))

    # Add date range filters to sidebar
    with st.sidebar:
        st.subheader("Date Range Filters")

        # TTM option
        use_ttm = st.checkbox("Trailing Twelve Months (TTM)", value=False)

        # Year selection (checkboxes)
        st.write("Select Years:")
        selected_years = []
        for year in available_years:
            if st.checkbox(str(year), value=not use_ttm, disabled=use_ttm):
                selected_years.append(year)

        # If TTM is selected, unselect all years
        if use_ttm:
            selected_years = []

    if use_ttm:
        # Get the latest month in the data
        latest_month = all_months.max()

        # Calculate the month 12 months before the latest month
        ttm_start_month = pd.Period(year=latest_month.year - (1 if latest_month.month != 12 else 0),
                                    month=latest_month.month + (1 if latest_month.month != 12 else -11),
                                    freq='M')

        # Filter data for TTM
        grouped_items = grouped_items.loc[grouped_items.index.get_level_values('Month') >= ttm_start_month]
        date_filter_description = f"Trailing Twelve Months ({ttm_start_month} to {latest_month})"
    elif selected_years:
        # Filter data for selected years
        grouped_items = grouped_items.loc[
            grouped_items.index.get_level_values('Month').map(lambda x: x.year in selected_years)]
        date_filter_description = f"Selected Years: {', '.join(map(str, selected_years))}"
    else:
        # If no years selected and not using TTM, use all data
        date_filter_description = "All Available Data"

    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=True, filterable=True)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_column("Item", headerName="Item (Click to Select)", pinned="left")
    grid_options = gb.build()

    # Display the AgGrid
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=400,
        key='margin_analysis_grid'
    )

    # Check if we have a selection from the grid
    selected_item = None

    selected_rows = grid_response.get('selected_rows', [])
    if len(selected_rows) > 0:
        if isinstance(selected_rows, pd.DataFrame):
            selected_item = selected_rows.iloc[0]['Item']
        else:
            selected_item = selected_rows[0]['Item']
    st.session_state.selected_item = selected_item

    try:
        fig = plot_price_cost_margin_figure(grouped_items, selected_item)
        st.pyplot(fig)
        series = grouped_items.xs(selected_item, level="Item").sort_index()
        st.dataframe(series)
    except ItemNotFoundError as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
