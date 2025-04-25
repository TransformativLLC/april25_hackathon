import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Inventory Parts Cost Change Analysis", layout="wide")

st.title("Inventory Parts: Cost Change Analysis")

# Initialize session state for selected item
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None


# Function to get cost changes
def get_cost_changes(df: pd.DataFrame, lower_threshold: float = 25, upper_threshold: float = 100,
                     first_month=None, last_month=None) -> pd.DataFrame:
    """Get items with significant cost increases between first and last month.

    Args:
        df: Input DataFrame with multi-index (Month, Item) containing cost data
        lower_threshold: Minimum percentage change threshold. Defaults to 25.
        upper_threshold: Maximum percentage change threshold. Defaults to 100.
        first_month: First month to use for comparison. If None, uses earliest month in data.
        last_month: Last month to use for comparison. If None, uses latest month in data.

    Returns:
        DataFrame containing items with cost changes between thresholds, sorted by percentage change.
        Columns include: Beginning Month Cost, Ending Month Cost, Description, and Pct Change.
    """
    # Get first and last months if not provided
    if first_month is None:
        first_month = df.index.get_level_values('Month').min()
    if last_month is None:
        last_month = df.index.get_level_values('Month').max()

    # Get beginning cost from first month and ending cost from last month for each item
    first_costs = df.xs(first_month, level='Month')['Beginning Average Cost']
    last_costs = df.xs(last_month, level='Month')['Ending Average Cost']

    # Calculate percentage change
    begin_cost_col_name = f"Begining Month Cost - {first_month.strftime('%Y-%m')}"
    end_cost_col_name = f"Ending Month Cost - {last_month.strftime('%Y-%m')}"

    cost_changes = pd.DataFrame({
        begin_cost_col_name: first_costs,
        end_cost_col_name: last_costs,
        'Description': df.xs(first_month, level='Month')['Description']
    })

    cost_changes['Pct Change'] = round(
        ((cost_changes[end_cost_col_name] / cost_changes[begin_cost_col_name]) - 1) * 100, 2)

    # replace float('inf') values with 1,000,000 in Pct Change column
    cost_changes['Pct Change'] = cost_changes['Pct Change'].replace([float('inf'), -float('inf')], 1000000)

    # Filter for items where percentage change is between thresholds
    significant_changes = cost_changes[(cost_changes['Pct Change'] <= upper_threshold) &
                                       (cost_changes['Pct Change'] >= lower_threshold)]

    # Sort by percentage change and get top N
    return significant_changes.sort_values('Pct Change', ascending=False)


# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Settings")

    # File uploader
    uploaded_file = st.file_uploader("Upload grouped_items parquet file", type=["parquet"])

    # Threshold inputs
    st.subheader("Cost Change Thresholds")
    lower_threshold = st.number_input("Lower Threshold (%)", min_value=-50.0, max_value=100000.0, value=10.0, step=5.0)
    upper_threshold = st.number_input("Upper Threshold (%)", min_value=-100.0, max_value=1000000.0, value=100.0,
                                      step=5.0)

    # Date range filters will be added after data is loaded

# Main content
if uploaded_file is not None:
    # Load the parquet file
    try:
        grouped_items = pd.read_parquet(uploaded_file)

        # Check if the DataFrame has the expected structure
        if not all(col in grouped_items.columns for col in
                   ['Beginning Average Cost', 'Ending Average Cost', 'Description', 'Subsidiary']):
            st.error(
                "The uploaded file does not contain the required columns: 'Beginning Average Cost', 'Ending Average Cost', 'Description', 'Subsidiary'")
            st.stop()

        # Check if the DataFrame has the expected multi-index
        if not (isinstance(grouped_items.index, pd.MultiIndex) and
                'Month' in grouped_items.index.names and
                'Item' in grouped_items.index.names):
            st.error("The uploaded file does not have the expected multi-index with 'Month' and 'Item' levels.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading the parquet file: {str(e)}")
        st.stop()

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

        # create filter for Subsidiaries
        st.subheader("Subsidiary Filter")
        all_subsidiaries = sorted([s for s in grouped_items['Subsidiary'].unique() if s is not None])
        selected_subsidiaries = [s for s in all_subsidiaries if st.checkbox(s, value=True)]

    # Filter data based on selected date range and subsidiaries
    original_grouped_items = grouped_items.copy()
    grouped_items = grouped_items[grouped_items['Subsidiary'].isin(selected_subsidiaries)]

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

    # Display basic info about the data
    st.subheader("Data Overview")

    # Check if we have data after filtering
    if grouped_items.empty or len(grouped_items.index.get_level_values('Month').unique()) < 2:
        st.error("Not enough data for the selected date range. Please select a different date range.")
        # Restore original data
        grouped_items = original_grouped_items
        st.write("Showing all available data instead:")
        date_filter_description = "All Available Data"

    st.write(f"Number of items: {grouped_items.index.get_level_values('Item').nunique():,}")
    st.write(
        f"Date range: {grouped_items.index.get_level_values('Month').min()} to {grouped_items.index.get_level_values('Month').max()}")
    st.write(f"Date filter: {date_filter_description}")

    # Get cost changes
    cost_changes_df = get_cost_changes(grouped_items, lower_threshold=lower_threshold, upper_threshold=upper_threshold)

    # Display cost changes with AgGrid instead of dataframe
    st.subheader(
        f"{len(cost_changes_df):,} Items with Cost Changes Between {lower_threshold:,}% and {upper_threshold:,}%")

    # Prepare the dataframe for AgGrid - reset index to make Item a column
    display_df = cost_changes_df.copy().reset_index()

    # round costs and pct to 2 decimal places
    display_df.iloc[:, [1, 2, 4]] = display_df.iloc[:, [1, 2, 4]].round(2)

    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(sortable=True, filterable=True)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_column("Item", headerName="Item (Click to Select)", pinned="left")
    grid_options = gb.build()

    # Display the AgGrid
    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=400,
        key='cost_changes_grid'
    )

    # Get unique items from the cost_changes_df for the dropdown
    items = cost_changes_df.index.tolist()
    item_descriptions = {item: cost_changes_df.loc[item, 'Description'] for item in items}

    # Create options with item code and description
    options = [f"{item} - {item_descriptions[item]}" for item in items]
    options = sorted(options)

    # Check if we have a selection from the grid
    selection_triggered = False
    selected_item = None

    selected_rows = grid_response.get('selected_rows', [])
    if len(selected_rows) > 0:
        if isinstance(selected_rows, pd.DataFrame):
            selected_item = selected_rows.iloc[0]['Item']
        else:
            selected_item = selected_rows[0]['Item']
    st.session_state.selected_item = selected_item
    selection_triggered = True

    # Find index of the selected item in options
    default_index = 0
    if st.session_state.selected_item is not None:
        for i, option in enumerate(options):
            if option.startswith(st.session_state.selected_item):
                default_index = i
                break

    # Item selection section
    st.subheader("Item Details")

    # Dropdown for item selection with search
    selected_option = st.selectbox(
        "Select an Item (or click an item in the table above)",
        options=options,
        index=default_index,
        key='item_selector'
    )

    # Extract item code from the selected option or use from session state
    if selected_option:
        # Extract the item code from the selected option
        selected_item = selected_option.split(" - ")[0]
        # Update session state to maintain selection consistency
        if not selection_triggered:
            st.session_state.selected_item = selected_item

    # Proceed with visualization if we have a selected item
    if selected_item:
        # Filter data for the specific item
        item_data = grouped_items.loc[(slice(None), selected_item), :].copy()
        item_data = item_data.reset_index()

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get the first month
        first_month = item_data['Month'].iloc[0]

        # plot beginning and ending costs for first month, then ending for all subsequent months
        ax.plot(str(first_month), item_data.loc[0, 'Beginning Average Cost'], marker='o', label='Beginning Avg Cost',
                color='green')
        ax.plot(str(first_month), item_data.loc[0, 'Ending Average Cost'], marker='o', label='Ending Avg Cost',
                color='blue')

        # Plot ending cost for all months
        ax.plot(item_data['Month'].astype(str), item_data['Ending Average Cost'], marker='o', color='blue')

        ax.set_title(f'Monthly Average Cost for {selected_item}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Cost ($)')
        plt.xticks(rotation=45)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Display the item data
        st.dataframe(item_data[['Month', 'Item', 'Description', 'Beginning Average Cost', 'Ending Average Cost']],
                     height=len(item_data) * 38)
else:
    st.info("Please upload a parquet file with grouped_items data to begin analysis.")

    # Example of what the app does
    st.subheader("How to use this app:")
    st.markdown("""
    1. Upload a parquet file containing grouped items data
    2. Adjust the lower and upper thresholds for cost changes
    3. Filter data by year or use trailing twelve months (TTM)
    4. View the table of items with cost changes within the specified range
    5. Click on an item in the table or select from the dropdown to see its cost trend over time
    """)