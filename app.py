from data_preprocessing import new_df, latest_date
from datetime import timedelta
import streamlit as st
import plotly.express as px
import pandas as pd
from bokeh.models.widgets import Div
import copy

latest = latest_date

df = copy.deepcopy(new_df)

st.title('COVID-19 Visualization')
st.text(f"updated by {latest.date()}")
def status_overview(dataframe):
    con1 = dataframe.status == 'confirmed'
    con2 = dataframe.dt_time == latest
    total_confirmed = dataframe.loc[con1 & con2]['count'].sum()
    yesterday = latest - timedelta(days=1)
    con3 = dataframe.dt_time == yesterday
    total_confirmed_yesterday = dataframe.loc[con1 & con3]['count'].sum()
    new_cases = total_confirmed - total_confirmed_yesterday
    increase_rate = new_cases/total_confirmed
    st.markdown('### Total Confirmed: {}'.format(total_confirmed))
    st.markdown('### Daily Increased New Cases: {}, ⬆️{:.2%}'.format(new_cases,increase_rate))

status_overview(df)

st.title('Map Plot')

# graphs
# create selector for map
date_selector = st.date_input('Select A Date', max(df['dt_time']))
status_selector = st.selectbox('Select Status', list(df.status.unique()))

# @st.cache
def gen_map(df):
    global date_selector  # for the use in data source table
    global status_selector  # for the use in data source table
    # create map
    MBToken = 'pk.eyJ1Ijoic2NvaGVuZGUiLCJhIjoiY2szemMxczZoMXJhajNrcGRsM3cxdGdibiJ9.2oazpPgLvgJGF9EBOYa9Wg'
    px.set_mapbox_access_token(MBToken)
    df.dropna(inplace=True)
    con1 = df['status'] == status_selector
    con2 = df['dt_time'] == pd.to_datetime(date_selector)
    dff = df.loc[con1 & con2]
    if date_selector <= dff.dt_time.max() and date_selector >= dff.dt_time.min():
        fig = px.scatter_mapbox(dff, text='country', opacity=0.6,
                                lat="latitude", lon="longitude", color='status', size="count", size_max=70, zoom=0,
                                width=1150,
                                height=600, color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                                                'confirmed': '#ADD8E6'}

                                )
        fig.update_layout(margin=dict(l=0, r=100), showlegend=False)
        return fig


st.write(gen_map(df))

# area plot
st.title('Trend Plot')
country_selector_df = df.groupby(['country', 'dt_time', 'status'])['count'].agg('sum').reset_index()
top_n = st.number_input('Select top N Highest Confirmed Case Country', value=5, min_value = 1,  max_value=10)
top_n_country = list(
    country_selector_df.sort_values(by='count', ascending=False).drop_duplicates(subset='country')['country'].head(
        top_n))
country_selector = st.multiselect('Select Country', list(df.country.unique()), top_n_country)

# @st.cache
def area_plot(country_selector_df):
    def Union(lst1, lst2):
        final_list = list(set().union(lst1, lst2))
        return final_list

    country_union_list = Union(top_n_country, country_selector)
    con1 = country_selector_df.country.isin(country_union_list)
    con2 = country_selector_df.status.isin(['active', 'recovered', 'death'])
    area_df = country_selector_df.loc[con1 & con2]
    fig = px.area(area_df,
                  x="dt_time", y="count", color='status', line_group="country", width=1000,
                  height=600,
                  color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                      'active': '#7B68EE'})
    return fig


st.write(area_plot(country_selector_df))

# sidebar
if st.sidebar.checkbox("Show Raw Data"):
    st.header('Raw Data')
    con1 = df['status'] == status_selector
    con2 = df['dt_time'] == pd.to_datetime(date_selector)
    data_source = df.loc[con1 & con2]
    st.write(data_source)


# layout customization
def _set_block_container_style(
        max_width: int = 1200,
        max_width_100_percent: bool = False,
        padding_top: int = 5,
        padding_right: int = 1,
        padding_left: int = 1,
        padding_bottom: int = 0,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def select_block_container_style():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    _set_block_container_style(
        1200,  # max width
        False,
        4,  # padding_top
        1,  # padding_right
        1,  # padding_left
        10,  # padding_bottom
    )


select_block_container_style()

if st.sidebar.button('LinkedIn'):
    js = "window.open('https://www.linkedin.com/in/zakkyang/')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)


st.sidebar.info('Contact: zakkyang@hotmail.com')

# image = Image.open('sample.jpeg')
# st.image(image, caption='Sunrise by the mountains',
#          use_column_width=True)
