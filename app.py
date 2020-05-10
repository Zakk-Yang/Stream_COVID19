from datetime import timedelta
import streamlit as st
import plotly.express as px
import requests
import io
from bokeh.models.widgets import Div
import pandas as pd
import warnings
import numpy as np
import altair as alt


warnings.filterwarnings('ignore')



# ---------------------------reading data----------------------------------------------------
@st.cache(allow_output_mutation=True)
def load_data(url1, url2, url3):
    r1 = requests.get(url1).content
    confirmed = pd.read_csv(io.StringIO(r1.decode('utf-8')), skiprows=[1, 1])
    r2 = requests.get(url2).content
    death = pd.read_csv(io.StringIO(r2.decode('utf-8')), skiprows=[1, 1])
    r3 = requests.get(url3).content
    recovered = pd.read_csv(io.StringIO(r3.decode('utf-8')), skiprows=[1, 1])

    def data_cleaning(df, status):
        # retrieve only necessary columns
        df['country'] = np.where(df['Province/State'].notna(), df['Country/Region'] + '-' + df['Province/State'],
                                 df['Country/Region'])
        dim_col = ['country', 'Region Name', 'Lat', 'Long', 'ISO 3166-1 Alpha 3-Codes']
        metric_col = list(df.columns)[4:-8]
        df = df[dim_col + metric_col]
        df['status'] = status
        df.dropna(inplace=True)
        df.rename(columns={'Region Name': 'region',
                           'ISO 3166-1 Alpha 3-Codes': 'iso_alpha',
                           'Lat': 'latitude',
                           'Long': 'longitude'}, inplace=True)
        return df

    confirmed_df = data_cleaning(confirmed, 'confirmed')
    death_df = data_cleaning(death, 'death')
    recovered_df = data_cleaning(recovered, 'recovered')
    concat_data = pd.concat([confirmed_df, death_df, recovered_df], axis=0)

    def feature_engineering(df):
        # pivot datetime
        date_col = df.columns[4:].to_list()
        date_col.remove('iso_alpha')
        date_col.remove('status')
        dim_col = df.columns[:4].to_list()
        add_list = ['status', 'iso_alpha']
        dim_col = dim_col + add_list
        df = pd.melt(df, id_vars=dim_col, value_vars=date_col, var_name='date', value_name='count')
        df['date'] = pd.to_datetime(df['date'])

        # calculate active cases
        new_dim = list(df.columns)
        new_dim.remove('count')
        new_dim.remove('status')
        new_df = pd.pivot_table(df, index=new_dim, columns='status', values='count').reset_index()
        new_df['active'] = new_df['confirmed'] - new_df['recovered']
        new_df = new_df.melt(id_vars=new_dim, value_vars=['confirmed', 'active', 'recovered', 'death'],
                             value_name='count')

        # calculate the increase rate and diff
        dim = ['country', 'region', 'iso_alpha', 'latitude', 'longitude', 'date', 'status']
        new_df.sort_values(by=dim, inplace=True)
        new_df['Daily Change%'] = new_df.groupby(['country', 'status'])['count'].apply(pd.Series.pct_change)
        new_df['Daily Case Change'] = new_df.groupby(['country', 'status'])['count'].transform(lambda x: x.diff())

        # import population data
        world_pop = pd.read_csv('world_population.csv')
        new_df = pd.merge(new_df, world_pop, how='left', left_on='iso_alpha', right_on='Country Code')

        # import china population and calculate the per million number
        china_pop = pd.read_csv('china_population.csv')
        new_df.country = new_df.country.str.replace('China-', '').str.strip()
        new_df = pd.merge(new_df, china_pop, how='left', left_on='country', right_on='province')
        new_df.population_y = new_df.population_y.fillna(new_df.population_x)
        new_df.drop(columns=['population_x', 'province'], inplace=True)
        new_df.rename(columns={'population_y': 'population'}, inplace=True)
        new_df['per_mil_count'] = round(new_df['count'] / new_df['population'])
        new_df = new_df.fillna(0).sort_values(by=['country', 'status', 'date'])
        return new_df

    new_df = feature_engineering(concat_data)

    # retrieve the first date when the confirmed case>100
    con1 = new_df['status'] == 'confirmed'
    con2 = new_df['count'] >= 100
    a = new_df.loc[con1 & con2] # slice the group data first
    a = a.groupby('country')['date'].apply(lambda x: x.min()).reset_index() # calculate the min date for each country
    new_df = pd.merge(new_df, a, how='left', on = 'country')
    new_df['Day Since the First 100 Cumulative Confirmed Records'] = new_df['date_x'] - new_df['date_y']
    new_df.rename(columns = {'date_x': 'date'},
                  inplace=True)
    new_df.drop('date_y', axis=1 ,inplace=True)

    return new_df


url1 = "https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_iso3_regions.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&merge-replace02=on&merge-overwrite02=on&tagger-match-all=on&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv"

url2 = "https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_iso3_regions.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&merge-replace02=on&merge-overwrite02=on&tagger-match-all=on&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv"

url3 = 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_iso3_regions.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&merge-replace02=on&merge-overwrite02=on&tagger-match-all=on&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv'

df_ = load_data(url1, url2, url3)

# retrieve the latest date
latest_date = max(df_.date)

width = 400
height = 300
# ---------------------------start page----------------------------------------------------


if st.button("What's New"):
    st.success("2020-5-5: Per million population case updated in the map plot!")
    st.success("2020-5-8: Improved the Area Plot for better country comparison.")
    st.success("2020-5-8: Added a status overview stacked bar.")
    st.success("2020-5-9: Correct the comparison plot and optimize the mobile view.")




st.title('COVID-19 Visualization')
st.text(f"updated by {latest_date.date()}")


def status_overview(x):
    con1 = x.status == 'confirmed'
    con2 = x.date == latest_date
    total_confirmed = x.loc[con1 & con2]['count'].sum()
    yesterday = latest_date - timedelta(days=1)
    con3 = x.date == yesterday
    total_confirmed_yesterday = x.loc[con1 & con3]['count'].sum()
    new_cases = total_confirmed - total_confirmed_yesterday
    increase_rate = new_cases / total_confirmed
    st.markdown('### Total Confirmed: {}'.format(total_confirmed))
    st.markdown('### Daily Increased New Cases: {}, ⬆️{:.2%}'.format(new_cases, increase_rate))


status_overview(df_)

# stackbar overview
st.markdown('#### The peak was gone, but we need to be aware of a second wave')
@st.cache
def stacked_data(df_):
    dff = df_.groupby(['status', 'date'])['Daily Case Change'].agg('sum').reset_index()
    con1 = dff.status.isin(['death', 'active', 'recovered'])
    con2 = dff.date >= '2020-03-01'
    dff = dff.loc[con1 & con2]
    return dff

dff = stacked_data(df_)



# stackedbar overview (altair)
domain = ['recovered', 'death','active']
range_ = ['#90EE90', '#DC143C','#9932CC']
c = alt.Chart(dff).mark_bar().encode(
    x=alt.X('date',axis=alt.Axis(ticks=False, domain=False)),
    y = alt.Y('Daily Case Change',axis=alt.Axis(ticks=False, domain=False)),
    order=alt.Order('status', sort='ascending'),
    color=alt.Color('status', scale=alt.Scale(domain=domain, range=range_),
                    legend=alt.Legend(title="Status", orient = 'top-left')),
    tooltip = ['status', 'Daily Case Change']).configure_axis(
                grid=False).configure_view(strokeWidth=0).properties(
    title = 'Global Daily Case Increase')
st.altair_chart(c, use_container_width=True)





st.header('Map Plot')
# graphs
# create selector for map
date_selector = st.date_input('Select A Date', max(df_['date']))
st.sidebar.markdown('### Widgets')
status_selector = st.sidebar.selectbox('Select A Status', list(df_.status.unique()))

st.info('1. Select a status on the sidebar')
st.info('2. 🔎 Pinch to zoom in or drag to move; select a status on the sidebar')

# --------------------------- Plot Section ----------------------------------------------------

# @st.cache(persist=True, allow_output_mutation=True)
st.markdown(f'### Current status selected: [{status_selector}]')
per_mil = st.checkbox('Per Million Cases (excluding population<1 million)')

def gen_map(df):
    # create map
    MBToken = 'pk.eyJ1Ijoic2NvaGVuZGUiLCJhIjoiY2szemMxczZoMXJhajNrcGRsM3cxdGdibiJ9.2oazpPgLvgJGF9EBOYa9Wg'
    px.set_mapbox_access_token(MBToken)
    con1 = df['status'] == status_selector
    con2 = df['date'] == pd.to_datetime(date_selector)
    ax = df.loc[con1 & con2]
    if per_mil:
        dff = ax[ax.population >= 1]
        fig1 = px.scatter_mapbox(dff, text='country', opacity=0.6,
                                 lat="latitude", lon="longitude", color='status', size="per_mil_count", size_max=50,
                                 zoom=0, hover_name= 'country',
                                 width=800,
                                 height=600, color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                                                 'confirmed': '#ADD8E6'}
                                 )

        fig1.update_layout(margin=dict(l=0, r=100, t=0), showlegend=False)
        return fig1

    elif date_selector <= ax.date.max() and date_selector >= ax.date.min():
        fig2 = px.scatter_mapbox(ax, text='country', opacity=0.6,
                                 lat="latitude", lon="longitude", color='status', size="count", size_max=50, zoom=0,
                                 width=800,
                                 height=600, color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                                                 'confirmed': '#ADD8E6'}
                                 )

        fig2.update_layout(margin=dict(l=0, r=100), showlegend=False)
        return fig2

    else:
        None

with st.spinner('Loading...'):
    st.write(gen_map(df_))




st.header('Time Series Visualization')
st.markdown('### Country Comparison')
st.markdown('##### ℹ️ Please select indicators and status.' )
top_n = st.number_input('Check top N Highest Confirmed Case Country', value=5, min_value=1, max_value=50)
top_n_country = list(
    df_.sort_values(by='count', ascending=False).drop_duplicates(subset='country')['country'].head(
        top_n))
country_rank = [i for i in range(1, top_n + 1)]
ziprank = zip(country_rank, top_n_country)
dictcountry = dict(ziprank)
st.markdown(dictcountry)
country_selector = st.multiselect('Select countries to comparison', list(df_.country.unique()), ['US', 'United Kingdom'])
kpi_selector = st.selectbox('Select an Indicator', ['Daily Change%', 'Daily Case Change'])
st.info('Select a status on the side bar')

# area plot
def alt_area(df, country_selector, kpi_selector):
    dff = df[['country', 'status', 'Day Since the First 100 Cumulative Confirmed Records', 'Daily Change%', 'Daily Case Change']]
    dff['Daily Change%'] = dff['Daily Change%']*100
    dff = pd.melt(dff, id_vars=['country', 'status', 'Day Since the First 100 Cumulative Confirmed Records'], value_vars=['Daily Change%', 'Daily Case Change'],
            value_name='value', var_name='kpi')
    dff = dff[dff['Day Since the First 100 Cumulative Confirmed Records'] >= timedelta(0)]
    dff['Day Since the First 100 Cumulative Confirmed Records'] = dff['Day Since the First 100 Cumulative Confirmed Records'].dt.days
    dff = dff.round(0)
    if country_selector:
        if country_selector is None:
            None
        else:
            con1 = dff.country.isin(country_selector)
            con2 = dff.status == status_selector
            con3 = dff.kpi == kpi_selector
            area_df = dff.loc[con1 & con2 & con3]
            c = alt.Chart(area_df).mark_area(opacity=0.5).encode(
                                x=alt.X("Day Since the First 100 Cumulative Confirmed Records",
                                        axis=alt.Axis(ticks=False, domain=False)
                                        ),
                                y=alt.Y("value", axis=alt.Axis(labels=True, title= kpi_selector,
                                                               ticks=False, domain=False)),
                color=alt.Color('country',
                                legend=alt.Legend(title="Country", orient='top-left')),
                tooltip =[ 'country', 'kpi','value' ]).configure_axis(
                grid=False).configure_view(strokeWidth=0,strokeOpacity=0.1).properties(
    title = f'{kpi_selector} Timeline Trend'
)

            return c

st.altair_chart(alt_area(df_, country_selector, kpi_selector), use_container_width=True)


# @st.cache
# def get_hbar_data(df_):
#     top_n_country = list(
#         df_.sort_values(by='count', ascending=False).drop_duplicates(subset='country')['country'].head(
#             15))
#     df_['month_date'] = df_['date'].dt.strftime("%y/%m/%d")
#     con1 = df_['country'].isin(top_n_country)
#     con2 = df_['status'] == status_selector
#     con3 = df_['month_date'] >= '20/02/25'
#     dff = df_.loc[con1 & con2 & con3]
#     dff.sort_values(by=['date', 'count'], ascending=True, inplace=True)
#     return dff
#
# dff = get_hbar_data(df_)
#
#
# st.markdown('###  Racing Bar Chart-- View the developing animation! ')
# st.markdown('##### ℹ️ The chart is animated to view the developing pace by different countries. It clearly revealed '
#             "how other countries caught up after Hubei's breakout (province of China)")
# st.info("Please select the 'status' on the sidebar and click the play button below to view the animation!")
# fig = px.bar(dff, x="count", y="country", animation_frame="month_date", animation_group="country", color='region',
#                    hover_name="country", width=480, height=600, orientation='h')
# fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',  showlegend=False)
#
# if status_selector:
#     with st.spinner('Loading...'):
#         st.write(fig)


if st.sidebar.checkbox("Show Raw Data"):
    st.header('Raw Data')
    con1 = df_['status'] == status_selector
    con2 = df_['date'] == pd.to_datetime(date_selector)
    data_source = df_.loc[con1 & con2].sort_values(by='date')
    data_source.date = pd.to_datetime(data_source.date, format='%Y%m%d')
    st.write(data_source)


# layout customization
def _set_block_container_style(
        max_width: int = 1200,
        max_width_100_percent: bool = False,
        padding_top: int = 5,
        padding_right: int = 2,
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
        2,  # padding_right
        0.5,  # padding_left
        0,  # padding_bottom
    )


select_block_container_style()

if st.sidebar.button('LinkedIn'):
    js = "window.open('https://www.linkedin.com/in/zakkyang/')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

st.sidebar.markdown('### Following Plans to Update:')
st.sidebar.warning('🎯  Normalize the time series plot for better apple-to-apple comparison')
st.sidebar.warning('🎯  Add per million population into map and time series plot')
st.sidebar.warning('🎯  Add animation to plot the time series developing trend ')
st.sidebar.warning('🎯  Predict when the COVID will end using time series ML tool')
st.sidebar.warning('🎯  Improve the sidebar function')
st.sidebar.warning(
    '🎯  Using NLP to anaylize the sentiment on COVID-19 -- whether we are gaining more confidence than yesterday in Twitter when tagging the COVID keyword')

st.sidebar.info('Contact: zakkyang@hotmail.com')

