from datetime import timedelta
import streamlit as st
import plotly.express as px
import pandas as pd
from bokeh.models.widgets import Div
import copy
import numpy as np

# ---------------------------loading packages---------------------------------------------------
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------loading local files---------------------------------------------

# country_gps_list = pd.read_csv('country_gps.csv')
# country_gps_list.columns = ['country_iso_2', 'latitude', 'longitude', 'country']
# country_gps_list.replace('United States', 'US', inplace=True)
# country_gps_list.country = country_gps_list.country.str.strip()


# ---------------------------reading data----------------------------------------------------
@st.cache(allow_output_mutation=True)
def file_opener(url):
    import pandas as pd
    import requests
    import io
    r = requests.get(url).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8')), skiprows=[1, 1])
    return df


url = "https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_iso3_regions.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&merge-replace02=on&merge-overwrite02=on&tagger-match-all=on&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv"
confirmed = file_opener(url)

url = "https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_iso3_regions.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&merge-replace02=on&merge-overwrite02=on&tagger-match-all=on&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv"
death = file_opener(url)

url = 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_iso3_regions.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bmain%2Bname%2C%23region%2Bsub%2Bcode%2C%23region%2Bsub%2Bname%2C%23region%2Bintermediate%2Bcode%2C%23region%2Bintermediate%2Bname&merge-replace02=on&merge-overwrite02=on&tagger-match-all=on&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv'
recovered = file_opener(url)


# ---------------------------restructure data----------------------------------------------------

@st.cache(allow_output_mutation=True)
def data_cleaning(df, status):
    # retrieve only necessary columns
    df['country'] = np.where(df['Province/State'].notna(), df['Country/Region'] + '-' + df['Province/State'], df['Country/Region'])
    dim_col = ['country','Region Name', 'Lat', 'Long','ISO 3166-1 Alpha 3-Codes']
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
df = pd.concat([confirmed_df, death_df, recovered_df], axis=0)


@st.cache(allow_output_mutation=True)
def feature_engineering(df):
    # pivot datetime
    date_col = df.columns[4:].to_list()
    unwanted_date_col = {'iso_alpha', 'status'}
    date_col = [date for date in date_col if date not in unwanted_date_col]
    dim_col = df.columns[:4].to_list()
    add_list = ['status', 'iso_alpha']
    dim_col = dim_col + add_list
    df = pd.melt(df, id_vars=dim_col, value_vars=date_col, var_name='dt_time', value_name='count')
    df['dt_time'] = pd.to_datetime(df['dt_time'])


    # calculate active cases
    new_dim = list(df.columns)
    new_dim.remove('count')
    new_dim.remove('status')
    new_df = pd.pivot_table(df, index = new_dim, columns = 'status', values= 'count').reset_index()
    new_df['active'] = new_df['confirmed'] -  new_df['recovered']
    new_df = new_df.melt(id_vars=new_dim, value_vars= ['confirmed', 'active', 'recovered', 'death'], value_name='count')


    # calculate the increase rate
    dim = ['country', 'region', 'iso_alpha', 'latitude', 'longitude', 'dt_time', 'status']
    new_df.sort_values(by=dim, inplace=True)
    new_df['diff_change'] = new_df.groupby(['country', 'status'])['count'].apply(pd.Series.pct_change)


    # import population data
    world_pop = pd.read_csv('world_population.csv')
    new_df = pd.merge(new_df, world_pop, how = 'left', left_on = 'iso_alpha', right_on= 'Country Code')

    # import china population and calculate the per million number
    china_pop = pd.read_csv('china_population.csv')
    new_df.country = new_df.country.str.replace('China-', '').str.strip()
    new_df = pd.merge(new_df, china_pop, how= 'left', left_on= 'country', right_on = 'province')
    new_df.population_y = new_df.population_y.fillna(new_df.population_x)
    new_df.drop(columns = ['population_x', 'province'], inplace=True)
    new_df.rename(columns= {'population_y':'population'}, inplace=True)
    new_df['per_mil_count'] = round(new_df['count']/new_df['population'])
    new_df.fillna(0, inplace=True)
    return new_df

df_= feature_engineering(df)



#retrieve the latest date
latest_date = max(df_.dt_time)

if st.button("What's New"):
	st.markdown("2020-5-5: Per million population case updated in the map plot!")

st.title('COVID-19 Visualization')
st.text(f"updated by {latest_date.date()}")


def status_overview(x):
    con1 = x.status == 'confirmed'
    con2 = x.dt_time == latest_date
    total_confirmed = x.loc[con1 & con2]['count'].sum()
    yesterday = latest_date - timedelta(days=1)
    con3 = x.dt_time == yesterday
    total_confirmed_yesterday = x.loc[con1 & con3]['count'].sum()
    new_cases = total_confirmed - total_confirmed_yesterday
    increase_rate = new_cases / total_confirmed
    st.markdown('### Total Confirmed: {}'.format(total_confirmed))
    st.markdown('### Daily Increased New Cases: {}, ⬆️{:.2%}'.format(new_cases, increase_rate))


status_overview(df_)

st.header('Map Plot')
# graphs
# create selector for map
date_selector = st.date_input('Select A Date', max(df_['dt_time']))
status_selector = st.selectbox('Select A Status', list(df_.status.unique()))
per_mil = st.checkbox('Per Million Cases (excluding population<1 million)')

st.success('🔎 Zoom in or drag to move')
@st.cache(persist=True, allow_output_mutation=True)
def gen_map(df):
    # create map
    MBToken = 'pk.eyJ1Ijoic2NvaGVuZGUiLCJhIjoiY2szemMxczZoMXJhajNrcGRsM3cxdGdibiJ9.2oazpPgLvgJGF9EBOYa9Wg'
    px.set_mapbox_access_token(MBToken)
    con1 = df['status'] == status_selector
    con2 = df['dt_time'] == pd.to_datetime(date_selector)
    ax = df.loc[con1 & con2]
    if per_mil:
        dff= ax[ax.population >=1]
        fig1 = px.scatter_mapbox(dff, text='country', opacity=0.6,
                                lat="latitude", lon="longitude", color='status', size="per_mil_count", size_max=50, zoom=0.6,
                                width=1000,
                                height=600, color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                                                'confirmed': '#ADD8E6'}

                                )
        fig1.update_layout(margin=dict(l=0, r=100, t=0), showlegend=False)
        return fig1

    if date_selector <= ax.dt_time.max() and date_selector >= ax.dt_time.min():
        fig2 = px.scatter_mapbox(ax, text='country', opacity=0.6,
                                lat="latitude", lon="longitude", color='status', size="count", size_max=50, zoom=0.6,
                                width=1000,
                                height=600, color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                                                'confirmed': '#ADD8E6'}

                                )
        fig2.update_layout(margin=dict(l=0, r=100), showlegend=False)
        return fig2


st.write(gen_map(df_))

# area plot
st.header('Time Series Visualization')
st.markdown('### Status Area Plot')
country_selector_df = df_.groupby(['country', 'dt_time', 'status'])['count'].agg('sum').reset_index()
top_n = st.number_input('Check top N Highest Confirmed Case Country', value=5, min_value=1, max_value=50)
top_n_country = list(
    country_selector_df.sort_values(by='count', ascending=False).drop_duplicates(subset='country')['country'].head(
        top_n))
country_rank = [i for i in range(1, top_n + 1)]
ziprank = zip(country_rank, top_n_country)
dictcountry = dict(ziprank)
st.markdown(dictcountry)
country_selector = st.multiselect('Select Country', list(df_.country.unique()), ['US', 'United Kingdom'])


# area plot
def area_plot(country_selector, country_selector_df):
    #     def Union(lst1, lst2):
    #         final_list = list(set().union(lst1, lst2))
    #         return final_list
    if country_selector:
        if country_selector is None:
            None
        else:
            con1 = country_selector_df.country.isin(country_selector)
            con2 = country_selector_df.status.isin(['active', 'recovered', 'death'])
            area_df = country_selector_df.loc[con1 & con2]
            fig = px.area(area_df,
                          x="dt_time", y="count", color='status', line_group="country", width=1000,
                          height=600,
                          color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                              'active': '#7B68EE'})
            return fig


st.write(area_plot(country_selector, country_selector_df))

# Daily Change Lineplot
st.markdown('### Daily Increase Trend')
country_selector_single = st.selectbox('Select A Country ', list(df_.country.unique()))

# def change_pct_line(df):
#     con1 = df.country == country_selector_single
#     con2 = df.status.isin(['death', 'active', 'recovered'])
#     dff = df.loc[con1 & con2]
#     dff.sort_values(by = ['country','status','dt_time'], inplace=True)
#     fig = px.line(dff, 'dt_time', 'diff_change', color = 'status',width=1000,
#                           height=600,
#                           color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
#                                               'active': '#7B68EE'})
#     return fig
#
# st.write(change_pct_line(df))

# sidebar
if st.sidebar.checkbox("Show Raw Data"):
    st.header('Raw Data')
    con1 = df_['status'] == status_selector
    con2 = df_['dt_time'] == pd.to_datetime(date_selector)
    data_source = df_.loc[con1 & con2].sort_values(by ='dt_time')
    data_source.dt_time = pd.to_datetime(data_source.dt_time, format='%Y%m%d')
    st.write(data_source)


# layout customization
def _set_block_container_style(
        max_width: int = 1200,
        max_width_100_percent: bool = False,
        padding_top: int = 5,
        padding_right: int = 2,
        padding_left: int = 1,
        padding_bottom: int = 1,
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

st.sidebar.markdown('### Following Plans to Update:')
st.sidebar.warning('🎯  Normalize the time series plot for better apple-to-apple comparison')
st.sidebar.warning('🎯  Add per million population into map and time series plot')
st.sidebar.warning('🎯  Add animation to plot the time series developing trend ')
st.sidebar.warning('🎯  Predict when the COVID will end using time series ML tool')
st.sidebar.warning('🎯  Improve the sidebar function')
st.sidebar.warning(
    '🎯  Using NLP to anaylize the sentiment on COVID-19 -- whether we are gaining more confidence than yesterday in Twitter when tagging the COVID keyword')

st.sidebar.info('Contact: zakkyang@hotmail.com')

# image = Image.open('sample.jpeg')
# st.image(image, caption='Sunrise by the mountains',
#          use_column_width=True)
