from datetime import timedelta
import streamlit as st
import plotly.express as px
import requests
import io
import pandas as pd
import warnings
import numpy as np
import altair as alt
warnings.filterwarnings("ignore")
import os
from sqlalchemy import create_engine
import psycopg2 as pg
import pandas.io.sql as psql

postgres_database_password = os.environ['postgres_password']
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
        new_df['active'] = new_df['confirmed'] - new_df['recovered'] - new_df['death']
        # update data error for Spain
        #new_df.at[26774, 'recovered'] = 196958.00
        new_df = new_df.melt(id_vars=new_dim, value_vars=['confirmed', 'active', 'recovered', 'death'],
                             value_name='count')

        # calculate the increase rate and diff
        dim = ['country', 'region', 'iso_alpha', 'latitude', 'longitude', 'date', 'status']
        new_df.sort_values(by=['country', 'status', 'date'], inplace=True)
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


# ---------------------------helper function ----------------------------------------------------

# ---------------------------main ----------------------------------------------------

def main():
    # Render the readme as markdown using st.markdown.
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section:",
        ["Overview", "Country Comparison", "Racing Bar Chart", "Sentiment Analysis"])
    if app_mode == "Overview":
        status_overview(df_)
        gen_stackedbar(df_)
        st.text('')
        st.write(gen_map(df_))
        st.text('')
        st.write(scatter_plot(get_scatter_data(df_)))
    elif app_mode == "Country Comparison":
        st.altair_chart(alt_area(df_), use_container_width=True)
    elif app_mode == "Racing Bar Chart":
        st.write(racing_bar(df_))
    elif app_mode == 'Sentiment Analysis':
        world_sentiment_bar(twitter_db)
        tweet_table(twitter_db)
    st.sidebar.markdown('''[🔗Raw data used in this project](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases/)'''
                                   ,unsafe_allow_html=True)
    st.sidebar.markdown('''[🔗LinkedIn](https://www.linkedin.com/in/zakkyang/)'''
                                   ,unsafe_allow_html=True)
    st.sidebar.info('Contact: zakkyang@hotmail.com')

# ---------------------------contents for all pages----------------------------------------------------
st.title('COVID-19 Visualization')
st.markdown('#### (best for desktop and landscape view in mobile)')
st.text(f"updated by {latest_date.date()}")
st.markdown('### 👈 Please select the section in the sidebar')

# ---------------------------overview page ----------------------------------------------------

def status_overview(x):
    con1 = x.status == 'confirmed'
    con2 = x.date == latest_date
    total_confirmed = x.loc[con1 & con2]['count'].sum()
    yesterday = latest_date - timedelta(days=1)
    con3 = x.date == yesterday
    total_confirmed_yesterday = x.loc[con1 & con3]['count'].sum()
    new_cases = total_confirmed - total_confirmed_yesterday
    increase_rate = new_cases / total_confirmed
    st.write('Total Confirmed: {}'.format(round(total_confirmed)))
    if increase_rate >0:
        st.write('Daily Increased New Cases: {}, ⬆ ️{:.2%}'.format((new_cases), increase_rate))
    else:
        st.write('Daily Increased New Cases: {}, ⬇ ️{:.2%}'.format((new_cases), increase_rate))





# stackedbar overview
def gen_stackedbar(df_):
    st.subheader('Overall Trend')
    # st.write('It is still growing but at a lower pace compared to April. Despite the peak was gone, '
    #         'we still need to be cautious of a second wave.')
    # st.text("")
    # st.text("")

    @st.cache
    def stacked_data(df_):
        dff = df_.groupby(['status', 'date'])['Daily Case Change'].agg('sum').reset_index()
        con1 = dff.status.isin(['death', 'confirmed', 'recovered'])
        con2 = dff.date >= '2020-03-01'
        dff = dff.loc[con1 & con2]
        return dff

    dff= stacked_data(df_)

    # stackedbar overview (altair)
    domain = ['recovered', 'death','confirmed']
    range_ = ['#90EE90', '#DC143C','#9932CC']
    c = alt.Chart(dff).mark_bar().encode(
        x=alt.X('date',axis=alt.Axis(ticks=False, domain=False, grid = False)),
        y = alt.Y('Daily Case Change',axis=alt.Axis(ticks=False, domain=False, grid = False)),
        order=alt.Order('status', sort='ascending'),
        color=alt.Color('status', scale=alt.Scale(domain=domain, range=range_),
                        legend=alt.Legend(title="Status", orient = 'top-left')),
        tooltip = ['date','status', 'Daily Case Change']).properties(
        title = 'Global Daily Case Increase')

    top_n_daily_country = df_.groupby(['country', 'status', 'date'])['Daily Case Change'].agg('sum').\
        reset_index().sort_values(by = 'Daily Case Change', ascending=False).drop_duplicates()
    con1 = top_n_daily_country.status == 'active'
    con2 = top_n_daily_country.date == latest_date
    top_n_daily_country = top_n_daily_country.loc[con1 & con2].head(15)
    c2 = alt.Chart(top_n_daily_country).mark_bar().encode(
        x= alt.X('Daily Case Change',axis=alt.Axis(ticks=False, domain=False, grid = False)),
        y=alt.Y("country", sort = '-x',axis=alt.Axis(ticks=False, domain=False, grid = False))
    ).properties(
        title = 'Top Daily Case Increase by Country')

    text = c2.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='Daily Case Change'
    )

    fig = alt.hconcat(c, (c2 + text)).configure_view(strokeWidth = 0)

    return st.altair_chart(fig, use_container_width=True)



def gen_map(df):
    st.subheader('Map Plot')
    # graphs
    # create selector for map
    global date_selector
    date_selector = st.date_input('Select A Date', max(df_['date']))
    global status_selector
    status_selector = st.selectbox('Select A Status', list(df_.status.unique()))
    # --------------------------- Plot Section ----------------------------------------------------

    # @st.cache(persist=True, allow_output_mutation=True)
    st.markdown(f'#### Current status selected: [{status_selector}]')
    per_mil = st.checkbox('Per Million Cases (excluding population<1 million)')
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
        fig1.update_layout(dragmode=False)
        return fig1

    elif date_selector <= ax.date.max() and date_selector >= ax.date.min():
        fig2 = px.scatter_mapbox(ax, text='country', opacity=0.6,
                                 lat="latitude", lon="longitude", color='status', size="count", size_max=50, zoom=0,
                                 width=800,
                                 height=600, color_discrete_map={'death': '#DC143C', 'recovered': '#90EE90',
                                                                 'confirmed': '#ADD8E6'}
                                 )

        fig2.update_layout(margin=dict(l=0, r=100), showlegend=False)
        fig2.update_layout(dragmode=False)

        return fig2

    else:
        None



@st.cache
def get_scatter_data(df_):
    # prepare the data
    con1 = df_.date >= '2020-03-10'
    con2 = df_.population >=5
    top_n_country = list(
        df_.sort_values(by='count', ascending=False).drop_duplicates(subset='country')['country'].head(
            20))
    con3 = df_.country.isin(top_n_country)
    dff = df_.loc[con1 & con2 & con3]
    dff = pd.pivot_table(dff,index = ['country',
                                      'date','region',
                                      'Day Since the First 100 Cumulative Confirmed Records'],
                         columns = 'status',
                         values = ['count']).reset_index()
    dff = dff.set_index(['country', 'region', 'date', 'Day Since the First 100 Cumulative Confirmed Records'])
    dff.columns = dff.columns.droplevel(0)
    dff = dff.reset_index()
    dff['death_rate'] = dff['death']/dff['confirmed']
    # join per capita
    world_pop = df_[['country', 'population']].drop_duplicates()
    dff = pd.merge(dff, world_pop, on = 'country')
    dff['per_capita_death' ]= dff['death']/dff['population']
    dff['per_capita_confirmed'] = dff['confirmed']/dff['population']
    dff['date'] = dff['date'].dt.strftime('%m/%d')
    dff.dropna(subset = ['per_capita_death'], inplace=True)
    dff.sort_values(by = 'date', inplace=True)
    return dff

def scatter_plot(dff):
    st.subheader('Death Trend')
    st.text('Note: bubble size represents per million population death count')
    fig = px.scatter(dff, x="per_capita_confirmed", y="death_rate",
                     animation_frame="date", size='per_capita_death',
                     animation_group="country", color="region", hover_name="country", size_max=100, text='country',
                     range_x=[0, max(dff['per_capita_confirmed'])], range_y=[0, max(dff['death_rate'])],
                     width=1000,
                     height=600)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_layout(yaxis_title="Death Rate",
                      xaxis_title="Per Million Population Confirmed Case")
    fig.update_layout(dragmode=False)


    return fig
# --------------------------- country comparison page ----------------------------------------------------

def alt_area(df):
    st.markdown('### Country Comparison')
    st.markdown('##### ℹ️ Please select indicators and status.' )
    country_selector = st.multiselect('Select countries to comparison', list(df_.country.unique()), ['US', 'United Kingdom'])
    kpi_selector = st.selectbox('Select an Indicator', ['Daily Change%', 'Daily Case Change'])
    status_selector = st.selectbox('Select A Status', list(df_.status.unique()))
    # area plot
    dff = df[['country', 'status', 'Day Since the First 100 Cumulative Confirmed Records', 'Daily Change%', 'Daily Case Change']]
    dff['Daily Change%'] = dff['Daily Change%']*100
    dff = pd.melt(dff, id_vars=['country', 'status', 'Day Since the First 100 Cumulative Confirmed Records'], value_vars=['Daily Change%', 'Daily Case Change'],
            value_name='value', var_name='kpi')
    dff = dff[dff['Day Since the First 100 Cumulative Confirmed Records'] >= timedelta(0)]
    dff['Day Since the First 100 Cumulative Confirmed Records'] = dff['Day Since the First 100 Cumulative Confirmed Records'].dt.days
    dff = dff.round(0)
    if country_selector is None:
        None
    else:
        con1 = dff.country.isin(country_selector)
        con2 = dff.status == status_selector
        con3 = dff.kpi == kpi_selector
        area_df = dff.loc[con1 & con2 & con3]
        c = alt.Chart(area_df).mark_area(opacity=0.65).encode(
                            x=alt.X("Day Since the First 100 Cumulative Confirmed Records",
                                    axis=alt.Axis(ticks=False, domain=False)
                                    ),
                            y=alt.Y("value", axis=alt.Axis(labels=True, title= kpi_selector,
                                                           ticks=False, domain=False)),
            color=alt.Color('country',
                            legend=alt.Legend(title="Country", orient='top-left')),
            tooltip =['Day Since the First 100 Cumulative Confirmed Records',
                      'country', 'kpi','value' ]).configure_axis(
            grid=False).configure_view(strokeWidth=0,strokeOpacity=0.1).properties(
title = f'{kpi_selector} Timeline Trend').interactive()

        return c


# ---------------------------racing bar page ----------------------------------------------------

def racing_bar(df_):
    st.markdown('###  Racing Bar Chart-- View the developing animation')
    status_selector_ = st.selectbox('Select A Status', ['active', 'death', 'recovered'])
    @st.cache
    def get_hbar_data(df_):
        global top_n_country
        top_n_country = list(
            df_.sort_values(by='count', ascending=False).drop_duplicates(subset='country')['country'].head(
                15))
        df_['month_date'] = df_['date'].dt.strftime("%y/%m/%d")
        con1 = df_['country'].isin(top_n_country)
        con2 = df_['status'] == status_selector_
        con3 = df_['month_date'] >= '20/02/25'
        dff = df_.loc[con1 & con2 & con3]
        dff.sort_values(by=['date', 'count'], ascending=True, inplace=True)
        return dff

    dff = get_hbar_data(df_)
    fig = px.bar(dff, x="count", y="country", animation_frame="month_date", animation_group="country", color='region',
                       hover_name="country", width=1000, height=600, orientation='h')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',  showlegend=True)
    fig.update_layout(yaxis_title='', yaxis_showticklabels=True)
    fig.update_layout(dragmode=False)
    return fig


# def world_sentiment_bar():
#     st.write('Note: the sentiment analysis is based on daily Twitter contents')
#     df = pd.read_csv('sentiment_df.csv')
#     country_sentiment =  df.groupby(['country', 'vader_sentiment']).agg({'name': 'count'})
#     sentiment_pcts = country_sentiment.groupby(level=0).apply(lambda x:
#                                                      round(100 * x / float(x.sum()))).reset_index()
#
#     sentiment_pcts.rename(columns = {'name': 'pct%'}, inplace=True)
#
#     domain = ['positive', 'negative', 'neutral']
#     range_ = ['#90EE90', '#DC143C', '#A9A9A9']
#
#     fig = alt.Chart(sentiment_pcts).mark_bar().encode(
#         x=alt.X('pct%'),
#         y='country',
#         color=alt.Color('vader_sentiment', scale=alt.Scale(domain=domain, range=range_),
#                         legend=alt.Legend(title="Sentiment", orient='right')),
#         tooltip=['country', 'vader_sentiment', 'pct%']).properties(
#         title='COVID Sentiment by Country')
#
#     return st.altair_chart(fig, use_container_width=True)
#
# def tweet_table():
#     df = pd.read_csv('sentiment_df.csv')
#     tweet_table = df.drop(df.columns[0], axis =1)
#     tweet_table.drop(['name', 'retweets', 'location', 'followers', 'is_user_verified'], axis =1, inplace=True)
#     if st.button('View Tweets'):
#         st.table(tweet_table)


def get_db():
    URI = os.environ['URI']
    engine = create_engine(URI)
    sql = """
    select *
    FROM sentiment
    """
    df = pd.read_sql_query(sql, con=engine)
    return df

twitter_db = get_db()

def world_sentiment_bar(df):
    st.write('Note: the sentiment analysis is based on daily Twitter contents')
    country_sentiment =  df.groupby(['country', 'vader_sentiment']).agg({'name': 'count'})
    sentiment_pcts = country_sentiment.groupby(level=0).apply(lambda x:
                                                     round(100 * x / float(x.sum()))).reset_index()

    sentiment_pcts.rename(columns = {'name': 'pct%'}, inplace=True)

    domain = ['positive', 'negative', 'neutral']
    range_ = ['#90EE90', '#DC143C', '#A9A9A9']

    fig = alt.Chart(sentiment_pcts).mark_bar().encode(
        x=alt.X('pct%'),
        y='country',
        color=alt.Color('vader_sentiment', scale=alt.Scale(domain=domain, range=range_),
                        legend=alt.Legend(title="Sentiment", orient='right')),
        tooltip=['country', 'vader_sentiment', 'pct%']).properties(
        title='COVID Sentiment by Country')

    return st.altair_chart(fig, use_container_width=True)

def tweet_table(df):
    tweet_table = df.drop(df.columns[0], axis =1)
    tweet_table.drop(['name', 'retweets', 'location', 'followers', 'is_user_verified'], axis =1, inplace=True)
    if st.button('View Tweets'):
        st.table(tweet_table)


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
        1,  # padding_top
        2,  # padding_right
        0.5,  # padding_left
        0,  # padding_bottom
    )


select_block_container_style()
#



if __name__ == "__main__":
    main()
