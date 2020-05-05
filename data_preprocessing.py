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
new_df['per_mil_count'] = new_df['count']/new_df['population']
new_df.fillna(0, inplace=True)

# calculate the increase rate
dim = ['country', 'region', 'iso_alpha', 'latitude', 'longitude', 'dt_time', 'status']
new_df.sort_values(by=dim, inplace=True)
new_df['diff_change'] = new_df.groupby(['country', 'status'])['count'].apply(pd.Series.pct_change)

#retrieve the latest date
latest_date = max(new_df.dt_time)

