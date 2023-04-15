#############################
####       PACKAGES      ####
#############################

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from altair import expr, datum
import geopandas as gpd
from shapely import wkt
import folium
from streamlit_folium import st_folium
import base64


st.set_page_config(layout = 'wide')

@st.cache_data

def full_initialisation():
    df = pd.read_csv("data/malaria_df_global.csv")
    df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    df["Gdp/capita"] = (df.Gdp/df.Pop_tot).astype(float)
    rename_dic = {"Congo, Dem. Rep." :  "Dem. Rep. Congo",
              'Congo, Rep.' : "Congo",
              "Cote d'Ivoire" : "Côte d'Ivoire",
              "Egypt, Arab Rep." : "Egypt",
              "Eswatini" : "eSwatini",
              "Gambia, The" : "Gambia",
              "Guinea-Bissau" : "Guinea-Bissau" , 
              "Sao Tome and Principe" : "Sao Tome And Principe",
              "South Sudan" : "S. Sudan",
              "Central African Republic" : "Central African Rep.",
              "Equatorial Guinea" : "Eq. Guinea",
    }
    df.replace(rename_dic, inplace=True)
    df.fillna(method="ffill", inplace=True)
    #df temperature and precipitation
    
    df_t_p = pd.read_csv("data/africa_temp_precip.csv", index_col=0)
    if "geometry" in df_t_p.columns:
        geometry = df_t_p['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df_t_p, geometry=geometry, crs=4326)
    #Find the center point
    gdf['Center_point'] = gdf['geometry'].centroid
    #Extract lat and lon from the centerpoint
    gdf["long"] = gdf.Center_point.map(lambda p: p.x)
    gdf["lat"] = gdf.Center_point.map(lambda p: p.y)
    # shapefile
    #Averaged df
    average_df = df.groupby('Year').mean(numeric_only=True).reset_index()

    #FULL PARAMS
    full_params = average_df.drop(columns=['Year']).columns
    full_countries = df['Country Name'].unique()

    # ALL DIFF
    all_diff_df = pd.merge(
        df,
        average_df,
        on='Year',
        how='inner',
        suffixes=['_country', '_avg']
    )

    for indic in full_params:
        all_diff_df[indic] = all_diff_df[indic+'_country'] - all_diff_df[indic+'_avg']

    #THRESHOLDS
    min_thresh = {indic: all_diff_df[indic].min() for indic in full_params}
    max_thresh = {indic: all_diff_df[indic].max() for indic in full_params}

    return df, all_diff_df, min_thresh, max_thresh, full_params, full_countries, gdf


def get_df_filtered(df, year,countries):
    df_=df.copy()
    df_= df_[df_.Year == year]
    df_.set_index('Country Name', inplace = True)
    df_ = df_[["Incidence rates", "Gdp/capita"]]
    df_["Gdp/capita"] = df_["Gdp/capita"].astype(float)
    df_["Gdp/capita"].fillna(0,inplace=True)
    df_["Gdp/capita"].replace({"":0}, inplace=True)
    df_["Gdp/capita"] = (-1)*df_["Gdp/capita"]
    
    df_.sort_values("Incidence rates", ascending=False, inplace=True)
    df_filtered = df_.loc[countries]
    df_filtered.loc[df_.iloc[0].name] = df_.iloc[0]
    df_filtered.loc[df_.sort_values('Gdp/capita', ascending=True).iloc[0].name] = df_.sort_values('Gdp/capita', ascending=True).iloc[0]
    df_filtered.loc["mean"] = [df_["Incidence rates"].mean() , df_["Gdp/capita"].mean()]
    
    return df_filtered.sort_values("Incidence rates", ascending=False)


def get_neighbouring_countries(gdf, selected_country):
    countries = {} # Maps country name to Shapely geometry.
    gdf_c = gdf.set_index("Country")
    for country in gdf["Country"].unique():
        
        outline = gdf_c.loc[country , "geometry"][0]

        countries[country] = outline

    border_country = [selected_country]
    outline = countries[selected_country]
    for other_country in sorted(countries.keys()):

        if selected_country == other_country: continue

        other_outline = countries[other_country]

        if outline.touches(other_outline):
            border_country.append(other_country)
    return border_country



def bar_chart(df_, country):
    
    base = alt.Chart(df_.reset_index()).properties(    
    width=1000,
    height=300
    )
    chart1 = base.mark_bar().encode(
        x=alt.X('Country Name', title=None, axis=None, sort='-y'),
        y=alt.Y('Incidence rates'),
        color = alt.condition(
            alt.datum["Country Name"] == country,  # If the year is 1810 this test returns True,
            alt.value('orange'),     # which sets the bar orange.
            alt.value('#fd625e')   # And if it's not true it sets the bar steelblue.
        )
    )
    chart2 = base.mark_bar().encode(
        x=alt.X('Country Name', title=None, axis=alt.Axis(labels=True), sort=alt.EncodingSortField(field="Incidence rates", order='descending')),
        y=alt.Y('Gdp/capita'),
        color = alt.condition(
            alt.datum["Country Name"] == country,
            alt.value('orange'),     # which sets the bar orange.
            alt.value('#01b8aa'))
        )

    chart = alt.vconcat(chart1, chart2, spacing=0).properties(
        title="Incidence rates per 1000 population vs GDP per capita in {} and neighbors countries".format(country)
    )

    return chart

def chart_time_serie(all_diff_df, params_to_plot, country):
    #The df to plot
    selected_col = [string for string in all_diff_df if any(substring in string for substring in params_to_plot)]
    try: 
        diff_df = all_diff_df[all_diff_df['Country Name'] == country][selected_col+['Year']]
    except:
        st.markdown(f'''
        No data available for {country}
        ''', unsafe_allow_html=True)
        return None

    charts = []
    for i, indic in enumerate(params_to_plot):
        chart = alt.Chart(diff_df).mark_rect().encode(
        x='Year:O',
        color=alt.Color(
            f'{indic}:Q', 
            scale=alt.Scale(scheme='redblue', domain=(min_thresh[indic], max_thresh[indic])),
            # field = indic
            ),
        tooltip=[
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip(f'{indic}_country:Q', title=f'{indic} for {country}'),
            alt.Tooltip(f'{indic}_avg:Q', title=f'Average {indic}')
        ]
        ).properties(
                width=1000,
                height=100
                )

        chart_line = alt.Chart(diff_df).mark_line().encode(
        x='Year:O',
        y=alt.Y(f'{indic}_country:Q', title = None),
        # y=alt.Y('y', scale=alt.Scale(reverse=True))
        color=alt.value('grey')
        ).properties(width=1000, height=300)

        charts.append(chart+chart_line)

    # Combine the charts vertically
    combined_chart = alt.vconcat(*charts).resolve_scale(color='independent')

    return combined_chart


def display_map(gdf,df, year, type, display_incidence_rates=True):
    if type=="Temperature":
        data_name = "AverageTemperature"
        data_display_name = "Average Temperature"
        data_unit = "°C"
        fill_color = "YlOrRd"
        loc = "Average Temperature in °C"
    else:
        data_name = "Precipitation (mm/year)"
        data_display_name = "Precipitation"
        data_unit = "mm/year"
        fill_color = "Blues"
        loc = "Precipitation in mm/year"
    
    gdf_year = gdf[gdf.Year==year]
    df_year = df[df.Year == year]
    #df_year.drop(["Country"], axis=1, inplace=True)
    df_final_year = gdf_year.merge(df_year, how="left", on= ["Country Code", "Year"])
    
    #st.write(df_final_year.drop(["geometry"], axis=1))
    geojson = gdf.drop(["Center_point"], axis=1).to_json()
    m = folium.Map(location=(0,25), zoom_start=3, scrollWheelZoom=False,tiles= 'cartodbpositron')
    folium.GeoJson(geojson).add_to(m)
    choropleth = folium.Choropleth(geo_data=geojson,
                                    data=gdf_year,
                                    columns=["Country", data_name],
                                    key_on="feature.properties.Country",
                                    fill_color=fill_color,
                                    nan_fill_color="grey",
                                    legend_name=data_display_name,
                                    highlight=True
                                    )
    
    choropleth.add_to(m)
    
    for feature in choropleth.geojson.data["features"]:
        df1 = gdf.set_index("Country")
        df2 = df_final_year.set_index("Country")
        country_name = feature["properties"]["Country"]
        
        
        feature["properties"][data_display_name] = data_display_name + ": " + str('{:.2f}'.format(df1.loc[country_name, data_name][0])) + data_unit     
        try:
            feature["properties"]["Incidence rate"] = "Incidence rate: "+ str('{:.2f}'.format(df2.loc[country_name, "Incidence rates"][0])) + " per 1000 pop."    
        except:
            feature["properties"]["Incidence rate"] = "Incidence rate: "+ str('{:.2f}'.format(df2.loc[country_name, "Incidence rates"])) + " per 1000 pop."    

    if display_incidence_rates:
        for index, row in df_final_year.iterrows():
            if row["Incidence rates"]>0:
                folium.CircleMarker(location=[row["lat"], row["long"]],
                                radius=row["Incidence rates"]/20,
                                fill=True,
                                fill_opacity=0.1,
                                color="black"
                                #tooltip=folium.features.GeoJsonTooltip(["Country",data_display_name, "Incidence rate"], labels=False)
                                ).add_to(m)
                
            
                                
    
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(["Country",data_display_name, "Incidence rate"], labels=False)
    )
    selected_country = ''
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format(loc)   

    m.get_root().html.add_child(folium.Element(title_html))
    
    
    
    st_map = st_folium(m, width=600)
    if st_map["last_active_drawing"]:
        selected_country = st_map["last_active_drawing"]["properties"]["Country"]

    return selected_country






df, all_diff_df, min_thresh, max_thresh, full_params, full_countries, gdf = full_initialisation()

#######################################
####       SIDE BAR STRUCTURE      ####
#######################################

# Set header title
st.title('Africa facing Malaria')



max_date = int(gdf.Year.max())
min_date = int(gdf.Year.min())
year = st.slider(
    "Select date:",
    min_date,
    max_date,
    max_date)
# Plot map
col1, col2 = st.columns(2)
selected_country = ''
if "prec_selec" not in st.session_state:
    st.session_state.prec_selec = ''
else:
    selected_country = st.session_state.prec_selec

with col1:
    col1.markdown("<h3 style='text-align: center;'>Average Temperature in °C</h3>", unsafe_allow_html=True)
    selected_country_temp = display_map(gdf, df, year, "Temperature")
with col2:
    col2.markdown("<h3 style='text-align: center;'>Precipitation in mm/year</h3>", unsafe_allow_html=True)
    selected_country_precip = display_map(gdf, df, year, "Precipitation")

if st.session_state.prec_selec != selected_country_temp:
    selected_country = selected_country_temp
    st.session_state.prec_selec   = selected_country_temp
elif st.session_state.prec_selec  != selected_country_precip:
    selected_country = selected_country_precip
    st.session_state.prec_selec = selected_country_precip
    
    
with col1:
    st.metric("Selected Country: " ,selected_country)
with col2:
    params_to_plot = st.multiselect('Select the parameter(s) to visualize', full_params)




if len(selected_country) == 0 and "prec_selec" in st.session_state:
      selected_country = st.session_state.prec_selec
    
if len(selected_country) > 0:
    countries = get_neighbouring_countries(gdf, selected_country)
    df_filtered= get_df_filtered(df, 2015,countries)
    st.altair_chart(bar_chart(df_filtered, selected_country), use_container_width=True)
    
    
    
    if len(params_to_plot) == 0:
        st.text('Choose at least 1 param to get started')
    else:
        time_serie_chart = chart_time_serie(all_diff_df, params_to_plot, selected_country)
        if time_serie_chart != None:
            st.markdown(f'''
            #### Criteria evolution difference of {selected_country} to the African average
            ''', unsafe_allow_html=True)

            st.altair_chart(time_serie_chart, use_container_width=True)

            st.markdown(f'''
            NB: The grey line show the trend of the selected country.
            ''', unsafe_allow_html=True)
        else: 
            pass
