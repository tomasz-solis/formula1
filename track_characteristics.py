import fastf1 as ff1
import pandas as pd
from datetime import datetime
import numpy as np
from fastf1.ergast import Ergast
import requests



pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

ff1.Cache.enable_cache('data')



def get_driver_max_throttle_ratio(session, driver, max_throttle_threshold = 98):
    """
    Get the max throttle ratio for the fastest lap in the session for a selected driver.
    Parameters:
    - session - loaded session (i.e. session.load() must be called before)
    - driver - string, abbreviated driver name,
    - max_throttle_threshold - optional parameter, the threshold used to categorise the readout as "MAX THROTTLE".
      100 is not suggested due to readout errors. Default 98.

    Returns a dataframe:
    - driver
    - fastest lap max throttle ratio
    - tyre info (age, compound, fresh check)
    - weather info (rainfall rate - part of the lap with the rain, avg track temp, avg air temp)
    """

    gp_name = session.event['EventName']
    race_weekend = session.event['Location']

    full_throttle = pd.DataFrame(columns = [
      'grand_prix',
      'location',
      'driver',
      'ratio',
      'compound',
      'tyre_age',
      'is_fresh_tyre',
      'avg_rainfall',
      'avg_track_temp',
      'avg_air_temp'])
    missing_info = pd.DataFrame(columns = [
      'grand_prix',
      'location',
      'driver'
      ])

    try:
      fastest_driver = session.laps.pick_driver(driver).pick_fastest()
      telemetry_driver = fastest_driver.get_telemetry().add_distance()
    
      #add weather info to the telemetry data
      telemetry_driver = pd.merge_asof(
          telemetry_driver,
          session.weather_data[['Time','Rainfall','TrackTemp', 'AirTemp']],
          left_on = 'SessionTime',
          right_on = 'Time'
          )

      #add info about the next/preious throttle input change
      telemetry_driver['nextThrottle'] = telemetry_driver.Throttle.shift(-1)
      telemetry_driver['previousThrottle'] = telemetry_driver.Throttle.shift(1)

      telemetry_driver_ltd = telemetry_driver.loc[
              (telemetry_driver.Throttle>=max_throttle_threshold)
              &((telemetry_driver.Throttle.shift(-1)<max_throttle_threshold)
                |((telemetry_driver.Throttle.shift(1)<max_throttle_threshold)
                  |(telemetry_driver.index.isin([telemetry_driver.index[0],telemetry_driver.index[-1]]))))
                  ].copy()

      #calculate the relative distance difference between portions of the track
      telemetry_driver_ltd['FTRelative'] = telemetry_driver_ltd.RelativeDistance - telemetry_driver_ltd.RelativeDistance.shift(1)

      telemetry_driver_ltd.FTRelative.fillna(0, inplace=True)

      #take every other row - i.e. include only rows with full throttle 
      ratio = telemetry_driver_ltd.loc[
          (telemetry_driver_ltd.nextThrottle<max_throttle_threshold)
          |(telemetry_driver_ltd.nextThrottle.isna())].FTRelative.sum()

      #create a dataframe with the results
      df = pd.DataFrame([{
         'grand_prix': gp_name,
         'location':race_weekend,
         'driver':driver,
         'ratio':ratio,
          'compound':fastest_driver['Compound'],
          'tyre_age':fastest_driver['TyreLife'],
          'is_fresh_tyre':fastest_driver['FreshTyre'],
          'avg_rainfall':telemetry_driver['Rainfall'].mean(),
          'avg_track_temp':telemetry_driver['TrackTemp'].mean(),
          'avg_air_temp':telemetry_driver['AirTemp'].mean()
          }])
      
      full_throttle = pd.concat([full_throttle, df], ignore_index=True, axis = 0)
      
    except KeyError:
      # in some cases we do not have the telemetry data due to tech issues.
      # this dataframe is created to keep a note about that.
      df_dictionary = pd.DataFrame([{
         'grand_prix':gp_name,
         'location':race_weekend,
         'driver':driver
         }])
      missing_info = pd.concat([missing_info, df_dictionary], ignore_index=True, axis = 0)

    return full_throttle,missing_info


def get_all_drivers_throttle_input(session):
    """
    Get the max throttle driven single rap ratio for all drivers.
    Params:
    - session - race weekend session (session.load() must be called before)\n
    Returns a dataframe containing:
    - year
    - race weekend name
    - session id
    - driver
    - single lap max throttle ratio
    """
    drivers = pd.unique(session.laps['Driver'])

    gp_name = session.event['EventName']
    race_weekend = session.event['Location']

    full_throttle = pd.DataFrame()
    missing_info = pd.DataFrame()

    for driver in drivers:
        #print(driver)
        driver_info, missing_data = get_driver_max_throttle_ratio(session, driver)
        driver_info['grand_prix'] = gp_name
        driver_info['location'] = race_weekend
        full_throttle = pd.concat([full_throttle, driver_info], ignore_index=True, axis = 0)
            
       
        missing_info = pd.concat([missing_info, missing_data], ignore_index=True, axis = 0)

    # removing obvious incorrect readings
    # there is not a single track with max throttle ratio > 85% or < 40%
    full_throttle.loc[(full_throttle.ratio>0.85)|(full_throttle.ratio<0.4), 'ratio'] = np.NaN

    correct_readings = full_throttle.loc[~full_throttle.ratio.isna()]
    incorrect_readings = full_throttle.loc[full_throttle.ratio.isna()]

    missing_info = pd.concat([missing_info, incorrect_readings[['grand_prix','location','driver']]], ignore_index=True, axis = 0)

    return correct_readings, missing_info
    

def get_all_tracks_full_throttle(season):
    """
    Get the info about all races to date in the current Formula 1 season.
    Currently fovering just Free Practice 1. Other sessions will be added in the future.
    Parameters:
    - season - int indicating the year

    Returns 2 dataframes:
    - throttle input df - containing all information about the fastest lap incl. the throttle input, weather and tyres
    - missing info df - containing all cases of missing or incorrect telemetry
    """

    schedule = ff1.get_event_schedule(season)
    races = schedule.loc[
        #every race weekend up BEFORE today
        (schedule.Session1DateUtc < datetime.utcnow())&
        #excluding testing
        (schedule.EventFormat!='testing')].Location

    throttle_df = pd.DataFrame()
    missing_df = pd.DataFrame()

    for race in races:
        print(race)
        session = ff1.get_session(season,race,'FP1')
        session.load()
        full_throttle, missing_info = get_all_drivers_throttle_input(session)

        throttle_df = pd.concat([throttle_df, full_throttle], ignore_index=True, axis = 0)
        
        missing_df = pd.concat([missing_df, missing_info], ignore_index=True, axis = 0)

    return throttle_df, missing_df


def get_open_elevation(latitude, longitude):
    """ 
    Get the elevation of a single point based on coordinates.
    
    Parameters:
    - latitude
    - longitude

    Returns single value - altitude above sea level in meters.
    """
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
    
    response = requests.get(url)
    data = response.json()
    elevation = data["results"][0]["elevation"]

    return elevation


def get_circuits(season):
    """ 
    Get main geolocation info of the tracks:
    - latitude
    - longitude
    - official circuit name
    - altitude above sea level

    Parameters:
    - season - int indicating the year/season of Formula 1

    Returns
    - dataframe containing all info about the track location.
    """

    ergast = Ergast()

    racetracks = ergast.get_circuits(season=2022)

    altitudes = pd.DataFrame()

    for racetrack in racetracks.circuitName:
        latitude = racetracks.loc[racetracks.circuitName==racetrack].iat[0,3]
        longitude = racetracks.loc[racetracks.circuitName==racetrack].iat[0,4]
        altitude = get_open_elevation(latitude,longitude)
        
        df = pd.DataFrame([{
            'circuitName': racetrack,
            'altitude': altitude,
            }])
        
        altitudes = pd.concat([altitudes, df], ignore_index=True, axis = 0)

    racetracks = racetracks.merge(altitudes,how = 'left',on='circuitName')

    return racetracks


# # Work in progress

"""
- function identifying fast/medium/slow corners
    Chickanes?

- function looking at long and lat load on the tires

- function looking at breaking points

"""