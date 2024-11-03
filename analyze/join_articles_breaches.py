# Can use this once we have all articles and are ready to join with breaches

import pandas as pd

# organisation, alternative name, records lost, year, date, story, sector,
# method, interesting story, data sensitivity,displayed records, source name,
# 1st source link, 2nd source link, ID
info = pd.read_csv('../data/breaches/breaches_information.csv', skiprows=range(1, 26))
info = info.rename(columns={'year   ': 'year'})
info['records lost'] = info['records lost'].str.replace(',', '')
info['records lost'] = pd.to_numeric(info['records lost'])
info = info.drop(columns=['Unnamed: 11'])
