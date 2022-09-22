#%%
#Libraries/modules
from decimal import Rounded
from itertools import groupby
import requests
import zipfile
import os
from io import BytesIO
import pandas as pd
import glob
import pickle
import urllib.request
import shutil
import numpy as np

"""
required datasets upload:
Result dataframe - score_df
Filter data for after 2012 only
"""
pd.set_option('display.max_columns', None)

link1 = "https://aicore-files.s3.amazonaws.com/Data-Science/Football.zip"
req=requests.get(link1)
filename=link1.split('/')[-1]
with open(filename,'wb') as output_file:
    output_file.write(req.content)

with zipfile.ZipFile(filename,"r") as zip_ref:     
    zip_ref.extractall()

path = "./Results"
csv_files = glob.glob(path + "/**/*.csv", recursive = True)
results_df = [pd.read_csv(f) for f in csv_files]
score_df  = pd.concat(results_df, ignore_index=True)
os.remove(filename)
shutil.rmtree(path)
score_df = score_df[score_df['Season']>2012]

#Matches - match_df
link2= 'https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv'
match_df = pd.read_csv(link2)

#ELOs - ELO_df
#ELO_scraper.py is the scraper code that scrapes the ELO scores from the website.
pickle_link='https://aicore-files.s3.amazonaws.com/Data-Science/elo_dict.pkl'
myfile = pickle.load(urllib.request.urlopen(pickle_link))
ELO_df=(pd.DataFrame(myfile)).T

#Stadiums - stadium_df
link3='https://aicore-files.s3.amazonaws.com/Data-Science/Team_Info.csv'
stadium_df = pd.read_csv(link3)

"""
Cleaning/preparing Data in match_df:
1. Use link column to  obtain names of the Home_Team, Away_Team and Season to match other databases in the training model.
2. Use Result column to create number of goals per game per team, and "Wins" column where categories are: -1, 0 or 1 (lost, draw, win) 
3. Drop extra columns
"""
score_df.drop(['Unnamed: 0', 'Season'], inplace=True, axis=1)
score_df['Season'] = score_df['Link'].map(lambda x: x.split("/")[-1])
score_df['HT_Link'] = score_df['Link'].map(lambda x: x.split("/")[-3])
score_df['AT_Link'] = score_df['Link'].map(lambda x: x.split("/")[-2])
score_df['Season']=score_df['Season'].str[:4]
score_df[['Home_Score','Away_Score']]=score_df['Result'].str.split('-', expand=True)
score_df['Home_Score'] = pd.to_numeric(score_df['Home_Score'], errors='coerce')
score_df['Away_Score'] = pd.to_numeric(score_df['Away_Score'], errors='coerce')
score_df = score_df.dropna()
cols = ['Season', 'Round', 'Home_Score','Away_Score']
score_df[cols]=score_df[cols].astype('int')

score_df['Wins']=0
score_df['Wins'] = np.where(score_df['Home_Score'] > score_df['Away_Score'], 1, score_df['Wins'])
score_df['Wins'] = np.where(score_df['Home_Score'] < score_df['Away_Score'], -1, score_df['Wins'])
score_df[['Home_Wins','Home_Losses','Home_Draws','Away_Wins','Away_Losses','Away_Draws']]=0 
score_df['Home_Wins'] = np.where(score_df['Home_Score'] > score_df['Away_Score'], 1, score_df['Home_Wins'])
score_df['Home_Losses'] = np.where(score_df['Home_Score'] < score_df['Away_Score'], 1, score_df['Home_Losses'])
score_df['Home_Draws'] = np.where(score_df['Home_Score'] == score_df['Away_Score'], 1, score_df['Home_Draws'])
score_df['Away_Wins']=np.where(score_df['Home_Score'] < score_df['Away_Score'], 1, score_df['Away_Wins'])
score_df['Away_Draws']=score_df['Home_Draws']
score_df['Away_Losses']= np.where(score_df['Home_Score'] > score_df['Away_Score'], 1, score_df['Away_Losses'])

"""
Stadium Dataframe provides info on the Home_Team stadium.
The capacity, the country, the type of pitch
Such information, can be useful for model training.
i.e. Home_team statium capacity can be linked to the pressure/ quality/ etc. of the team.
"""
#Merge stadium information with score dataframe
stadium_df=stadium_df.set_index('Team')
score_df=score_df.set_index('Home_Team')
score_df['Stadium']=''
score_df['Capacity']=''
score_df['Pitch']=''
score_df['Country']=''
score_df.update(stadium_df)
score_df.reset_index(inplace=True)

""" 
Cleaning Data:  MATCH DATAFRAME
1. Strip the Referee data to the referees names only and make it a category (set number of referees only).
2. Split data columns to only year values to turn into Season.
3. Dropped unnecessary columns: the temp ones and Link.
4. Changed data type for the yellow and red cards into integers and dropped NW values (see notes).

Notes:
1. After data cleaning DF size dropped from 7.7MB+ to 6.5MB. 
2. There were 20550 rows with NA values, which is 14.3%. NA values would not be beneficial for the analysis, therefore the values were removed. 
"""

match_df['Referee'] = match_df['Referee'].str.strip('\r\nReferee:')
cols=['Home_Yellow', 'Away_Yellow', 'Home_Red', 'Away_Red']
match_df[cols] = match_df[cols].astype('Int64')
match_df['Referee'] = match_df['Referee'].astype('category')
match_df['Season'] = match_df['Link'].map(lambda x: x.split("/")[-1])
match_df['HT_Link'] = match_df['Link'].map(lambda x: x.split("/")[-3])
match_df['AT_Link'] = match_df['Link'].map(lambda x: x.split("/")[-2])
cols=['Date_New']
match_df.drop(cols,inplace=True, axis=1)
match_df['Season']=match_df['Season'].astype('int')
match_df = match_df[match_df['Season']>2012]

"""
Match_df and ELO_df 
Continue Cleaning/Processing Data:
- dataframe index is a link that includes participating teams and season (first 4 digits)
- drop all info prior to 2012 to match other data
- merge match dataframe and ELO dataframe into one.
"""
ELO_df.reset_index(inplace=True)
ELO_df['Season'] = ELO_df['index'].map(lambda x: x.split("/")[-1])
ELO_df['HT_Link'] = ELO_df['index'].map(lambda x: x.split("/")[-3])
ELO_df['AT_Link'] = ELO_df['index'].map(lambda x: x.split("/")[-2])
ELO_df['Season']=ELO_df['Season'].astype('str')
ELO_df['Season']=ELO_df['Season'].str[:4]
ELO_df['Season']=ELO_df['Season'].astype('int')
ELO_df = ELO_df[ELO_df['Season']>2012]
ELO_df.drop('index',inplace=True, axis=1)
merged = pd.merge(match_df,ELO_df, how ='left',on=['HT_Link', 'AT_Link','Season'])
merged.drop('Link',inplace=True, axis=1)
"""
FINAL STEP
Final Merge, merge together score_df and merged dataframes.
The final_df consists of data that combines scores, EOL per team, statium data and cards given per each game.
All NA are removed and data includes all seasons since 2012 only.
"""

final_df=pd.merge(score_df,merged, how ='left',on=['HT_Link', 'AT_Link','Season'])

"""
After dropping all NA values the database is at 35104 rows instead of 40464
It is about 13% of data lost.
Which is significant, however might create incosistency with the training model.
There are still significnat number of data points to use for the model training.

"""
final_df= final_df.dropna()
final_df['League']=final_df['League'].astype('category')
final_df['Pitch']=final_df['Pitch'].str.lower()
final_df['Elo_home']=final_df['Elo_home'].astype('int')
final_df['Elo_away']=final_df['Elo_away'].astype('int')
final_df=final_df.sort_values(by=['Season', 'League', 'Round'])

"""
ADDITIONAL DATA POINTS
We are going to create the following additional data per team per season
* scores up to date per team per season
* resultant streaks
* total home/away wins
* total home/away losses
* total home/away draws

"""

final_df['HT_home_total_scores_season']=final_df.groupby(['Season','League','Home_Team'])['Home_Score'].cumsum()
final_df['AT_away_total_scores_season']=final_df.groupby(['Season','League','Away_Team'])['Away_Score'].cumsum()
final_df['HT_home_total_wins_season']=final_df.groupby(['Season','League','Home_Team'])['Home_Wins'].cumsum()
final_df['AT_away_total_wins_season']=final_df.groupby(['Season','League','Away_Team'])['Away_Wins'].cumsum()
final_df['HT_home_total_losses_season']=final_df.groupby(['Season','League','Home_Team'])['Home_Losses'].cumsum()
final_df['AT_away_total_losses_season']=final_df.groupby(['Season','League','Away_Team'])['Away_Losses'].cumsum()
final_df['HT_home_total_draws_season']=final_df.groupby(['Season','League','Home_Team'])['Home_Draws'].cumsum()
final_df['AT_away_total_draws_season']=final_df.groupby(['Season','League','Away_Team'])['Away_Draws'].cumsum()

final_df=final_df[['Home_Team', 'Away_Team','Link','Result','Wins','Season', 'League', 'Round', 'Referee','Stadium','Capacity', 'Pitch', 'Country',\
    'HT_Link','Elo_home','Home_Score', 'HT_home_total_scores_season','HT_home_total_wins_season', 'HT_home_total_losses_season',\
    'HT_home_total_draws_season' , 'Home_Yellow', 'Home_Red',\
    'AT_Link', 'Elo_away', 'Away_Score', 'AT_away_total_scores_season', 'AT_away_total_wins_season', 'AT_away_total_losses_season',\
    'AT_away_total_draws_season', 'Away_Yellow','Away_Red']]




#%%
"""
To create "scores_up-to_date" additional values, the code is set up to calculate cummulative scores per team per season
The code obtains scores for each team, regardless if it is Home or Away.
The Home_score_so_far and Away_score_so_far  are columns representing cumulative scores per team per season
The scores reset every season.
"""
final_df['HT_total_score_so_far_season']=0
final_df['AT_total_score_so_far_season']=0
for year in final_df['Season'].unique():
    for league in final_df['League'].unique():
        try_one=final_df[(final_df['Season']==year) & (final_df['League']==league)]
        for team in try_one['Home_Team']:
            cumul=0
            for ind in try_one.index:
                if try_one['Home_Team'][ind]==team:
                    try_one['HT_total_score_so_far_season'][ind]=try_one['Home_Score'][ind]+cumul
                    cumul=try_one['HT_total_score_so_far_season'][ind]
                elif try_one['Away_Team'][ind]==team:
                    try_one['HT_total_score_so_far_season'][ind]=try_one['Away_Score'][ind]+cumul
                    cumul=try_one['HT_total_score_so_far_season'][ind]
        
        for team in try_one['Away_Team']:
            cumul=0
            for ind in try_one.index:
                if try_one['Away_Team'][ind]==team:
                    try_one['AT_total_score_so_far_season'][ind]=try_one['Away_Score'][ind]+cumul
                    cumul=try_one['AT_total_score_so_far_season'][ind]
                elif try_one['Home_Team'][ind]==team:
                    try_one['AT_total_score_so_far_season'][ind]=try_one['Home_Score'][ind]+cumul
                    cumul=try_one['AT_total_score_so_far_season'][ind]
        
        final_df['HT_total_score_so_far_season']=try_one['HT_total_score_so_far_season']
        final_df['AT_total_score_so_far_season']=try_one['AT_total_score_so_far_season']


#%%
final_df=final_df[['Home_Team', 'Away_Team','Link','Result','Wins','Season', 'League', 'Round', 'Referee','Stadium','Capacity', 'Pitch', 'Country',\
    'HT_Link','HT_total_score_so_far_season','Elo_home','Home_Score', 'HT_home_total_scores_season','HT_home_total_wins_season', 'HT_home_total_losses_season',\
    'HT_home_total_draws_season' , 'Home_Yellow', 'Home_Red',\
    'AT_Link', 'AT_total_score_so_far_season','Elo_away', 'Away_Score', 'AT_away_total_scores_season', 'AT_away_total_wins_season', 'AT_away_total_losses_season',\
    'AT_away_total_draws_season', 'Away_Yellow','Away_Red']]
# %%
final_df.to_csv(r'football_cleaned_data.csv', index = False)


# %%
