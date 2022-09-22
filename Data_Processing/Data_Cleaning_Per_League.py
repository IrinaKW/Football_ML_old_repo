#%%
#Libraries/modules
from decimal import Rounded
from itertools import groupby
from operator import index
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
import re

pd.set_option('display.max_columns', None)

def longest_win(string):
    try: 
        return max(len(i) for i in re.findall("W+", string))
    except:
        return 0

def longest_loss(string):
    try: 
        return max(len(i) for i in re.findall("L+", string))
    except:
        return 0

#DATA FILES are being uploaded
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
stadium_df['Team'].replace(['Gimnàstic','Mönchengladbach','Eintracht','Würzburger','Fortuna',\
    'Evian Thonon Gail.','Olympique','Queens Park Range.','Brighton Hove Alb.','Paços Ferreira',\
    'Sheffield', 'West Bromwich Alb.'],\
    ['Gimnàstic Tarragona', 'B. Mönchengladbach', 'Eintracht Frankfurt', 'Würzburger Kickers', 'Fortuna Düsseldorf',\
    'Evian Thonon Gaillard','Olympique Marseille','Queens Park Rangers', 'Brighton & Hove Albion','Paços Ferreira',\
    'Sheffield Wednesday', 'West Bromwich Albion'], inplace=True)

stadium_df=stadium_df.set_index('Team')

""" 
Cleaning Data:  MATCH DATAFRAME
1. Strip the Referee data to the referees names only and make it a category (set number of referees only).
2. Split data columns to only year values to turn into Season.
3. Dropped unnecessary columns: the temp ones and Link.
4. Changed data type for the yellow and red cards into integers and dropped NW values (see notes).
"""
cols=['Home_Yellow', 'Away_Yellow', 'Home_Red', 'Away_Red']
match_df[cols] = match_df[cols].astype('Int64')
match_df['Season'] = match_df['Link'].map(lambda x: x.split("/")[-1])
match_df['HT_Link'] = match_df['Link'].map(lambda x: x.split("/")[-3])
match_df['AT_Link'] = match_df['Link'].map(lambda x: x.split("/")[-2])
match_df['HT_Game_Penalty_Cards']=match_df['Home_Yellow']+match_df['Home_Red']
match_df['AT_Game_Penalty_Cards']=match_df['Away_Yellow']+match_df['Away_Red']
cols=['Date_New','Link','Referee','Home_Yellow', 'Away_Yellow', 'Home_Red', 'Away_Red']
match_df.drop(cols,inplace=True, axis=1)
match_df['Season']=match_df['Season'].astype('int')
match_df = match_df[match_df['Season']>2012]

"""
MERGE: Match_df and ELO_df 
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
ELO_df=ELO_df.dropna()
ELO_df[['Season','Elo_home','Elo_away']]=ELO_df[['Season','Elo_home','Elo_away']].astype('int')
ELO_df = ELO_df[ELO_df['Season']>2012]
ELO_df.drop('index',inplace=True, axis=1)
merged = pd.merge(match_df,ELO_df, how ='left',on=['HT_Link', 'AT_Link','Season'])


# UPLOAD DATA per league
link1 = "https://aicore-files.s3.amazonaws.com/Data-Science/Football.zip"
req=requests.get(link1)
filename=link1.split('/')[-1]
with open(filename,'wb') as output_file:
    output_file.write(req.content)
with zipfile.ZipFile(filename,"r") as zip_ref:     
    zip_ref.extractall()

folder_list=glob.glob("./Results/*/", recursive = True)
folder_list = [item.split('/')[-2] for item in folder_list]

#Next step it to create the csv files for the cleaned/ prepared data per each league
for league in folder_list:
    path = "./Results/"+league
    results_df=[]
    clean_df=pd.DataFrame()
    final_df=pd.DataFrame()
    csv_files = glob.glob(path + "/*.csv", recursive = True)
    results_df = [pd.read_csv(f) for f in csv_files]
    clean_df  = pd.concat(results_df, ignore_index=True)
    clean_df.drop('Unnamed: 0',inplace=True, axis=1, errors='ignore')
    clean_df=clean_df[clean_df['Season']>2012]

    #Step 1: Split RESULT column to create goals per team and WIN column for outcome
    clean_df[['Home_Score','Away_Score']]=clean_df['Result'].str.split('-', expand=True)
    clean_df['Home_Score'] = pd.to_numeric(clean_df['Home_Score'], errors='coerce')
    clean_df['Away_Score'] = pd.to_numeric(clean_df['Away_Score'], errors='coerce')

    clean_df = clean_df.dropna()
    cols = ['Season', 'Round', 'Home_Score','Away_Score']
    clean_df[cols]=clean_df[cols].astype('int')
    clean_df['Outcome']=0
    clean_df['Outcome'] = np.where(clean_df['Home_Score'] > clean_df['Away_Score'], 1, clean_df['Outcome'])
    clean_df['Outcome'] = np.where(clean_df['Home_Score'] < clean_df['Away_Score'], -1, clean_df['Outcome'])
    
    #Step 2: Create Wins/Draws/Losses home and away based on the above
    clean_df[['Home_Wins','Home_Losses','Home_Draws','Away_Wins','Away_Losses','Away_Draws']]=0 
    clean_df['Home_Wins'] = np.where(clean_df['Home_Score'] > clean_df['Away_Score'], 1, clean_df['Home_Wins'])
    clean_df['Home_Losses'] = np.where(clean_df['Home_Score'] < clean_df['Away_Score'], 1, clean_df['Home_Losses'])
    clean_df['Home_Draws'] = np.where(clean_df['Home_Score'] == clean_df['Away_Score'], 1, clean_df['Home_Draws'])
    clean_df['Away_Wins']=np.where(clean_df['Home_Score'] < clean_df['Away_Score'], 1, clean_df['Away_Wins'])
    clean_df['Away_Draws']=clean_df['Home_Draws']
    clean_df['Away_Losses']= np.where(clean_df['Home_Score'] > clean_df['Away_Score'], 1, clean_df['Away_Losses'])

    #Step 2: use LINK column to obtain team names and season that match stadium and match dataframes in order to merge them
    clean_df['Season'] = clean_df['Link'].map(lambda x: x.split("/")[-1])
    clean_df['HT_Link'] = clean_df['Link'].map(lambda x: x.split("/")[-3])
    clean_df['AT_Link'] = clean_df['Link'].map(lambda x: x.split("/")[-2])
    clean_df['Season']=clean_df['Season'].str[:4]
    clean_df=clean_df.sort_values(by=['Season','Round'])

    #Step 3: merge additional data
    """
    Stadium Dataframe provides info on the Home_Team stadium.
    The capacity, the country, the type of pitch
    Such information, can be useful for model training.
    i.e. Home_team statium capacity can be linked to the pressure/ quality/ etc. of the team.
    """
    #Merge stadium information with score dataframe
    clean_df=clean_df.set_index('Home_Team')
    clean_df['Capacity']=''
    clean_df['Pitch']=''
    clean_df['Country']=''
    clean_df.update(stadium_df)
    clean_df.reset_index(inplace=True)

    """
    FINAL STEP
    Final Merge, merge together score_df and merged dataframes.
    The final_df consists of data that combines scores, EOL per team, statium data and cards given per each game.
    All NA are removed and data includes all seasons since 2012 only.
    """
    clean_df['Season']=clean_df['Season'].astype('int')
    final_df=pd.merge(clean_df,merged, how ='left',on=['HT_Link', 'AT_Link','Season'])
    final_df= final_df.dropna()
    final_df=final_df.sort_values(by=['Season', 'League', 'Round'])

    #Step 4: convert Pitch and Country categorical type variables into numeric values:
    # There are 3 types of pitches: grass(1), artificial(2) and hybrid(3) 
    final_df['Pitch']=final_df['Pitch'].str.lower()
    final_df['Country']=final_df['Country'].str.lower()
    final_df['Pitch'].replace(['','natural', 'cesped real', 'césped artificial', 'airfibr ', \
        'césped natural', 'grass', 'césped', 'cesped natural'], [0,1,1,2,3,1,1,1,1], inplace=True)
    final_df['Country'].replace(['','germany', 'netherlands', 'france', 'england', 'portugal',\
        'spain', 'italy'], [0,1,2,3,4,5,6,7], inplace=True)

    #Step 5: create cummulative scores, cards and resultant streak per team, per season
    cummulate={}
    cumm_df = pd.DataFrame()
    final_df = final_df.reset_index(drop=True) 
    for team in final_df['Home_Team'].unique():
        for index, row in final_df.iterrows():
            if row['Home_Team']==team:
                cummulate['Score']=row['Home_Score']
                cummulate['Cards']=row['HT_Game_Penalty_Cards']
                cummulate['Wins']=row['Home_Wins']
                cummulate['Draws']=row['Home_Draws']
                cummulate['Losses']=row['Home_Losses']
                if row['Outcome']==1:
                    cummulate['Streak']='W'
                elif row['Outcome']==0:
                    cummulate['Streak']='D'
                else: cummulate['Streak']='L'
            elif row['Away_Team']==team:
                cummulate['Score']=row['Away_Score']
                cummulate['Cards']=row['AT_Game_Penalty_Cards']
                cummulate['Wins']=row['Away_Wins']
                cummulate['Draws']=row['Away_Draws']
                cummulate['Losses']=row['Away_Losses']
                if row['Outcome']==1:
                    cummulate['Streak']='L'
                elif row['Outcome']==0:
                    cummulate['Streak']='D'
                else: cummulate['Streak']='W'
            else:
                continue
            cummulate['Team'] = team
            cummulate['Season']=row['Season']
            cummulate['Round']= row['Round'] 
            temp_df = pd.DataFrame([cummulate])
            cumm_df = pd.concat([cumm_df, temp_df], ignore_index=True)

    cumm_df=cumm_df.sort_values(by=['Season','Team','Round'])
    cumm_df=cumm_df.reset_index(drop=True) 
    cumm_df['Cum_Cards'] = (cumm_df.groupby(['Season','Team'])['Cards'].transform(\
        lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
    cumm_df['Cum_Scores'] = (cumm_df.groupby(['Season','Team'])['Score'].transform(\
        lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
    cumm_df['Cum_Wins'] = (cumm_df.groupby(['Season','Team'])['Wins'].transform(\
        lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
    cumm_df['Cum_Draws'] = (cumm_df.groupby(['Season','Team'])['Draws'].transform(\
        lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
    cumm_df['Cum_Losses'] = (cumm_df.groupby(['Season','Team'])['Losses'].transform(\
        lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
    cumm_df['Cum_Streak'] = (cumm_df.groupby(['Season','Team'])['Streak'].transform(\
        lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
    cumm_df['Cum_Streak']=cumm_df['Cum_Streak'].astype('string')
    cumm_df['Longest_Win_Streak']=cumm_df['Cum_Streak'].apply(longest_win)
    cumm_df['Longest_Loss_Streak']=cumm_df['Cum_Streak'].apply(longest_loss)


    #Step 6: update initial dataset and leave only neccessary data for model training
    final_df=pd.merge(final_df, cumm_df, how='left', left_on=['Home_Team','Season','Round'], right_on=['Team','Season','Round'])

    final_df=final_df[['Outcome','Home_Team', 'Away_Team', 'Season', 'Round',\
        'Capacity', 'Pitch', 'Country', 'Elo_home', 'Elo_away',\
        'Cum_Cards', 'Cum_Scores', 'Cum_Wins', 'Cum_Draws', 'Cum_Losses', 'Cum_Streak',\
        'Longest_Win_Streak', 'Longest_Loss_Streak']]

    final_df.rename(columns = {'Cum_Cards':'HT_Cum_Cards','Cum_Scores':'HT_Cum_Scores',\
        'Cum_Wins':'HT_Cum_Wins','Cum_Draws':'HT_Cum_Draws',\
        'Cum_Losses':'HT_Cum_Losses','Cum_Streak':'HT_Cum_Streak',\
        'Longest_Win_Streak': 'HT_Longest_Win_Streak', 'Longest_Loss_Streak':'HT_Longest_Loss_Streak'}, inplace = True)

    final_df=pd.merge(final_df, cumm_df, how='left', left_on=['Away_Team','Season','Round'], right_on=['Team','Season','Round'])

    final_df=final_df[['Outcome','Home_Team', 'Away_Team', 'Season', 'Round',\
        'Capacity', 'Pitch', 'Country', 'Elo_home', 'Elo_away',\
        'HT_Cum_Cards', 'HT_Cum_Scores', 'HT_Cum_Wins', 'HT_Cum_Draws', 'HT_Cum_Losses', 'HT_Cum_Streak',\
        'HT_Longest_Win_Streak', 'HT_Longest_Loss_Streak',\
        'Cum_Cards', 'Cum_Scores', 'Cum_Wins', 'Cum_Draws', 'Cum_Losses', 'Cum_Streak',\
        'Longest_Win_Streak', 'Longest_Loss_Streak']]

    final_df.rename(columns = {'Cum_Cards':'AT_Cum_Cards','Cum_Score':'AT_Cum_Score',\
        'Cum_Wins':'AT_Cum_Wins','Cum_Draws':'AT_Cum_Draws',\
        'Cum_Losses':'AT_Cum_Losses','Cum_Streak':'AT_Cum_Streak',\
        'Longest_Win_Streak': 'AT_Longest_Win_Streak', 'Longest_Loss_Streak':'AT_Longest_Loss_Streak'}, inplace = True)

    #Step 7: write data into csv file
    os.makedirs('clean_data', exist_ok=True)  
    csv_filename='clean_data/'+league+'.csv'
    final_df.to_csv(csv_filename, index=False)  

os.remove(filename)
shutil.rmtree(path)

#%%

