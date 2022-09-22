# Football Match Outcome Prediction
The Football Match Outcome Prediction project: the user processes a large number of files that contain information about football matches that have taken place since 1990. The data has to be cleaned so it can be fed to the model. Then, different models are trained with the dataset, and the best performing model is selected. The hyperparameters of this model are tuned, so its performance is improved.

CONCLUSION:
The best prediction accuracy results has been obtained through SVM_Scaler model training. Scaler means the stadium capacity features had been scaled to match other features.
The score is the same for all sets: 49.32%

The second highest scores been shown by SVM_Ext model. Ext means the normalisation and standartisation preprocessing pipline has been applied to features set.
The score is: 49.20%

After that the results were as follows:
[best score](img/best_score.png)


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Processed and cleaned a dataset with more than 100K samples using pandas.

- Groupby and Cummulative scores with pivot tables approach is used to create new features.

- Carried out Exploratory Data Analysis to form hypotheses on the dataset.

- Trained various models to obtain optimal result.


## Technologies Used
- Python
- Pandas
- Seaborn
- Selenium
- Webdriver / Chrome
- SciKit Learn 


## Features
- Data obtained from the following links:
* https://aicore-files.s3.amazonaws.com/Data-Science/Football.zip - Score Data
* https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv - Match Data
* https://aicore-files.s3.amazonaws.com/Data-Science/Team_Info.csv - Stadium Data 

- Scraper is used to scrape ELO scores for each team for Model training purposes from the provided links. The used website: www.besoccer.com
* Key features: accept cookies and cancel subscription prior to scraping

- Model Training results show Accuracy scores per league as well as for the whole set
* Models used: Logistic Regression, Random Forest, Support Vector Machine (SVM) 
* X value set is used unmodified, scaled and piplined through standartisation and normalisation.


## Screenshots
[Score Table after the download and data cleanup/processing](img/score_dataframe.png)

[Heatmap shows no NA values in the data set with all features](img/na_summary.png)

[Countplot that shows total counts of all outcomes for the Home_Team: -1 for Loss, 0 for Draw, 1 for Win](img/Output_countplot.png)

[Overview of the distribution of all features](img/features_distribution.png)

[Correlation matrix between features/columns](img/corr_matrix_heatmap.png)

[Cleaned Data from 2_liga League as DataFrame](img/clean_data_2_liga.png)

[List of accuracy scores per model for 2_liga](img/Model_accuracy_ligue2.png)

[Mean accuracy values per league](img/mean_values_accuracy_league.png)

[]

## Setup
Required Libraries:
    pandas
    numpy
    seaborn
    matplotlib
    sklearn
    glob
    os
    sys
    time
    selenium
    webdriver_manager.chrome
    decimal
    itertools
    operator
    requests
    zipfile
    io
    pickle
    urllib.request
    shutil


## Usage
Sample of code/data-processing

Upload multiple csv files into one dataframe:
```
#Read all csv files into one pandas df
path = "./Results"
csv_files = glob.glob(path + "/**/*.csv", recursive = True)
results_df = [pd.read_csv(f) for f in csv_files]
pd.set_option('display.max_columns', None)
final_df   = pd.concat(results_df, ignore_index=True)
final_dfs
```

Cleaning Match data
```
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
```

ELO score Scraper Code sample
```    
def scraping(self):
for i in range(3): #(len(self.match_link.index)):
    link=str(self.match_link.loc[i, 'Link'])
    print(link)
    self.driver.get(link)
    time.sleep(5)
    self.driver.find_element(By.XPATH, config.ANALYSIS_XPATH).click()
    time.sleep(2)
    home_elo=self.driver.find_element(By.XPATH, config.HOME_ELO_XPATH).text
    away_elo=self.driver.find_element(By.XPATH, config.AWAY_ELO_XPATH).text
    self.match_link.loc[i, ['Home_ELO']]=home_elo
    self.match_link.loc[i, ['Away_ELO']]=away_elo
    time.sleep(2)
self.match_link.to_csv('match_ELO.csv')
```

Model Training Function and application
```
def train_model (X,Y,model):
    """This function applies pre-defined model to the given set of features and outcomes.
    The train/test split is set at 80/20.
    Return accuracy score

    Args:
        X (dataframe): dataset with features (can be scaled, normalised, standardised)
        Y (array or dataframe): outcomes (set of the results)
        model (assigned model from SkLearn): model is assigned prior to application 
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    return(model.score(X_test, y_test))
#LOGISTIC REGRESSION
LR_model=LogisticRegression()
mscore=train_model(X, Y, LR_model)
temp_df=update_dict('All', 'LogReg', mscore)
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)
```

## Project Status
Project is: _in progress_ 


## Room for Improvement

To do:
- Train the following models: Decision Tree, K-Nearest Neighbour, Naive Bayes


## Acknowledgements
- This project was inspired by AiCore training program


## Contact
Created by [@irinawhite](irina.k.white@gmail.com) - feel free to contact me!

