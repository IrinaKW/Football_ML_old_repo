#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import glob

#%%
#MODEL TRAINING MODULES

#Write in results
Models_results=pd.DataFrame()
scaler = StandardScaler()

def update_dict(n1,n2,n3,n4):
    """Function is used to update dataframe through dictionary update.
    Args:
        n1 (str): Name of the League, obtained from the file name. Or all for all data set
        n2 (str): Name of the model being applied
        n3 (float): Accuracy score calculated after model application to the train set
        n4 (float): Accuracy score calculated after model application to the test set
    """
    result={}
    result['League']=n1
    result['Model']=n2
    result['Train_Accuracy']=n3
    result['Test_Accuracy']=n4
    return(pd.DataFrame([result]))


def train_model (X,Y,model):
    """This function applies pre-defined model to the given set of features and outcomes.
    The train/test split is set at 80/20.
    Return accuracy scores for train set and for test set

    Args:
        X (dataframe): dataset with features (can be scaled, normalised, standardised)
        Y (array or dataframe): outcomes (set of the results)
        model (assigned model from SkLearn): model is assigned prior to application 
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    n3=model.score(X_train, y_train)
    n4=model.score(X_test, y_test)
    return(n3, n4)
    
#%%
#Upload all data for the first model training.
# Only 802 total NA value, they can be eliminated, considering 35095 data points
path = "./clean_data"
csv_files = glob.glob(path + "/*.csv", recursive = True)
all_df = [pd.read_csv(f) for f in csv_files]
pd.set_option('display.max_columns', None)
all_df  = pd.concat(all_df, ignore_index=True)
sns.heatmap(all_df.isnull())
all_df=all_df.dropna()
sns.heatmap(all_df.isnull())
all_df['Capacity']=all_df['Capacity'].astype('str')
all_df.Capacity = all_df.Capacity.apply(lambda x : x.replace(',',''))
all_df.Capacity = all_df.Capacity.apply(lambda x : x.replace('.0',''))
all_df['Capacity']=all_df['Capacity'].astype('int64')

#%%
"""
Overview of the distribution of all features, there are couple of notes:
1. Natural Grass Pitch is the most common, the feature might be not essential
2. Capacity, Round, EOL_home and EOL_away are the only normal distributed (Gaussian)  features.
3. Considering most of the features are cummulation of the previous results, it is clear they are not normaly distributed
4. Capacity is the only feature with extremely large X values, that should be scaled 
"""
hist_df=all_df.drop(['Outcome','Season','Home_Team', 'Away_Team','HT_Cum_Streak','AT_Cum_Streak'], axis=1)
hist_df.hist(alpha=0.5, figsize=(25, 15))
plt.show()

#%%
#Indication of the distribution of the outcomes overall
sns.countplot(x='Outcome', data=all_df, palette='Set1').\
    set(title='Countplot of Output Variable "Outcomes" for the all data')

#%%
#Create set of Features and Outcomes
Y=all_df.Outcome
X=all_df.drop(['Outcome', 'Season','HT_Cum_Streak','AT_Cum_Streak','Home_Team','Away_Team'],axis=1)

# Correlation matrix between columns
#Correlation matrix shows certain level of positive correlation between Home_team data sets and Away_team data sets.
#This is totally explainable and acceptable as both sets have similar group of teams. 
sns.heatmap(X.corr(), xticklabels=X.columns, yticklabels=X.columns)

#%%
#LOGISTIC REGRESSION
LR_model=LogisticRegression()
mscore=train_model(X, Y, LR_model)
temp_df=update_dict('All', 'LogReg', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

# LOGISTIC REGRESSION with scaling Capacity data
mscore=train_model(scaler.fit_transform(X), Y, LR_model)
temp_df=update_dict('All', 'LogReg_Scale', mscore[0],mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#LOGISTIC REGRESSION EXTENDED: standardizing Gaussian distributed features and normalizing Non-Gaussian features
#Prepare the pipeline
Standardize_Var = ['Capacity', 'Round', 'Elo_home', 'Elo_away']
Standardize_transformer = Pipeline(steps=[('standard', StandardScaler())])
Normalize_Var = ['HT_Cum_Cards', 'HT_Cum_Scores', 'HT_Cum_Wins', 'HT_Cum_Draws', 'HT_Cum_Losses',\
    'HT_Longest_Win_Streak', 'HT_Longest_Loss_Streak','AT_Cum_Cards', 'Cum_Scores', 'AT_Cum_Wins',\
    'AT_Cum_Draws', 'AT_Cum_Losses', 'AT_Longest_Win_Streak', 'AT_Longest_Loss_Streak']
Normalize_transformer = Pipeline(steps=[('norm', MinMaxScaler())])
preprocessor = ColumnTransformer(transformers=[('standard', Standardize_transformer, Standardize_Var),('norm', Normalize_transformer, Normalize_Var)])

#LOGISTIC REGRESSION Extended
LR_Ext_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(solver='lbfgs'))])
mscore=train_model(X, Y, LR_Ext_model)
temp_df=update_dict('All', 'LogReg_Ext', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#RANDOM FOREST MODEL
RF_model=RandomForestClassifier()
mscore=train_model(X, Y, RF_model)
temp_df=update_dict('All', 'RandForest', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#RANDOM FOREST MODEL Scaler
mscore=train_model(scaler.fit_transform(X), Y, RF_model)
temp_df=update_dict('All', 'RandForest_Scaler', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#SVM - Support Vector Machines - with Scaler
SVM_model = svm.SVC(kernel='linear')
mscore=train_model(scaler.fit_transform(X), Y, SVM_model)
temp_df=update_dict('All', 'SVM_Scaler', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#SVM Ext: standardizing Gaussian distributed features and normalizing Non-Gaussian features
SVM_Ext_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', svm.SVC(kernel='linear'))])
mscore=train_model(X, Y, SVM_Ext_model)
temp_df=update_dict('All', 'SVM_Ext', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#Building DECISION TREE MODEL for the set of all data features
DTree_model = DecisionTreeClassifier(max_depth=1, random_state=42)
mscore=train_model(scaler.fit_transform(X), Y, DTree_model)
temp_df=update_dict('All', 'DTree_Scaler', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#ADABOOST Model on Decision Tree model
AdaBoost_model = AdaBoostClassifier(base_estimator=DTree_model, n_estimators=500, learning_rate=0.5, random_state=42)
mscore=train_model(scaler.fit_transform(X), Y, AdaBoost_model)
temp_df=update_dict('All', 'AdaBoost_DecTree', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#ADABOOST Model on LogReg_Scaler
AdaBoost_model = AdaBoostClassifier(base_estimator=LR_model, n_estimators=500, learning_rate=0.5, random_state=42)
mscore=train_model(scaler.fit_transform(X), Y, AdaBoost_model)
temp_df=update_dict('All', 'AdaBoost_LogReg', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#GRADIENT BOOST Model
GradBoost_model= GradientBoostingClassifier(learning_rate=0.5, n_estimators=100)
mscore=train_model(scaler.fit_transform(X), Y, GradBoost_model)
temp_df=update_dict('All', 'GradBoost', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

#%%
#FINAL STEP is to apply above models per league data sets: 
#Obtain Data per League
result_files=glob.glob("./clean_data/*", recursive = True)
for f in result_files:
    model_df=pd.read_csv(f)
    league=f.split('/')[-1]
    league=league.split('.')[-2]
    model_df['Capacity']=model_df['Capacity'].astype('str')
    model_df.Capacity = model_df.Capacity.apply(lambda x : x.replace(',',''))
    model_df.Capacity = model_df.Capacity.apply(lambda x : x.replace('.0',''))
    model_df.Capacity = model_df.Capacity.apply(lambda x : x.replace('nan',''))
    model_df[model_df['Capacity']==''] = np.nan
    model_df=model_df.dropna()
    model_df['Capacity']=model_df['Capacity'].astype('Int64')
    
    #LOGISTIC REGRESSION
    LR_model=LogisticRegression()
    mscore=train_model(X, Y, LR_model)
    temp_df=update_dict(league, 'LogReg', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    # LOGISTIC REGRESSION with scaling Capacity data
    scaler = StandardScaler()
    mscore=train_model(scaler.fit_transform(X), Y, LR_model)
    temp_df=update_dict(league, 'LogReg_Scale', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #LOGISTIC REGRESSION EXTENDED: standardizing Gaussian distributed features and normalizing Non-Gaussian features
    #Prepare the pipeline
    Standardize_Var = ['Capacity', 'Round', 'Elo_home', 'Elo_away']
    Standardize_transformer = Pipeline(steps=[('standard', StandardScaler())])
    Normalize_Var = ['HT_Cum_Cards', 'HT_Cum_Scores', 'HT_Cum_Wins', 'HT_Cum_Draws', 'HT_Cum_Losses',\
        'HT_Longest_Win_Streak', 'HT_Longest_Loss_Streak','AT_Cum_Cards', 'Cum_Scores', 'AT_Cum_Wins',\
        'AT_Cum_Draws', 'AT_Cum_Losses', 'AT_Longest_Win_Streak', 'AT_Longest_Loss_Streak']
    Normalize_transformer = Pipeline(steps=[('norm', MinMaxScaler())])
    preprocessor = ColumnTransformer(transformers=[('standard', Standardize_transformer, Standardize_Var),('norm', Normalize_transformer, Normalize_Var)])

    #LOGISTIC REGRESSION Extended
    LR_Ext_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(solver='lbfgs'))])
    mscore=train_model(X, Y, LR_Ext_model)
    temp_df=update_dict(league, 'LogReg_Ext', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #RANDOM FOREST MODEL
    RF_model=RandomForestClassifier()
    mscore=train_model(X, Y, RF_model)
    temp_df=update_dict(league, 'RandForest', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #RANDOM FOREST MODEL Scaler
    mscore=train_model(scaler.fit_transform(X), Y, RF_model)
    temp_df=update_dict(league, 'RandForest_Scaler', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #SVM - Support Vector Machines - with Scaler
    SVM_model = svm.SVC(kernel='linear')
    mscore=train_model(scaler.fit_transform(X), Y, SVM_model)
    temp_df=update_dict(league, 'SVM_Scaler', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #SVM Ext: standardizing Gaussian distributed features and normalizing Non-Gaussian features
    SVM_Ext_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', svm.SVC(kernel='linear'))])
    mscore=train_model(X, Y, SVM_Ext_model)
    temp_df=update_dict(league, 'SVM_Ext', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #Building DECISION TREE MODEL for the set of all data features
    DTree_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    mscore=train_model(scaler.fit_transform(X), Y, DTree_model)
    temp_df=update_dict(league, 'DTree_Scaler', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #ADABOOST Model on Decision Tree
    AdaBoost_model = AdaBoostClassifier(base_estimator=DTree_model, n_estimators=500, learning_rate=0.5, random_state=42)
    mscore=train_model(scaler.fit_transform(X), Y, AdaBoost_model)
    temp_df=update_dict(league, 'AdaBoost_DTree', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #ADABOOST Model on LogReg
    AdaBoost_model = AdaBoostClassifier(base_estimator=LR_model, n_estimators=500, learning_rate=0.5, random_state=42)
    mscore=train_model(scaler.fit_transform(X), Y, AdaBoost_model)
    temp_df=update_dict(league, 'AdaBoost_LogReg', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)

    #GRADIENT BOOST Model
    GradBoost_model= GradientBoostingClassifier(learning_rate=0.5, n_estimators=100)
    mscore=train_model(scaler.fit_transform(X), Y, GradBoost_model)
    temp_df=update_dict(league, 'GradBoost', mscore[0], mscore[1])
    Models_results = pd.concat([Models_results, temp_df], ignore_index=True)
    
    print(f"Done {league}!")

# %%
Models_results.to_csv('model_accuracy.csv', index=False)  

# %%

all_df.columns

# %%
#MLP Model: multilayer perceptron (MLP) is feedforward artificial neural network. 
MLP_model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mscore=train_model(scaler.fit_transform(X), Y, MLP_model)
temp_df=update_dict('All', 'MLP', mscore[0], mscore[1])
Models_results = pd.concat([Models_results, temp_df], ignore_index=True)
# The outcome results Train- 50.01% and Test- 49.50%
#Almost as high as AddBoost

# %%
