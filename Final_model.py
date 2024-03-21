# %% [markdown]
# # NCAA Bracket Challenge 2024

# %% [markdown]
# ## Team Starford

# %% [markdown]
# Install required libraries

# %%
#pip install -r requirements.txt

# %% [markdown]
# ## Import libraries

# %%
import random
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate,  Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# %% [markdown]
# Set seeds

# %%
seed = 21
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)

# %% [markdown]
# ## Load data

# %%
teamname_df = pd.read_csv('data/MTeams.csv')
teamseeds_df = pd.read_csv('data/MNCAATourneySeeds.csv')
gameresults_df = pd.read_csv('data/MRegularSeasonCompactResults.csv')
teamseeds_2024_df = pd.read_csv('data/2024_BracketSeeds.csv')
teams_2024_df = pd.read_csv('data/Teams_2024.csv')

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# Label Encoding

# %%
def label_encoding(df):
    """
    parameters:
    df: dataframe to be encoded

    returns:
    labels: encoded dataframe
    num: number of unique characters in the dataframe

    """
    label_encoder = LabelEncoder()
    unique_chars = list(set(df.tolist()))
    num =len(unique_chars)
    labels = label_encoder.fit(unique_chars)
    return labels, num

# %%
team_encode, num_teams = label_encoding(teamname_df['TeamID'])
teamname_df['TeamID_encoded'] = team_encode.transform(teamname_df['TeamID'])

# %%
teamname_df.head()

# %%
teams_2024_df['TeamID_encoded']= team_encode.transform(teams_2024_df['TeamID'])
teams_2024_df.head()

# %% [markdown]
# Get game results of previous season matches to train the model

# %%
gameresults_df['WTeamID_encoded'] = team_encode.transform(gameresults_df['WTeamID'])
gameresults_df['LTeamID_encoded'] = team_encode.transform(gameresults_df['LTeamID'])

# %%
gameresults_df.head()

# %%
gameresults_df.shape

# %% [markdown]
# Train the model using game results until 2023 and predict for 2024

# %% [markdown]
# Get match history from 1985 to 2023 of the teams playing in 2024

# %%
result_history_df_w = gameresults_df.merge(teams_2024_df[['TeamID_encoded']], left_on='WTeamID_encoded', right_on='TeamID_encoded', how='inner').drop(columns=['TeamID_encoded'])

# Merge for 'LTeamID_encoded'
result_history_df_l = gameresults_df.merge(teams_2024_df[['TeamID_encoded']], left_on='LTeamID_encoded', right_on='TeamID_encoded', how='inner').drop(columns=['TeamID_encoded'])

# Concatenate the two merged dataframes
result_history_df = pd.concat([result_history_df_w, result_history_df_l])

result_history_df.head()

# %%
result_history_df.shape

# %%
def data_filter(row):

    """
    parameters:
    row: row of dataframe

    returns:
    new_row: new dataframe with filtered data
    """
    if np.random.uniform() < 0.5:
        new_row = {
            'team_a': row['WTeamID'],
            'team_a_encoded': row['WTeamID_encoded'],
            'team_a_score': row['WScore'],
            'team_b': row['LTeamID'],
            'team_b_encoded': row['LTeamID_encoded'],
            'team_b_score': row['LScore']

        }
    else:
        new_row = {
            'team_b': row['WTeamID'],
            'team_b_encoded': row['WTeamID_encoded'],
            'team_b_score': row['WScore'],
            'team_a': row['LTeamID'],
            'team_a_encoded': row['LTeamID_encoded'],
            'team_a_score': row['LScore']

        }
    return new_row

# %%
result_history_df = result_history_df.apply(data_filter, axis=1).tolist()
result_history_df = pd.DataFrame(result_history_df)
result_history_df.head()

# %%
def calculate_score_diff(row):
    """
    parameters:
    row: row of dataframe

    returns:
    new_row: new dataframe with filtered data
    """
    
    return row['team_a_score'] - row['team_b_score']

# %%
result_history_df["score_diff"] = result_history_df.apply(calculate_score_diff, axis=1)
result_history_df.head()

# # %% [markdown]
# # Split data into train and test

# # %%
# features = ['team_a_encoded', 'team_b_encoded']
# target = 'score_diff'

# X = result_history_df[features]
# y = result_history_df[target]


# # %%
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(len(X_train), len(X_test), len(y_train), len(y_test))

# # %% [markdown]
# # ## Build model

# # %%

# def get_model(num_teams, embedding_dim=32):
#     """
#     parameters:
#     num_teams: number of teams in dataset
#     embedding_dim: dimension of embedding layer

#     returns:
#     model: compiled tensorflow model
    
#     """
#     team_a = Input(shape=(1,), name='team_a')
#     team_b = Input(shape=(1,), name='team_b')

#     team_a_embedding = Embedding(num_teams, embedding_dim, name='team_a_embedding')(team_a)
#     team_b_embedding = Embedding(num_teams, embedding_dim, name='team_b_embedding')(team_b)
    
#     team_a_flat = Flatten()(team_a_embedding)
#     team_b_flat = Flatten()(team_b_embedding)
#     merged = Concatenate()([team_a_flat, team_b_flat])
    
#     dense_1 = Dense(1024, kernel_regularizer=l2(0.001))(merged)  
#     dense_1 = BatchNormalization()(dense_1)
#     dense_1 = LeakyReLU()(dense_1)
#     dense_1 = Dropout(0.6)(dense_1)  
    
#     dense_2 = Dense(512, kernel_regularizer=l2(0.001))(dense_1)  
#     dense_2 = BatchNormalization()(dense_2)
#     dense_2 = LeakyReLU()(dense_2)
#     dense_2 = Dropout(0.6)(dense_2)  

#     dense_3 = Dense(256, kernel_regularizer=l2(0.001))(dense_2)  
#     dense_3 = BatchNormalization()(dense_3)
#     dense_3 = LeakyReLU()(dense_3)
#     dense_3 = Dropout(0.6)(dense_3)  

#     output_layer = Dense(1, activation='linear', name='output')(dense_3)

#     model = Model(inputs=[team_a, team_b], outputs=output_layer)

#     optimizer = Adam(learning_rate=0.0001)  
#     loss_fn = MeanSquaredError()
#     metrics = [MeanAbsoluteError()]
    
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#     model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    
#     return model, reduce_lr, early_stop



# # %% [markdown]
# # Fit the model

# # %%
# Brackets_model, reduce_lr, early_stop = get_model(num_teams)   
# Brackets_model.summary()

# # %%
# Brackets_model.fit(
#     [X_train['team_a_encoded'], X_train['team_b_encoded']],
#     y_train,
#     epochs=100,
#     batch_size=64,
#     validation_split=0.2,
#     callbacks=[reduce_lr, early_stop]
# )

# # %% [markdown]
# # Model Prediction

# # %%
# prediction = Brackets_model.predict([X_test['team_a_encoded'], X_test['team_b_encoded']])

# # %%
# #Calculate the model accuracy

# preds = prediction.reshape(-1).tolist()
# total_preds = len(prediction)
# correct_preds = 0

# y_test_list = y_test.tolist() if isinstance(y_test, np.ndarray) else y_test

# for y_pred, y_true in zip(preds, y_test_list):
#     if y_pred > 0 and y_true > 0:
#         correct_preds += 1
#     elif y_pred < 0 and y_true < 0:
#         correct_preds += 1

# accuracy = correct_preds / total_preds * 100

# print(f"Model Accuracy: {accuracy:.2f}%")

# # %% [markdown]
# # Save model

# # %%
# Brackets_model.save('Brackets_model.keras')

# %% [markdown]
# ### 2024 Brackets results

# %% [markdown]
# Now we have trained the model, we can use the trained model to predict the results of the 2024 tournament brackets

# %%
# load the model

Brackets_model = load_model('Brackets_model.keras')

# %%
teamseeds_2024_df["TeamID_encoded"] = team_encode.transform(teamseeds_2024_df["TeamID"])


# %%
teamseeds_2024_df

# %%
# create the match pairs
def pair_teams(df,col):
    """
    parameters:
    df: dataframe to be encoded
    col: column to be used for encoding
    
    returns:
    match_df: encoded dataframe
    
    """
    num = len(df)
    match_pairs = [(df[col].iloc[i], df[col].iloc[num - 1 - i]) for i in range(num // 2)]
    match_df = pd.DataFrame(match_pairs, columns=['TeamA', 'TeamB'])
    return match_df

def team_pair_names(df):
    """
    parameters:
    df: dataframe to get team names
    
    returns:
    df: dataframe with team names
    
    """
    df['TeamA_name'] = df.merge(teamseeds_2024_df[['TeamID_encoded', 'TeamName']], left_on='TeamA', right_on='TeamID_encoded', how='left')['TeamName']
    df['TeamB_name'] = df.merge(teamseeds_2024_df[['TeamID_encoded', 'TeamName']], left_on='TeamB', right_on='TeamID_encoded', how='left')['TeamName']
    return df

# %%
# Function to determine the winning team based on predictions
def determine_winner(row):
    """
    parameters:
    row: row of dataframe
    
    returns:
    new_row: new dataframe with filtered data

    """
    if row['predictions'] > 0:
        return row['TeamA']
    else:
        return row['TeamB']

def preds(model,df,col1,col2):
    """
    parameters:
    model: model to be used
    df: dataframe to be predicted
    col1: column 1 to be used
    col2: column 2 to be used
    
    returns:
    df: predicted dataframe

    """
    prediction = model.predict([df[col1], df[col2]])
    prediction = prediction.reshape(-1).tolist()
    df['predictions'] = prediction

    df['Team_won'] = df.apply(lambda row: int(determine_winner(row)), axis=1)
    return df

def get_winning_teamname(df):
    """
    parameters:
    df: dataframe to get team names
    
    returns:
    df: dataframe with team names
    
    """
    df = df.merge(teamseeds_2024_df[['TeamID_encoded', 'TeamName']], left_on='Team_won', right_on='TeamID_encoded', how='left')
    df = df.drop(columns=['TeamID_encoded'])
    df = df.rename(columns={'TeamName': 'TeamName_won'})
    return df

# %%
W_team_seeds_df = teamseeds_2024_df[teamseeds_df['Seed'].str[0] == 'W']
X_team_seeds_df = teamseeds_2024_df[teamseeds_df['Seed'].str[0] == 'X']
Y_team_seeds_df = teamseeds_2024_df[teamseeds_df['Seed'].str[0] == 'Y']
Z_team_seeds_df = teamseeds_2024_df[teamseeds_df['Seed'].str[0] == 'Z']

# %% [markdown]
# ### Round of 64

# %% [markdown]
# East region brackets

# %%
W01_pairs_df = pair_teams(W_team_seeds_df, 'TeamID_encoded')
W01_pairs_df = team_pair_names(W01_pairs_df)
W01_pairs_df

# %%
W_1st_round_df = preds(Brackets_model,W01_pairs_df,'TeamA','TeamB')
W_1st_round_df=get_winning_teamname(W_1st_round_df)
W_1st_round_df

# %% [markdown]
# West region brackets

# %%
X01_pairs_df = pair_teams(X_team_seeds_df,'TeamID_encoded')
X01_pairs_df = team_pair_names(X01_pairs_df)
X01_pairs_df

# %%
X_1st_round_df = preds(Brackets_model,X01_pairs_df,'TeamA','TeamB')
X_1st_round_df = get_winning_teamname(X_1st_round_df)
X_1st_round_df

# %% [markdown]
# South region brackets

# %%
Y01_pairs_df = pair_teams(Y_team_seeds_df, 'TeamID_encoded')
Y01_pairs_df = team_pair_names(Y01_pairs_df)
Y01_pairs_df

# %%
Y_1st_round_df = preds(Brackets_model,Y01_pairs_df,'TeamA','TeamB')
Y_1st_round_df = get_winning_teamname(Y_1st_round_df)
Y_1st_round_df

# %% [markdown]
# Midwest region brackets

# %%
Z01_pairs_df = pair_teams(Z_team_seeds_df, 'TeamID_encoded')
Z01_pairs_df = team_pair_names(Z01_pairs_df)
Z01_pairs_df

# %%
Z_1st_round_df = preds(Brackets_model,Z01_pairs_df,'TeamA','TeamB')
Z_1st_round_df = get_winning_teamname(Z_1st_round_df)
Z_1st_round_df

# %% [markdown]
# ### Round of 32

# %% [markdown]
# East region brackets

# %%
W02_pairs_df = pair_teams(W_1st_round_df,'Team_won')
W02_pairs_df = team_pair_names(W02_pairs_df)
W02_pairs_df

# %%
W_2nd_round_df = preds(Brackets_model,W02_pairs_df,'TeamA','TeamB')
W_2nd_round_df = get_winning_teamname(W_2nd_round_df)
W_2nd_round_df

# %% [markdown]
# West region brackets

# %%
X02_pairs_df = pair_teams(X_1st_round_df,'Team_won')
X02_pairs_df = team_pair_names(X02_pairs_df)
X02_pairs_df

# %%
X_2nd_round_df = preds(Brackets_model,X02_pairs_df,'TeamA','TeamB')
X_2nd_round_df = get_winning_teamname(X_2nd_round_df)
X_2nd_round_df

# %% [markdown]
# South region brackets

# %%
Y02_pairs_df = pair_teams(Y_1st_round_df,'Team_won')
Y02_pairs_df = team_pair_names(Y02_pairs_df)
Y02_pairs_df

# %%
Y_2nd_round_df = preds(Brackets_model,Y02_pairs_df,'TeamA','TeamB')
Y_2nd_round_df = get_winning_teamname(Y_2nd_round_df)
Y_2nd_round_df

# %% [markdown]
# Midwest region brackets

# %%
Z02_pairs_df = pair_teams(Z_1st_round_df,'Team_won')
Z02_pairs_df = team_pair_names(Z02_pairs_df)
Z02_pairs_df

# %%
Z_2nd_round_df = preds(Brackets_model,Z02_pairs_df,'TeamA','TeamB')
Z_2nd_round_df = get_winning_teamname(Z_2nd_round_df)
Z_2nd_round_df

# %% [markdown]
# ### Sweet 16

# %% [markdown]
# East region brackets

# %%
W_03_pairs_df = pair_teams(W_2nd_round_df,'Team_won')
W_03_pairs_df = team_pair_names(W_03_pairs_df)
W_03_pairs_df

# %%
W_3rd_round_df = preds(Brackets_model,W_03_pairs_df,'TeamA','TeamB')
W_3rd_round_df = get_winning_teamname(W_3rd_round_df)
W_3rd_round_df

# %% [markdown]
# West region brackets

# %%
X03_pairs_df = pair_teams(X_2nd_round_df,'Team_won')
X03_pairs_df = team_pair_names(X03_pairs_df)
X03_pairs_df

# %%
X_3rd_round_df = preds(Brackets_model,X03_pairs_df,'TeamA','TeamB')
X_3rd_round_df = get_winning_teamname(X_3rd_round_df)
X_3rd_round_df

# %% [markdown]
# South region brackets

# %%
Y03_pairs_df = pair_teams(Y_2nd_round_df,'Team_won')
Y03_pairs_df = team_pair_names(Y03_pairs_df)
Y03_pairs_df

# %%
Y_3rd_round_df = preds(Brackets_model,Y03_pairs_df,'TeamA','TeamB')
Y_3rd_round_df = get_winning_teamname(Y_3rd_round_df)
Y_3rd_round_df

# %% [markdown]
# Midwest region brackets

# %%
Z03_pairs_df = pair_teams(Z_2nd_round_df,'Team_won')
Z03_pairs_df = team_pair_names(Z03_pairs_df)
Z03_pairs_df

# %%
Z_3rd_round_df = preds(Brackets_model,Z03_pairs_df,'TeamA','TeamB')
Z_3rd_round_df = get_winning_teamname(Z_3rd_round_df)
Z_3rd_round_df

# %% [markdown]
# ### Round of 8

# %% [markdown]
# East region brackets

# %%
W_04_pairs_df = pair_teams(W_3rd_round_df,'Team_won')
W_04_pairs_df = team_pair_names(W_04_pairs_df)
W_04_pairs_df

# %%
W_4th_round_df = preds(Brackets_model,W_04_pairs_df,'TeamA','TeamB')
W_4th_round_df = get_winning_teamname(W_4th_round_df)
W_4th_round_df

# %% [markdown]
# West region brackets

# %%
X04_pairs_df = pair_teams(X_3rd_round_df,'Team_won')
X04_pairs_df = team_pair_names(X04_pairs_df)
X04_pairs_df

# %%
X_4th_round_df = preds(Brackets_model,X04_pairs_df,'TeamA','TeamB')
X_4th_round_df = get_winning_teamname(X_4th_round_df)
X_4th_round_df

# %% [markdown]
# South region brackets

# %%
Y04_pairs_df = pair_teams(Y_3rd_round_df,'Team_won')
Y04_pairs_df = team_pair_names(Y04_pairs_df)
Y04_pairs_df

# %%
Y_4th_round_df = preds(Brackets_model,Y04_pairs_df,'TeamA','TeamB')
Y_4th_round_df = get_winning_teamname(Y_4th_round_df)
Y_4th_round_df

# %% [markdown]
# Midwest region brackets

# %%
Z04_pairs_df = pair_teams(Z_3rd_round_df,'Team_won')
Z04_pairs_df = team_pair_names(Z04_pairs_df)
Z04_pairs_df

# %%
Z_4th_round_df = preds(Brackets_model,Z04_pairs_df,'TeamA','TeamB')
Z_4th_round_df = get_winning_teamname(Z_4th_round_df)
Z_4th_round_df

# %% [markdown]
# Concatting teams selected for the playoffs from each region

# %%
columns = ['Teams_selected']
Playoffs_selected_teams_df = pd.DataFrame(columns=columns)

# %%
Playoffs_selected_teams_df.loc[len(Playoffs_selected_teams_df)] = W_4th_round_df['Team_won'].values
Playoffs_selected_teams_df.loc[len(Playoffs_selected_teams_df)] = Z_4th_round_df['Team_won'].values
Playoffs_selected_teams_df.loc[len(Playoffs_selected_teams_df)] = Y_4th_round_df['Team_won'].values
Playoffs_selected_teams_df.loc[len(Playoffs_selected_teams_df)] = X_4th_round_df['Team_won'].values
Playoffs_selected_teams_df

# %% [markdown]
# ### Semi-finals

# %%
semifinals_pairs_df = pair_teams(Playoffs_selected_teams_df, 'Teams_selected')
semifinals_pairs_df = team_pair_names(semifinals_pairs_df)
semifinals_pairs_df

# %%
semifinals_results_df = preds(Brackets_model,semifinals_pairs_df,'TeamA','TeamB')
semifinals_results_df = get_winning_teamname(semifinals_results_df)
semifinals_results_df

# %% [markdown]
# ### Finals

# %%
Finals_pairs_df = pair_teams(semifinals_results_df,'Team_won')
Finals_pairs_df = team_pair_names(Finals_pairs_df)
Finals_pairs_df

# %%
Finals_results_df = preds(Brackets_model,Finals_pairs_df,'TeamA','TeamB')
Finals_results_df = get_winning_teamname(Finals_results_df)
Finals_results_df

# %%
print("Team won the tournament: " + Finals_results_df['TeamName_won'].values[0])


