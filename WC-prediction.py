#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import pickle
from scipy.stats import poisson


# In[31]:


df_WC_2022 = pd.read_csv('WC_Datasets/wc_matches_2022_crawl.csv')


# In[32]:


dict_table = pickle.load(open('WC_Datasets/dict_table','rb'))
df_historical_matches = pd.read_csv('WC_Datasets/clean_fifa_worldcup_matches.csv')
df_fixture = pd.read_csv('WC_Datasets/clean_fifa_worldcup_fixture.csv')


# In[33]:


# df_historical_data = df_historical_matches.append(df_WC_2022, ignore_index=True)

df_historical_data = pd.concat([df_historical_matches,df_WC_2022])


# ## Calculate Team Strength

# In[34]:


df_home = df_historical_data[['HomeTeam', 'HomeGoals', 'AwayGoals']]
df_away = df_historical_data[['AwayTeam', 'HomeGoals', 'AwayGoals']]

df_home = df_home.rename(columns={'HomeTeam':'Team', 'HomeGoals': 'GoalsScored', 'AwayGoals': 'GoalsConceded'})
df_away = df_away.rename(columns={'AwayTeam':'Team', 'HomeGoals': 'GoalsConceded', 'AwayGoals': 'GoalsScored'})

df_team_strength = pd.concat([df_home, df_away], ignore_index=True).groupby(['Team']).mean()
df_team_strength


# ## Function Predict_points

# In[35]:


def predict_points(home, away):
    if home in df_team_strength.index and away in df_team_strength.index:
        # goals_scored * goals_conceded
        lamb_home = df_team_strength.at[home,'GoalsScored'] * df_team_strength.at[away,'GoalsConceded']
        lamb_away = df_team_strength.at[away,'GoalsScored'] * df_team_strength.at[home,'GoalsConceded']
        prob_home, prob_away, prob_draw = 0, 0, 0
        for x in range(0,11): #number of goals home team
            for y in range(0, 11): #number of goals away team
                p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                if x == y:
                    prob_draw += p
                elif x > y:
                    prob_home += p
                else:
                    prob_away += p
        
        points_home = 3 * prob_home + prob_draw
        points_away = 3 * prob_away + prob_draw
        return (points_home, points_away)
    else:
        return (0, 0)


# ## Testing Function

# In[36]:


print(predict_points('England', 'Wales'))
print(predict_points('Argentina', 'Saudi Arabia'))
print(predict_points('Qatar (H)', 'Ecuador')) # Qatar vs Team X -> 0 points to both


# In[37]:


df_fixture_group_48 = df_fixture[:48].copy()
df_fixture_knockout = df_fixture[48:56].copy()
df_fixture_quarter = df_fixture[56:60].copy()
df_fixture_semi = df_fixture[60:62].copy()
df_fixture_final = df_fixture[62:].copy()


# In[38]:


for group in dict_table:
    teams_in_group = dict_table[group]['Team'].values
    df_fixture_group_6 = df_fixture_group_48[df_fixture_group_48['home'].isin(teams_in_group)]
    for index, row in df_fixture_group_6.iterrows():
        home, away = row['home'], row['away']
        points_home, points_away = predict_points(home, away)
        dict_table[group].loc[dict_table[group]['Team'] == home, 'Pts'] += points_home
        dict_table[group].loc[dict_table[group]['Team'] == away, 'Pts'] += points_away

    dict_table[group] = dict_table[group].sort_values('Pts', ascending=False).reset_index()
    dict_table[group] = dict_table[group][['Team', 'Pts']]
    dict_table[group] = dict_table[group].round(0)


# In[39]:


dict_table['Group A']


# In[40]:


dict_table['Group B']


# In[41]:


dict_table['Group C']


# In[42]:


dict_table['Group D']


# In[43]:


dict_table['Group E']


# In[44]:


dict_table['Group F']


# In[45]:


dict_table['Group G']


# In[46]:


dict_table['Group H']


# ## Knock Out

# In[47]:


df_fixture_knockout


# In[48]:


for group in dict_table:
    group_winner = dict_table[group].loc[0, 'Team']
    runners_up = dict_table[group].loc[1, 'Team']
    df_fixture_knockout.replace({f'Winners {group}':group_winner,
                                 f'Runners-up {group}':runners_up}, inplace=True)

df_fixture_knockout['winner'] = '?'
df_fixture_knockout


# In[49]:


def get_winner(df_fixture_updated):
    for index, row in df_fixture_updated.iterrows():
        home, away = row['home'], row['away']
        points_home, points_away = predict_points(home, away)
        if points_home > points_away:
            winner = home
        else:
            winner = away
        df_fixture_updated.loc[index, 'winner'] = winner
    return df_fixture_updated


# In[50]:


get_winner(df_fixture_knockout)


# ## Quarter Final

# In[51]:


def update_table(df_fixture_round_1, df_fixture_round_2):
    for index, row in df_fixture_round_1.iterrows():
        winner = df_fixture_round_1.loc[index, 'winner']
        match = df_fixture_round_1.loc[index, 'score']
        df_fixture_round_2.replace({f'Winners {match}':winner}, inplace=True)
    df_fixture_round_2['winner'] = '?'
    return df_fixture_round_2


# In[52]:



update_table(df_fixture_knockout, df_fixture_quarter)


# In[53]:


get_winner(df_fixture_quarter)


# ## SemiFinal

# In[54]:


update_table(df_fixture_quarter, df_fixture_semi)


# In[55]:


get_winner(df_fixture_semi)


# ## Final

# In[56]:


update_table(df_fixture_semi, df_fixture_final)


# In[57]:


get_winner(df_fixture_final)


# In[ ]:





# In[ ]:




