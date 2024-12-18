import pandas as pd
import numpy as np
import math

#Some code here is adapted from https://github.com/lm-sys/FastChat and https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=tyl5Vil7HRzd 

#Get the counts of all the battles recorded in the data through a pivot table
def get_counts(df): 
    ptbl = pd.pivot_table(df, index="model_a", columns="model_b", aggfunc="size",
                            fill_value=0)
    counts = ptbl + ptbl.T
    return counts

#Get the outcome of battles (model_a wins, model_b wins, tie) in pivot table of same shape as counts
def get_results(df):
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    ptbl_tie = pd.pivot_table(
        df[df["winner"].isin(["tie", "tie (bothbad)"])],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    return ptbl_a_win, ptbl_b_win, ptbl_tie

#run multinomial trial given row with columns for count and probabilities of win, tie, loss
def perform_multinomial(row):
    trials = row['count']
    probs = [row['win_pr'], row['tie_pr'], row['lose_pr']]
    return np.random.multinomial(trials, probs)

#Given a pivot table, reshape the data to allow us to run multinomial trials
def unpack(df, new_col_name): 
    # Reshape the data using stack, reset the index, and filter
    reshaped_df = df.stack().reset_index()
    reshaped_df.columns = ['model_a', 'model_b', new_col_name]

    # Remove duplicate battles, keeping only one direction for each pair
    reshaped_df['pair'] = reshaped_df.apply(lambda row: frozenset([row['model_a'], row['model_b']]), axis=1)
    unique_battles_df = reshaped_df.drop_duplicates(subset='pair').drop(columns='pair').reset_index(drop=True)
    return unique_battles_df

#Given data generated by a multinomial trial, reshape it back into a pivot table in order to run MLE algorithm 
def rebuild(df, lose): 
    # Create a new DataFrame with model pairs filled in both directions
    # Add reverse rows for model_2 beating model_1 by setting `1 - pr`
    col_1 = df.columns[0]
    col_2 = df.columns[1]
    col_3 = df.columns[2]
    

    reverse_df = pd.DataFrame({
        col_1 : df[col_2],
        col_2 : df[col_1],
        col_3 : lose
    })

    # Combine the original and reversed DataFrames
    full_df = pd.concat([df, reverse_df])
    full_df = full_df.drop_duplicates(subset=['model_a', 'model_b'], keep='first')

    full_df = full_df.reset_index(drop=True)
    # Pivot to recreate the matrix
    reconstructed_df = full_df.pivot(index=col_1, columns=col_2, values=col_3)

    # Fill any NaN values with 0 to match the original matrix
    reconstructed_df = reconstructed_df.fillna(0)
    reconstructed_df.index.name = None
    reconstructed_df.columns.name = None
    return reconstructed_df

#Algorithm provided by lm-sys team, with a few tweaks to output at the end 
def compute_mle_elo(
    ptbl_a_win, ptbl_tie, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    from sklearn.linear_model import LogisticRegression
    
    ptbl_win = ptbl_a_win * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    ranking = pd.Series(elo_scores, index=models.index).sort_values(ascending=False)
    ranking_df = pd.DataFrame(ranking, columns=['bt_score'])
    ranking_df["rank"] = ranking_df['bt_score'].rank(method='first', ascending=False).astype(int)
    return ranking_df

#read in the data
battles = pd.read_csv("data/battles.csv")

#process data with pivot tables
a_win, b_win, tie = get_results(battles)
wins = a_win  + b_win.T #this is the number of wins 
tie #this is the number of ties 
counts = get_counts(battles) #this is the total number of battles

#get pivot tables with probabilities of win, tie, loss by dividing counts of each outcome by total counts of battles
win_pr = wins.div(counts, level=1).fillna(0)
tie_pr = tie.div(counts, level=1).fillna(0)
lose_pr = wins.T.div(counts, level=1).fillna(0)

#Reshape probabilities pivot tables in order to run multinomial trials
win_pr = unpack(win_pr, 'pr')
lose_pr = unpack(lose_pr, 'pr')
tie_pr = unpack(tie_pr, 'pr')
counts = unpack(counts, 'count')

#create dataframe to store all the info to run multinomial trials
multi_pr = counts
multi_pr["win_pr"] = win_pr['pr']
multi_pr["tie_pr"] = tie_pr['pr']
multi_pr["lose_pr"] = lose_pr['pr']

#now run multinomial trials based on this!
# Apply the function to each row and expand results into new columns
multi_pr[['win_sim', 'tie_sim', 'lose_sim']] = multi_pr.apply(perform_multinomial, axis=1, result_type='expand')

win_reshape = multi_pr[['model_a', 'model_b', 'win_sim']]
tie_reshape = multi_pr[['model_a', 'model_b', 'tie_sim']]
lose = multi_pr['lose_sim']

#Reshape data back into pivot table to run mle alg
wins_trial = rebuild(win_reshape, lose)
ties_trial = rebuild(tie_reshape, multi_pr['tie_sim'])

#compute rankings/scores
simulations = compute_mle_elo(wins_trial, ties_trial)
simulations_score = simulations['bt_score']
simulations_rank = simulations['rank']

#now repeat the same code to get to 1000 trials 
for x in range(999): 
    multi_pr[['win_sim', 'tie_sim', 'lose_sim']] = multi_pr.apply(perform_multinomial, axis=1, result_type='expand')

    win_reshape = multi_pr[['model_a', 'model_b', 'win_sim']]
    tie_reshape = multi_pr[['model_a', 'model_b', 'tie_sim']]
    lose = multi_pr['lose_sim']

    wins_trial = rebuild(win_reshape, lose)
    ties_trial = rebuild(tie_reshape, multi_pr['tie_sim'])

    trial_result = compute_mle_elo(wins_trial, ties_trial)
    trial_result.rename(inplace=True, columns={'bt_score':f'bt_score_{x+1}',
                                  'rank' : f'rank_{x+1}'})
    trial_result_score = trial_result[f'bt_score_{x+1}']
    trial_result_rank = trial_result[f'rank_{x+1}']
    
    simulations_score = pd.concat([simulations_score, trial_result_score], axis=1)
    simulations_rank = pd.concat([simulations_rank, trial_result_rank], axis=1)
    print(x)

#save data for analyis 
simulations_rank.to_csv('data/replay_sim_rank.csv')
simulations_score.to_csv('data/replay_sim_score.csv')