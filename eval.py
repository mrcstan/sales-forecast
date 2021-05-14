import numpy as np
import pandas as pd

from mymain import mypredict



train = pd.read_csv('train_ini.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])

# save weighed mean absolute error WMAE
n_folds = 10
next_fold = None
wae = []
# time-series CV
for t in range(1, n_folds+1):
    print('Fitting fold {}...'.format(t))
    # *** THIS IS YOUR PREDICTION FUNCTION ***
    # old code: train, test = mypredict(train, test, next_fold, t)
    train, test_pred = mypredict(train, test, next_fold, t)

    # Load fold file
    # You should add this to your training data in the next call to mypredict()
    fold_file = 'fold_{t}.csv'.format(t=t)
    next_fold = pd.read_csv(fold_file, parse_dates=['Date'])

    # extract predictions matching up to the current fold
    # old code: scoring_df = next_fold.merge(test, on=['Date', 'Store', 'Dept'], how='left')
    scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

    wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

mean = sum(wae)/len(wae)
wae.append(mean)
wae = [ round(elem, 2) for elem in wae ]

names = [str(x) for x in range(10)]
names.append('mean')

df = pd.DataFrame(wae)
df = df.transpose()
df.columns = names
df.to_csv("wae.csv")

print(df)
