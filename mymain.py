import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

def mypredict(train, test, next_fold, t):
    print('*************** Fold %d ***************'%t)
    enc = OneHotEncoder(handle_unknown='ignore')
    wk_enc = enc.fit((np.arange(52) + 1).reshape(-1, 1))
    wk_cols = wk_enc.get_feature_names(['Week'])
    Xcols = list(wk_cols)
    Xcols.insert(0, 'Year')
    if next_fold is None:
        train['Year'] = pd.DatetimeIndex(train['Date']).year
        train['Week'] = pd.DatetimeIndex(train['Date']).week
        train_enc_wk = pd.DataFrame(wk_enc.transform(train[['Week']]).toarray())
        train_enc_wk.columns = wk_cols
        train = pd.concat([train, train_enc_wk], axis=1)
    else:
        next_fold['Year'] = pd.DatetimeIndex(next_fold['Date']).year
        next_fold['Week'] = pd.DatetimeIndex(next_fold['Date']).week
        next_fold_enc_wk = pd.DataFrame(wk_enc.transform(next_fold[['Week']]).toarray())
        next_fold_enc_wk.columns = wk_cols
        next_fold = pd.concat([next_fold, next_fold_enc_wk], axis=1)
        train = train.append(next_fold, ignore_index=True)

    test['Year'] = pd.DatetimeIndex(test['Date']).year
    test['Week'] = pd.DatetimeIndex(test['Date']).week
    test['Weekly_Pred'] = np.nan
    test_enc_wk = pd.DataFrame(wk_enc.transform(test[['Week']]).toarray())
    test_enc_wk.columns = wk_cols
    test[wk_cols] = test_enc_wk

    # Get the prediction date range, which is the next two months after
    # the last month of the training date range
    pred_start_date = datetime.fromisoformat('2011-03-01') + DateOffset(months=2 * (t - 1))
    pred_end_date = datetime.fromisoformat('2011-05-01') + DateOffset(months=2 * (t - 1))
    print('Training date range =', train['Date'].min(), ' to ', train['Date'].max())
    #print('Test date range =', test['Date'].min(), ' to ', test['Date'].max())
    print('Prediction date range =', pred_start_date, ' to ', pred_end_date)

    # Extract the subset of the test data frame within
    # the prediction date range
    sel_test_ind = np.logical_and(test['Date']>=pred_start_date, test['Date']<=pred_end_date)
    test_subset = test.loc[sel_test_ind].copy()
    train_grps = train.groupby(['Dept','Store'])
    for key, train_grp in train_grps:
        dept = key[0]
        store = key[1]
        ind_test_subset = np.logical_and(test_subset['Dept'] == dept, test_subset['Store'] == store)
        if not ind_test_subset.any():
            continue

        ## Linear regression with week as categorical variable and year as continuous variable
        model = Ridge(alpha=0.1, normalize=True, solver='svd').fit(train_grp[Xcols], train_grp['Weekly_Sales'])
        y_pred = model.predict(test_subset.loc[ind_test_subset, Xcols])
        y_pred[y_pred <= 0] = 0
        y_pred[np.isnan(y_pred)] = 0
        test_subset.loc[ind_test_subset, 'Weekly_Pred'] = y_pred

    #print('Prediction: ')
    #print(test_subset.head())
    #print(test_subset.tail())
    print('Max predict = ', test_subset['Weekly_Pred'].max(),
          ', min predict = ', test_subset['Weekly_Pred'].min())
    ind_na = test_subset['Weekly_Pred'].isna()
    if ind_na.any():
        print('Number of NA predictions = ', ind_na.sum())
        #print(test_subset.loc[ind_na, :])
    test.loc[sel_test_ind, ['Weekly_Pred']] = test_subset['Weekly_Pred']
    print('*************** End prediction ***************')
    return train, test

