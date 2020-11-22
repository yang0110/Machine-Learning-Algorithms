import xgboost as xgb
N=10000
# mod_train = train[used_columns]
x = mod_train.iloc[:N, :-1].values
y = mod_train.iloc[:N, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
x_test = sc.transform(x_test)

dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test, label=y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 50, 'eta':0.3, 'colsample_bytree': 0.3, 'max_depth': 4, 'subsample': 0.8, 'lambda': 1, 'booster': 'gbtree', 'eval_metric': 'rmse', 'objective':'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 100, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=10)
# print('Modeling RMSLE %.5f' % model.best_score)
xgb_train_error = model.best_score
xgb_y = model.predict(dtest)
xgb_test_error = np.sqrt(mean_squared_error(xgb_y, y_test))
print('train error %s, test_error %s'%(xgb_train_error, xgb_test_error))
train_error_list.append(xgb_train_error)
test_error_list.append(xgb_test_error)
