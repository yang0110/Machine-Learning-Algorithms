import lightgbm as lgb
d_train = lgb.Dataset(x_train, y_train)
lgb_params = {
    'learning_rate': 0.2, # try 0.2
    'max_depth': 4,
    'num_leaves': 10, 
    'objective': 'regression',
    'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 20}       # 1000
n_rounds = 10
model_lgb = lgb.train(lgb_params, 
                      d_train, 
                      # feval=lgb_rmsle_score, 
                      num_boost_round=n_rounds)

light_yt = model_lgb.predict(x_train)
light_y = model_lgb.predict(x_test)
light_train_error = np.sqrt(mean_squared_error(light_yt, y_train))
light_test_error = np.sqrt(mean_squared_error(light_y, y_test))
print('light train error %s, light test error %s'%(light_train_error, light_test_error))
train_error_list.append(light_train_error)
test_error_list.append(light_test_error)

