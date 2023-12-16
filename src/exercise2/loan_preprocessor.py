import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

yes_no_mapping = {"n": False, "N": False, "y": True, "Y": True}
initial_list_status_mapping = {'w': 'Whole loan', 'f': 'Fractional loan'}
yes_no_features = ["pymnt_plan", "debt_settlement_flag", "hardship_flag"]

scale_features = [
    'loan_amnt',
    'funded_amnt',
    'funded_amnt_inv',
    'int_rate',
    'installment',
    'annual_inc',
    'dti',
    'fico_range_low',
    'fico_range_high',
    'revol_bal',
    'revol_util',
    'total_acc',
    'out_prncp',
    'out_prncp_inv',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_prncp',
    'total_rec_int',
    'total_rec_late_fee',
    'last_pymnt_amnt',
    'tot_cur_bal',
    'total_rev_hi_lim',
    'avg_cur_bal',
    'bc_open_to_buy',
    'bc_util'
]
category_features = ["emp_length", "term","application_type"
                     ,"disbursement_method","loan_status","verification_status"
                     ,"initial_list_status", "home_ownership","purpose","addr_state"]

column_transformer = make_column_transformer(
    (StandardScaler(), scale_features),
    (OneHotEncoder(handle_unknown='ignore'), category_features),
    remainder="passthrough"
)

def preprocess(data):
    # extract X an y
    X = data.drop("grade", axis=1)
    y = data["grade"]

    # clean X
    for column in yes_no_features:
        if column in X.columns:
            X[column] = X[column].map(yes_no_mapping)
        X["initial_list_status"] = X["initial_list_status"].replace(initial_list_status_mapping)
        X.drop("policy_code", axis=1)
    X = pd.DataFrame(column_transformer.fit_transform(X), columns=column_transformer.get_feature_names_out())

    # encode y
    y = LabelEncoder().fit_transform(y)

    return X, y
