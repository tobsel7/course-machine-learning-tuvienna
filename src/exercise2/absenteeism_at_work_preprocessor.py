import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

season_mapping = {
    1: "Summer",
    2: "Autumn",
    3: "Winter",
    4: "Spring"
}
weekday_mapping = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}
month_mapping = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}
education_mapping = {
    1: "High scool",
    2: "Graduate",
    3: "Postgraduate",
    4: "Master and doctor"
}
absence_reason_mapping = {
    0: "Unknown",
    1: "Infectious and parasitic",
    2: "Neoplasms",
    3: "Blood and blood-forming organ",
    4: "Endocrine, nutritional and metabolic",
    5: "Mental and behavioural disorders",
    6: "Nervous system",
    7: "Eye and adnexa",
    8: "Ear and mastoid process",
    9: "Circulatory system",
    10: "Respiratory system",
    11: "Digestive system",
    12: "Skin and subcutaneous tissue",
    13: "Musculoskeletal system",
    14: "Genitourinary system",
    15: "Pregnancy, childbirth and puerperium",
    16: "Perinatal period conditions",
    17: "Congenital malformations, deformations",
    18: "Abnormal clinical symptoms",
    19: "Injury, poisoning",
    20: "Morbidity and mortality ",
    21: "Health service encounters",
    22: "Patient follow-up",
    23: "Medical consultation",
    24: "Blood donation",
    25: "Laboratory examination",
    26: "Unjustified absence",
    27: "Physiotherapy",
    28: "Dental consultation"
}

boolean_features = ["Disciplinary failure", "Social drinker", "Social smoker"]
scale_features = ["Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day", "Hit target", "Son", "Pet", "Weight", "Height", "Body mass index", "Absenteeism time in hours"]
category_features = ["Day of the week", "Month of absence", "Seasons", "Education"]

column_transformer = make_column_transformer(
    (StandardScaler(), scale_features),
    (OneHotEncoder(), category_features),
    remainder="passthrough"
)

def preprocess(data):
    # drop missing values
    data = data[data["Month of absence"] != 0]

    # extract X and y
    X = data.drop(["Reason for absence", "Absenteeism time in hours"], axis=1)
    y = data["Reason for absence"]

    # clean X
    X[boolean_features] = X[boolean_features].astype(bool)
    X["Day of the week"] = X["Day of the week"].replace(weekday_mapping).astype("category")
    X["Month of absence"] = X["Month of absence"].replace(month_mapping).astype("category")
    X["Seasons"] = X["Seasons"].replace(season_mapping).astype("category")
    X["Education"] = X["Education"].replace(education_mapping).astype("category")
    X["Work load Average/day"] = X["Work load Average/day"].round(0).astype(int)
    X = pd.DataFrame(column_transformer.fit_transform(data), columns=column_transformer.get_feature_names_out())

    return X, y
