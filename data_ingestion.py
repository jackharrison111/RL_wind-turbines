

import pandas as pd


description = pd.read_csv("engie_data/data_description_csv.csv", skiprows=1)
print(description.columns)


data = pd.read_csv("engie_data/wind_data_separated_csv.csv")
data.columns

turbines = ['R80721', 'R80711', 'R80736', 'R80790']
for turb in turbines:

    turbine1 = data.loc[data['Wind_turbine_name'] == turb]
    turbine1.to_csv(f"engie_data/turbine{turb}.csv")