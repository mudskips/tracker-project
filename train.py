from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

in_vd = pd.read_csv("infinite void.csv", header = None)
natural = pd.read_csv("natural.csv", header = None)

ovr_data = pd.concat([in_vd, natural])
nums = ovr_data.iloc[:,1:]
name = ovr_data.iloc[:,0]

nums_train, nums_test, name_train, name_test = train_test_split(nums, name, test_size=0.3, random_state=13)

my_model = RandomForestClassifier(n_estimators=150, random_state=13)
my_model.fit(nums_train, name_train)
print(f"model training complete :)")
model_predictions= my_model.predict(nums_test)
print(accuracy_score(name_test, model_predictions))


