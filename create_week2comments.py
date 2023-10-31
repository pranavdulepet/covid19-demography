import json
import pickle
import csv

csv_file = './data.F.v.csv'

data_dict = {}

# Read the CSV file
with open(csv_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  

    for row in csv_reader:
        week = int(row[0]) 
        comment = row[1] 

        if week not in data_dict:
            data_dict[week] = []  

        data_dict[week].append(comment)

# Save the dictionary as a PKL file
with open('./pickle/week2comments.F.pkl', 'wb') as pkl_file:
    pickle.dump(data_dict, pkl_file)

# Save the dictionary as a JSON file
with open('week2comments.F.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

print("done")


csv_file = './data.M.v.csv'

data_dict = {}

# Read the CSV file
with open(csv_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  

    for row in csv_reader:
        week = int(row[0]) 
        comment = row[1] 

        if week not in data_dict:
            data_dict[week] = []  

        data_dict[week].append(comment)

# Save the dictionary as a PKL file
with open('./pickle/week2comments.M.Test.pkl', 'wb') as pkl_file:
    pickle.dump(data_dict, pkl_file)

# Save the dictionary as a JSON file
with open('week2comments.M.Test.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

print("done")


with open('./pickle/week2comments.M.pkl', 'rb') as pkl_file:
    data_dict = pickle.load(pkl_file)

with open('week2comments.M.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

print("done")









