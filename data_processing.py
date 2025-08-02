import pandas as pd
import re
from tqdm import tqdm
import copy
from sklearn.model_selection import train_test_split

# Importing contractions
with open("contractions.txt", "r") as inp_cont:
    contractions_list = inp_cont.read()
contractions_list = [re.sub('["]', '', x).split(":") for x in re.sub(r"\s+", " ", re.sub(r"(.*{)|(}.*)", '', contractions_list)).split(',')]
contractions_dict = dict((k.lower().strip(), re.sub('/.*', '', v).lower().strip()) for k, v in contractions_list)


# Function to remove special characters
def remove_sc(_line, lang="en"):
    if lang == "hi":
        _line = re.sub(r'[+\-*/#@%>=;~{}×–`’"()_]', "", _line)
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', _line)
    elif lang == "en":
        _line = re.sub(r'[+\-*/#@%>=;~{}×–`’"()_|:]', "", _line)
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', _line)
    return _line


# Function to clean text
def clean_text(_text, lang="en"):
    if lang == "en":
        _text = remove_sc(_line=_text, lang=lang)
        for cn in contractions_dict:
            _text = re.sub(cn, contractions_dict[cn], _text)
    elif lang == "hi":
        _text = remove_sc(_line=_text, lang=lang)
    return _text


# Loading hindi text
with open("C:\\Translation Model\\parallel-n\\IITB.en-hi.hi", "r", encoding='utf-8') as hindi_inp:
    _text = hindi_inp.read()
hindi_text = _text.split('\n')


# Loading english text
with open("C:\\Translation Model\\parallel-n\\IITB.en-hi.en", "r", encoding='utf-8') as eng_inp:
    _text = eng_inp.read()
eng_text = _text.split('\n')


# Removing Hindi sentences containing English letters
ids_to_remove = {}
for _id, _t in tqdm(enumerate(hindi_text)):
    if len(re.findall(r'[a-zA-Z]', _t)) > 0:
        ids_to_remove[_id] = _t


# Filtering out unwanted sentences
ids_to_keep = [i for i in range(len(hindi_text)) if i not in ids_to_remove.keys()]
filtered_eng_text = []
filtered_hindi_text = []
for _id in tqdm(ids_to_keep):
    filtered_eng_text.append(eng_text[_id].lower())
    filtered_hindi_text.append(hindi_text[_id])


# Cleaning English sentences
clean_eng_text = []
for sent in tqdm(filtered_eng_text):
    clean_eng_text.append(clean_text(_text=copy.deepcopy(sent), lang="en"))


# Cleaning Hindi sentences
clean_hindi_text = []
for sent in tqdm(filtered_hindi_text):
    clean_hindi_text.append(clean_text(_text=copy.deepcopy(sent), lang="hi"))


# Creating filtered DataFrame
clean_data = pd.DataFrame({"eng_text": clean_eng_text, "hindi_text": clean_hindi_text})


# Filtering data based on sentence length
clean_data["eng_len"] = clean_data.eng_text.str.count(" ")
clean_data["hindi_len"] = clean_data.hindi_text.str.count(" ")
small_len_data = clean_data.query('eng_len < 50 & hindi_len < 50')


# Train-validation split
# Full set
train_set, val_set = train_test_split(small_len_data.loc[:, ["eng_text", "hindi_text"]], test_size=0.1)
train_set.to_csv("train.csv", index=False)
val_set.to_csv("val.csv", index=False)

# Small set
small_data = small_len_data.loc[:, ["eng_text", "hindi_text"]].sample(n=150000)
train_set_sm, val_set_sm = train_test_split(small_data, test_size=0.3)
train_set_sm.to_csv("train_sm.csv", index=False)
val_set_sm.to_csv("val_sm.csv", index=False)
