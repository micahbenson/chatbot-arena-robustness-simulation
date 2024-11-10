import pandas as pd
import requests

#Mostly code from https://github.com/lm-sys/FastChat and https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=tyl5Vil7HRzd 
# we use the latest data
url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json"
response = requests.get(url)

with open('local_file_name.json', 'wb') as file:
    file.write(response.content)

# load the JSON data from the local file
with open('local_file_name.json', 'r') as file:
    battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

# we use anony battles only for leaderboard
battles = battles[battles["anony"] == True]

# we de-duplicate top 0.1% redudant prompts
# see https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quality-through-de-duplication
print("Before dedup: ", len(battles))
battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
print("After dedup: ", len(battles))

battles.to_csv('data/battles.csv', index=False)