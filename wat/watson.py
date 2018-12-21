from watson_developer_cloud import PersonalityInsightsV3, WatsonApiException
import json
import csv
import os

personality_insights = PersonalityInsightsV3(
            version='2018-09-11j',
            username='USERNAME',
            password='PASSWORD',
            url='https://gateway.watsonplatform.net/personality-insights/api')

csv_file = None
csv_first = True

def storePersonality(user, tweets, out_loc, csv_obj = None):
    print("[" + user + "] Analyzing personality...")

    with open(tweets, 'r') as file:
        tweet_data = json.load(file)

        # Get personality analysis
        try:
            profile = personality_insights.profile(tweet_data['tweets'], 'text/plain')
        except WatsonApiException as ex:
            print("Method failed with status code " + str(ex.code) + ": " + ex.message)
            return None
            
        # Write personality analysis
        personality_file = out_loc + user + ".personality.json"
        with open(personality_file, 'w+') as file:
            json.dump(profile, file, sort_keys = True, indent = 4)
        if csv_obj != None:
            csv_obj.write(user, profile, tweet_data)
            
        return personality_file
    
class personalityCsv:
    
    def __init__(self, name, out_loc):
        self.csv_file_path = out_loc + name + '.csv'
        self.first = not os.path.isfile(self.csv_file_path)
        self.csv_file = csv.writer(open(self.csv_file_path, 'a+'))
            
    def writeHeader(self, personality):
        vals = ["user"]
        vals.extend([need['name'] for need in personality['needs']])
        for trait in personality['personality']:
            n = trait['name']
            vals.append(n)
            vals.extend(n + '.' + child['name'] for child in trait['children'])
        vals.extend([value['name'] for value in personality['values']])
        vals.extend(['Tweets', 'Following', 'Followers'])
        self.csv_file.writerow(vals)
                
    def write(self, user, personality, tweet_data):
        if self.first:
            self.writeHeader(personality)
            self.first = False
        vals = [user]
        vals.extend([need['percentile'] for need in personality['needs']])
        for trait in personality['personality']:
            vals.append(trait['percentile'])
            vals.extend(child['percentile'] for child in trait['children'])
        vals.extend([value['percentile'] for value in personality['values']])
        vals.append(tweet_data['tweet_count'])
        vals.append(tweet_data['following'])
        vals.append(tweet_data['followers'])
        self.csv_file.writerow(vals)