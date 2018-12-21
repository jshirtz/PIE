'''
Created on Oct 2, 2018

@author: Jonathan Schertz
'''

from twit import miner as tw
from wat import watson
from viz import visualize
import argparse
import sys
import os
import pandas as pd

DATA = "./data/"

# Set arguments for command line usage
parser = argparse.ArgumentParser(description='Run Watson personality analysis on tweets')
parser.add_argument('--hashtag', action='store', help='hashtag to search for')
parser.add_argument('--user', action='store', help='specific user to run tweet analysis on')
parser.add_argument('--file', type=argparse.FileType('r'), action='store', help='list of users to run tweet analysis on')
parser.add_argument('--list', action='store', help='twitter list to analyze, must be owned by --user')
parser.add_argument('--tweets', action='store_true', default=False, help='store collection of user tweets')
parser.add_argument('--personality', action='store_true', default=False, help='analyze a users personality based on tweets')
parser.add_argument('--friends', action='store_true', default=False, help='store collection of user friends')
parser.add_argument('--full', action='store_true', default=False, help='run full analysis i.e. --tweets --peronality --friends')
parser.add_argument('--csv', action='store_true', default=False, help='store personalities to a csv')
parser.add_argument('--force', action='store_true', default=False, help='force redownload of tweets')
parser.add_argument('--report', action='store_true', default=False, help='generate a report')
parser.add_argument('--complexReport', action='store_true', default=False, help='cross analyze mutltiple lists from a hashtag (not currently useful)')

args = parser.parse_args()
if args.full:
    args.tweets = True
    args.personality = True
    args.friends = True

if not args.tweets and not args.personality and not args.friends and not args.complexReport:
    sys.exit("No operation set")

if args.list and not args.user:
    sys.exit("If usuing a twitter list, --user must be specified as the owner of that list")

if args.report and not args.csv:
    sys.exit("Report requires that --csv is set")

def complexReport():

    if args.hashtag != None:
        out_dir = DATA + args.hashtag + '/'
        lists = tw.findLists(args.hashtag, out_dir)

    elif args.file != None:
        name = os.path.basename(args.file.name)
        out_dir = DATA + name[:name.find('.')] + '/'
        lists = [x.rstrip().split('\t') for x in args.file.readlines()]
    
    else:
        sys.exit('Need list or hashtag for a complex report')

    os.makedirs(out_dir, exist_ok = True)
    for l in lists:
        name = l[0]

        l_out_dir = out_dir + name + '/'
        os.makedirs(l_out_dir, exist_ok = True)
        #users = tw.getList(name, l[1], l_out_dir)
        csv = watson.personalityCsv(name, l_out_dir)

        for user in users:
            tweets = tw.storeTweets(user, l_out_dir, args.force)
            if tweets == None:
                continue
            watson.storePersonality(user, tweets, l_out_dir, csv)
                
            print("Done")
            
        v = visualize.Viz(csv.csv_file_path)
        clusters = v.report()

    frame = pd.DataFrame()
    for cluster in clusters:
        val = cluster[0] + "(" + str(cluster[1]) + ")"
        row = cluster[2]
        col = cluster[3]
        if row in frame.index and col in frame.columns and not pd.isnull(frame.at[row,col]):
            frame.loc[row,col] = frame.at[row,col] + '\n' + val
        else:
            frame.loc[row,col] = val
    frame.to_csv(out_dir + 'big5_clusters.csv') 

if __name__ == '__main__':
    users = []
    out_dir = DATA
    name = "personality"
    
    if args.complexReport:
        complexReport()
    
    # Setup list of users
    if args.list != None:
        name = args.list
        if args.user == None:
            sys.exit("List owner must be specified as --user")
        out_dir = out_dir + args.list + '/'
        os.makedirs(out_dir, exist_ok = True)
        if args.tweets or args.friends:
            users = tw.getList(args.list, args.user, out_dir)
        else:
            users = [file[:file.find('.tweets')] for file in os.listdir(out_dir) if file.endswith('tweets')]
        
    elif args.hashtag != None:
        name = args.hashtag
        out_dir = out_dir + args.hashtag + '/'
        os.makedirs(out_dir, exist_ok = True)
        if args.tweets or args.friends:
            users = tw.serachHashtag(args.hashtag, out_dir, 5000)
        else:
            users = [file[:file.find('.tweets')] for file in os.listdir(out_dir) if file.endswith('tweets')]

    elif args.user != None:
        name = args.user
        users.append(args.user)

    elif args.file != None:
        out_dir = os.path.dirname(args.file.name) + '/'
        name = os.path.basename(args.file.name)
        name = name[:name.find('.')]
        users.extend([x.rstrip() for x in args.file.readlines()])

    else:
        sys.exit("User or user list required")
        
    # Start csv
    if args.csv:
        csv = watson.personalityCsv(name, out_dir)
    else:
        csv = None
        
    # Run functions on users
    for user in users:

        tweets = None
        if args.tweets:
            tweets = tw.storeTweets(user, out_dir, args.force)
            if tweets == None:
                continue

        if args.personality:
            if tweets == None:
                tweets = out_dir + user + ".tweets"
                if not os.path.isfile(tweets):
                    sys.exit("Cannot analyze personality until tweets have been collected") 
            personality = watson.storePersonality(user, tweets, out_dir, csv)
            
        if args.friends:
            friends = tw.storeFreinds(user, out_dir)
            
        print("Done")
        
    # Generate report
    if args.csv and args.report:
        v = visualize.Viz(csv.csv_file_path)
        v.report()
