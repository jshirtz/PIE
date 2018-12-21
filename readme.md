# Overview

Command line script to mine tweets, analyze personality, and create clusters. Main script is analyzer.py with supplementing scripts in twit, wat, and viz folders. Users can download tweets for an individual user, a file containing a set of users, a Twitter list, or a hashtag. Data from runs are stored in the /data folder, under a subfolder unless analyzing a single user. Potential data files include:
  username.tweets : all of a users tweets and metrics as a json
  username.personality.json : personality analysis as a json
  username.friends : list of all a Twitter user's friends
  cluster.csv : cluster descriptions and usersnames in those clusters
  listname.csv : table of each user's personality scores
  listname_big5.pdf : report of cluster analysis according to big5
  listname_need.pdf : report of cluster analysis according to need
  listname_val.pdf : report of cluster analysis according to val

# Dependencies

Requires python 3.5 and multiple python libraries. Library requirements are listed in requirements.txt and can be installed with

  pip install -r requirements.txt

Also requires API keys for twitter and watson. To generate keys, you must create developer accounts with each service and place the keys in their respective files. Twitter developer keys go in twit/miner.py and IBM Watson developer keys go in wat/watson.py. For more details on how to generate keys, see the documentation below.

  https://python-twitter.readthedocs.io/en/latest/getting_started.html
  https://github.com/watson-developer-cloud/python-sdk

# Usage

usage: analyzer.py [-h] [--hashtag HASHTAG] [--user USER] [--file FILE]
                   [--list LIST] [--tweets] [--personality] [--friends]
                   [--full] [--csv] [--force] [--report] [--complexReport]

Run Watson personality analysis on tweets

optional arguments:
  -h, --help         show this help message and exit
  --hashtag HASHTAG  hashtag to search for
  --user USER        specific user to run tweet analysis on
  --file FILE        list of users to run tweet analysis on
  --list LIST        twitter list to analyze, must be owned by --user
  --tweets           store collection of user tweets
  --personality      analyze a users personality based on tweets
  --friends          store collection of user friends
  --full             run full analysis i.e. --tweets --peronality --friends
  --csv              store personalities to a csv
  --force            force redownload of tweets
  --report           generate a report
  --complexReport    cross analyze mutltiple lists from a hashtag (not
                     currently useful)

# Example Use Case

To verfiy installation, try running the following command

  python3 analyzer.py --user walkwest --list walk-westies --full --csv --report

The resulting /data folder should be identical to the /example_data folder
