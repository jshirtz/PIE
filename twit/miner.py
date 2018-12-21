import twitter
import os
import json

api = twitter.Api(consumer_key='CONSUMER_KEY', 
                  consumer_secret='CONSUMER_SECRET',
                  access_token_key='TOKEN_KEY',
                  access_token_secret='TOKEN_SECRET',
                  sleep_on_rate_limit=True)

def storeTweets(user, store_loc, force = False):
    
    # Check if already downloaded
    tweet_file = store_loc + user + ".tweets"
    if not force and os.path.isfile(tweet_file):
        print("Already downloaded tweets for " + user)
        return None
    
    # Get users stats
    u = api.GetUser(screen_name = user)
    stats = {'tweet_count' : u.statuses_count,
             'following' : u.friends_count,
             'followers' : u.followers_count}
    
    # Start download
    print("[" + user + "] Downloading tweets...")
    tweets = api.GetUserTimeline(screen_name=user, include_rts=False,exclude_replies=True,count=200)
    if len(tweets) == 0:
        print("No tweets found for " + user)
        return None
    last = tweets[len(tweets)-1].id - 1
    
    # Iterate through timeline tweets
    while True:
        more_tweets = api.GetUserTimeline(screen_name=user,include_rts=False,exclude_replies=True,count=200,max_id=last)
        if len(more_tweets) == 0:
            break
        last = more_tweets[len(more_tweets)-1].id - 1
        tweets.extend(more_tweets)
    
    # Write tweets to file
    with open(tweet_file, 'w+') as file:
        stats['tweets'] = "\n".join([tweet.text for tweet in tweets])
        json.dump(stats, file, sort_keys = True, indent = 4)
        
    return tweet_file

def storeFreinds(user, store_loc):
    print("[" + user + "] Collecting friends...")
    next_cursor = -1
    dummy_cursor = 0
    friends = []
    
    # Iterate through paged friends list
    while next_cursor != 0:
        next_cursor, dummy_cursor, more_friends = api.GetFriendsPaged(screen_name=user, cursor=next_cursor, skip_status=True)
        friends.extend(more_friends)
        
    # Write friends to file
    friend_file = store_loc + user + ".friends"
    with open(friend_file, 'w+') as file:
        file.write("\n".join([friend.screen_name for friend in friends]))
        
    return friend_file

def serachHashtag(hashtag, store_loc, n=100):
    list_file = store_loc + hashtag + ".list"
    hashtag = '#' + hashtag
    tweets = [tweet for tweet in api.GetSearch(hashtag, count=100) if hashtag in tweet.text]
    print("[" + hashtag + "]" + "Searching tweets... " + "{0:.2%}".format(len(tweets)/float(n)))
    last = tweets[len(tweets)-1].id - 1
    while len(tweets) < n:
        more_tweets = [tweet for tweet in api.GetSearch(hashtag, count=100, max_id=last) if hashtag in tweet.text]
        if len(more_tweets) == 0:
            break
        last = more_tweets[len(more_tweets)-1].id - 1
        tweets.extend(more_tweets)
        print("[" + hashtag + "]" + "Searching tweets... " + "{0:.2%}".format(len(tweets)/float(n)))
    users = list(set([tweet.user.screen_name for tweet in tweets]))
    with open(list_file, 'w+') as fp:
        fp.write("\n".join(users))
        print("Found " + str(len(users)) + " users of " + str(n) + " mentioning the hashtag")
    return users

def getList(list_name, user, store_loc):
    list_file = store_loc + list_name + '.list' 
    if os.path.isfile(list_file):
        return [x.rstrip() for x in open(list_file, 'r').readlines()]
    print(list_name)
    print(user)
    users = api.GetListMembers(slug = list_name, owner_screen_name = user)
    users = [user.screen_name for user in users]
    with open(list_file, 'w+') as fp:
        fp.write("\n".join(users))
        print("Found " + str(len(users)) + " users in list " + list_name)
    return users

def findLists(hashtag, store_loc, n=100, c_min=40, c_max=200):
    list_file = store_loc + hashtag + ".lists"
    hashtag = '#' + hashtag
    lists = set()
    last = None
    while len(lists) < n:
        tweets = api.GetSearch(hashtag, count=10, max_id=last)
        if len(tweets) == 0:
            break
        last = tweets[len(tweets)-1].id - 1
        users = [tweet.user.screen_name for tweet in tweets if hashtag in tweet.text]
        for user in users:
            user_lists = [(l.slug, l.user.screen_name) for l in api.GetMemberships(screen_name=user) if l.member_count > c_min and l.member_count < c_max] 
            if len(user_lists) > 0:
                lists.update(user_lists)
        print("[" + hashtag + "]" + "Searching lists... " + "{0:.2%}".format(len(lists)/float(n)))
    with open(list_file, 'w+') as fp:
        fp.write("\n".join("%s\t%s" % tup for tup in lists))
        print("Found " + str(len(lists)) + " lists of " + str(n) + "from users mentioning the hashtag")
    return lists
