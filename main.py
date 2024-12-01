import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import nltk
import matplotlib.pyplot as plt
from datetime import datetime

nltk.download('vader_lexicon')

def main():
    tweets = []
    times = []
    averageCompounds = []
    datapoints = {}
    #Read csv file
    with open('Valorant.csv', newline='', encoding='utf8') as csvfile:

        spamreader = csv.DictReader(csvfile, delimiter=',')

        #Add tweets to python list
        for row in spamreader:
            #print(row)
            tweets.append(row)


    for tweet in tweets:
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(tweet['Text'])
        #for k in sorted(ss):
            #print('{0}: {1}, '.format(k, ss[k]), end='')
        #print()

        tweet_date = datetime.fromisoformat(tweet['Date'])

        if datapoints.get(f'{tweet_date.month}/{tweet_date.day}') == None:
            datapoints[f'{tweet_date.month}/{tweet_date.day}'] = {'Count': 1,'Compound': ss['compound']}
        else:
            previousCount =  datapoints[f'{tweet_date.month}/{tweet_date.day}']['Count']
            previousCompound = datapoints[f'{tweet_date.month}/{tweet_date.day}']['Compound']
            datapoints[f'{tweet_date.month}/{tweet_date.day}'] = {'Count': 1 + previousCount,'Compound': previousCompound + ss['compound']}

        

    for date, data in datapoints.items():
        averageCompounds.append(data['Compound']/data['Count'])
        times.append(date)


    x = []
    for i in range(len(times)):
        x.append(i)

    
    trend = np.polyfit(x, np.array(averageCompounds), 6)
    trendFunction = np.poly1d(trend)

    plt.scatter(times, averageCompounds, marker = "o")
    plt.plot(x, trendFunction(x), color = 'r')
    plt.xlabel('Time')
    plt.xticks(times, rotation='vertical')
    plt.ylabel('Sentiment')
    plt.title("Sentiment over time in 2020")
    plt.show()

if __name__ == '__main__':
    main()