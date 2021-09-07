import random, re, string

from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk import FreqDist, classify, NaiveBayesClassifier, word_tokenize


def remove_noise(single_tweet_tokens, stop_words=()):
    """
    This method clean the tweets and returns filtered tokens.
    """
    cleaned_tokens = []

    for token, tag in pos_tag(single_tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)

        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


def seperate_all_tokens(cleaned_tokens_list):
    """
    This method seperates all tokens and creates the single token list.
    """
    for single_tweet_tks in cleaned_tokens_list:
        for token in single_tweet_tks:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    """
    This method creates dictionary for filtered tokens.
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def test_sentiment(user_tweet):
    """
    This method shows the sentiment associated with the user tweet.
    """
    user_tweet_tokens = remove_noise(word_tokenize(user_tweet))

    print(f'\n\n {user_tweet} ==> {classifier.classify(dict([token, True] for token in user_tweet_tokens))}')


if __name__ == '__main__':

    pos_filter_tokens, neg_filter_tokens = [], []

    stopwords = stopwords.words('english')

    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    tweet_text = twitter_samples.strings('tweets.20150430-223406.json')

    pos_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    neg_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    # Cleaning tokens
    for pos_tks in pos_tweet_tokens:
        pos_filter_tokens.append(remove_noise(pos_tks, stopwords))

    for neg_tks in neg_tweet_tokens:
        neg_filter_tokens.append(remove_noise(neg_tks, stopwords))

    # Bifurcating each tokens of tweet.
    single_pos_tokens = seperate_all_tokens(pos_filter_tokens)
    single_neg_tokens = seperate_all_tokens(neg_filter_tokens)

    # Calculating frequency of each tokens in the datset.
    freq_single_pos_tks = FreqDist(single_pos_tokens)
    freq_single_neg_tks = FreqDist(single_neg_tokens)

    # Preparing tweet dictionary
    positive_tokens_for_model = get_tweets_for_model(pos_filter_tokens)
    negative_tokens_for_model = get_tweets_for_model(neg_filter_tokens)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    tweet_dataset = positive_dataset + negative_dataset

    random.shuffle(tweet_dataset)  # Shuffling dataset

    train_data = tweet_dataset[:7000]
    test_data = tweet_dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Model Accuracy is:", classify.accuracy(classifier, test_data))

    # print(classifier.show_most_informative_features(10))

    # User tweet input for model test
    user_tweet = str(input('ENTER THE NEW TWEET ==> '))
    test_sentiment(user_tweet)
