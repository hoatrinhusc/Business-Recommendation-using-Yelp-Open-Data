'''

Method Description:
I use XGBoost to predict the rating. The features are: average star of user and business, attributes of business,
number of user's friends, longtitude, latitude, review_count and total number of categories a restaurant has.
For new business, I use item-based to predict rating.
For new user or new user and new business I use a default value 3.5 for rating score
I tried to calculate correlation between each feature and the predicted rating, but the correlation is very low.
This problem proves that selecting features are hard.

Error Distribution:
>=0 and <1: 94911
>=1 and <2: 38819
>=2 and <3: 7272
>=3 and <4: 1038
>=4: 4

RMSE: 0.97857

Execution Time on Local Machine: 46 seconds

'''


import json
import sys
import os
from pyspark import SparkContext, SparkConf
import time
import numpy as np
import xgboost as xgb
from itertools import combinations
import ast
import pandas as pd


os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
jaccard_threshold = 0.01
co_rating_threshold = 5


def get_test_data(string):
    row = string.split(",")
    return row[0], row[1]


def get_train_data(string):
    row = string.split(",")
    return row[0], (row[1], float(row[2]))


def calculate_rmse(test_file_path, out_file):
    true_rating_rdd = sc.textFile(test_file_path).filter(lambda string: string[:7] != "user_id")\
        .map(lambda string: string.split(","))\
        .map(lambda x: ((x[0], x[1]), float(x[2])))

    pred_rating_rdd = sc.textFile(out_file).filter(lambda string: string[:7] != "user_id")\
        .map(lambda string: string.split(","))\
        .map(lambda x: ((x[0], x[1]), float(x[2])))

    compare = pred_rating_rdd.join(true_rating_rdd).map(lambda x: x[1])
    total = compare.map(lambda x: (x[0]-x[1])*(x[0]-x[1])).sum()
    rmse = np.sqrt(total/len(true_rating_rdd.collect()))

    return rmse

# ************************************************************************** #

# clean business.json #


def process_attributes(a_dict):

    attr_dict = a_dict.get('attributes', -1)

    if isinstance(attr_dict, dict):

        boolean_features = ['RestaurantsGoodForGroups', 'RestaurantsReservations', 'RestaurantsTakeOut', 'Caters',
                            'OutdoorSeating', 'RestaurantsDelivery','BusinessAcceptsCreditCards', 'GoodForKids', 'HasTV',
                            'RestaurantsTableService', 'AcceptsInsurance',
                            'WheelchairAccessible', 'CoatCheck', 'Corkage', 'BikeParking']

        numeric_features = []

        for feat in boolean_features:
            if feat in attr_dict:
                if attr_dict[feat] == 'True':
                    numeric_features.append(1)
                else:
                    numeric_features.append(0)
            else:
                numeric_features.append(0.5)

        if 'RestaurantsPriceRange2' in attr_dict:
            numeric_features.append(int(attr_dict['RestaurantsPriceRange2']))
        else:
            numeric_features.append(2.5)

        count_diet = 0
        if 'DietaryRestrictions' in attr_dict:
            #diet = ['dairy-free', 'gluten-free', 'vegan', 'kosher', 'halal', 'soy-free', 'vegetarian']
            dict_diet = ast.literal_eval(attr_dict['DietaryRestrictions'])

            for k, v in dict_diet.items():
                if v:
                    count_diet += 1
        else:
            count_diet = 0.5

        numeric_features.append(count_diet)

        # BusinessParking: 'garage', 'street', 'validated', 'lot', 'valet'
        parking_feature = 0
        bus_parking = ['garage', 'validated', 'lot', 'valet']

        if 'BusinessParking' in attr_dict:
            dict_parking = ast.literal_eval(attr_dict['BusinessParking'])
            for bpk in bus_parking:
                if dict_parking[bpk]:
                    parking_feature += 1

        numeric_features.append(parking_feature)

        # GoodForMeal: 'dessert', 'latenight', 'lunch', 'dinner', 'breakfast', 'brunch'

        meals = ['dessert', 'latenight', 'lunch', 'dinner', 'breakfast', 'brunch']
        meal_feature = 0

        if 'GoodForMeal' in attr_dict:
            dict_meal = ast.literal_eval(attr_dict['GoodForMeal'])
            for index, meal in enumerate(meals):
                if meal in dict_meal:
                    if dict_meal[meal]:
                        meal_feature += 1
        else:
            meal_feature = 2

        numeric_features.append(meal_feature)

        count_best_nights = 0
        if 'BestNights' in attr_dict:
            dict_best_nights = ast.literal_eval(attr_dict['BestNights'])

            for k, v in dict_best_nights.items():
                if v:
                    count_best_nights += 1
        else:
            count_best_nights = 0.5

        numeric_features.append(count_best_nights)

        return numeric_features

    else:
        return [0.5]*20


def process_categories(a_dict):

    string_categories = a_dict.get('categories', -1)

    # ['Restaurants', 'Shopping', 'Food', 'Beauty & Spas', 'Home Services', 'Health & Medical', 'Local Services', 'Automotive', 'Nightlife'
    # , 'Bars', 'Event Planning & Services', 'Active Life', 'Fashion', 'Coffee & Tea', 'Sandwiches', 'Hair Salons', ]

    if isinstance(string_categories, str):
        return len(string_categories.split(","))
    else:
        return 3


def bus_star(a_dict):
    # median of average_stars in business.json is 3.5

    a_float = a_dict.get("stars", "na")

    if isinstance(a_float, float):
        return a_float
    else:
        return 3.5


def bus_reviews(a_dict):
    # median of review_count in business.json is 9

    an_int = a_dict.get('review_count', 'na')

    if isinstance(an_int, int):
        return an_int
    else:
        return 9


def categorize_business(a_dict, list_popular):

    string_categories = a_dict.get('categories', -1)

    if isinstance(string_categories, str):
        list_categories = string_categories.split(",")

        for category in list_categories:
            if category in list_popular:
                return category
        return "na"
    else:
        return "na"


# ***************************** end cleaning business.json ********************************************* #


# cleaning "user.json: replace a missing value by median value for all users in user.json file "


def user_star(a_dict):
    # median of average_stars in user.json is 3.92

    a_float = a_dict.get("average_stars", "na")

    if isinstance(a_float, float):
        return a_float
    else:
        return 3.92


def user_reviews(a_dict):
    # median of review_count in user.json is 5

    an_int = a_dict.get('review_count', 'na')

    if isinstance(an_int, int):
        return an_int
    else:
        return 5


def number_friends(a_dict):

    friend_string = a_dict.get("friends", -1)

    if isinstance(friend_string, str):
        if friend_string != "None" and friend_string != "" and friend_string != " ":
            return len(friend_string.split(","))
        else:
            return 0

    else:
        return 25   # median value


def yelp_since(a_dict):

    string = a_dict.get("yelping_since", -1)

    if isinstance(string, str):
        return int(string.split("-")[0])
    else:
        return 2014

# ********************** end cleaning "user.json" *************************************

# ***************************** item-based prediction module ********************************************* #


def min_hash(indices, num_users):
    '''
        f(x) = (ax+b) % m
        a, b: prime numbers
        x:[0, num_users]
        m: num_users, ideally a prime number
        Use 105 hash functions
    '''

    primes = [11, 73, 179, 283, 419, 547, 661, 811, 947, 1087, 2131, 4241, 5953, 6679, 7417]
    params = list(combinations(primes, 2))

    all_hash = []

    for param in params:
        one_hash = []

        for index in indices:
            temp = (param[0] * index + param[1]) % num_users
            one_hash.append(temp)

        all_hash.append(min(one_hash))

    return all_hash


def locality_sensitive_hashing(indices):
    ''' Use 105 hash functions, 105 = 5*21, (1/21)**(1/5) ~ 0.54
        indices is a list of 105 elements
    '''

    nRows = 5
    #nBands = 21
    buckets = []

    for idx in range(0, len(indices), nRows):
        buckets.append((idx, hash(tuple(indices[idx: idx + nRows]))))

    return buckets


def jaccard_similarity(pair, bu_dict):
    ''' Calculate the original jaccard similarity for each candidate pairs '''

    denominator = set(bu_dict[pair[0]]).union(set(bu_dict[pair[1]]))
    numerator = set(bu_dict[pair[0]]).intersection(set(bu_dict[pair[1]]))
    jaccard = len(numerator) / len(denominator)

    return jaccard


def get_similar_pairs(train, number_of_users):
    ''' find business pairs which have jaccard similarity >= jaccard_threshold, which is a hyper-parameter to tune '''

    ''' **** (business, list of numeric indices of users who rated the business) *** '''
    business_users = train.map(lambda x: (x[1][0], x[0])).groupByKey().mapValues(list).persist()
    business_users_dict = business_users.collectAsMap()

    ''' **** (business, N values of minhash) where N: number of hash functions *** '''
    signature_matrix = business_users.mapValues(lambda x: min_hash(x, number_of_users)).persist()
    business_users.unpersist()

    lsh_buckets = signature_matrix.mapValues(locality_sensitive_hashing).flatMap(lambda x: [(y, x[0]) for y in x[1]])\
        .groupByKey().mapValues(tuple).map(lambda x: x[1]).filter(lambda x: len(x) > 1).distinct().persist()

    similar_pairs = lsh_buckets.flatMap(lambda x: [y for y in combinations(x, 2)]).distinct().\
        map(lambda x: (x, jaccard_similarity(x, business_users_dict))).filter(lambda x: x[1] >= jaccard_threshold)\
        .map(lambda x: x[0]).persist()

    return similar_pairs


def normalize_rating(user_stars, co_indices):
    co_rating = np.array([v for x in user_stars for k, v in x.items() if k in co_indices])
    co_rating = co_rating - np.average(co_rating)
    return co_rating


def number_co_users(bid1, bid2, star_dict):
    ''' star_dict = {bid: ({uid: star}, {uid: star}, ... } '''
    u1_list = [[y for y in x][0] for x in star_dict[bid1]]
    u2_list = [[y for y in x][0] for x in star_dict[bid2]]

    common_users = list(set(u1_list).intersection(set(u2_list)))
    return common_users


def item_based_prediction(bid1, bid2, common_users, star_dict):

    if len(common_users) > co_rating_threshold:
        b1_stars = normalize_rating(star_dict[bid1], common_users)
        b2_stars = normalize_rating(star_dict[bid2], common_users)

        numerator = np.dot(b1_stars, b2_stars)
        if numerator == 0:
            return tuple(sorted([bid1, bid2])), 0

        denominator = np.sqrt(b1_stars.dot(b1_stars)*b2_stars.dot(b2_stars))
        if denominator == 0:
            return tuple(sorted([bid1, bid2])), 0

        return tuple(sorted([bid1, bid2])), numerator/denominator

    else:
        return tuple(sorted([bid1, bid2])), 0

# ***************************** end item-based prediction module ********************************************* #


def make_prediction(data, weight_dict, user_ave_score, bus_ave_score, max_uid, max_bid, u_dict, b_dict):
    ''' neighbors: (test_bid, list of tuples of (train_bid, star))
        weightRDD: ((bid1, bid2), pearson weight)
    '''

    pairs = []
    num_neighbors = co_rating_threshold
    unknown_score = 3.5
    test_uid = data[0]
    test_bid = data[1][0]
    neighbors = data[1][1]

    if neighbors:

        for neighbor in neighbors:

            train_bid = neighbor[0]

            if test_bid < train_bid:
                pair = tuple([test_bid, train_bid])
            else:
                pair = tuple([train_bid, test_bid])

            pairs.append((neighbor[1], weight_dict.get(pair, 0)))

        top_neighbors = sorted(pairs, key=lambda x: x[1], reverse=True)[:num_neighbors]

        scores = np.array([x[0] for x in top_neighbors])
        weights = np.array([x[1] for x in top_neighbors])

        numerator = np.dot(scores, weights)

        ''' since we only keep positive pearson weights, so don't need to do absolute values '''
        denominator = np.sum(weights)
        if numerator == 0 or denominator == 0:
            if test_uid < max_uid:
                return (u_dict[test_uid], b_dict[test_bid]), user_ave_score[test_uid]

            if test_uid >= max_uid and test_bid >= max_uid:
                return (u_dict[test_uid], b_dict[test_bid]), unknown_score

            if test_bid < max_bid:
                return (u_dict[test_uid], b_dict[test_bid]), bus_ave_score[test_bid]
        else:
            return (u_dict[test_uid], b_dict[test_bid]), numerator/denominator

    else:
        return (u_dict[test_uid], b_dict[test_bid]), unknown_score


if __name__ == '__main__':
    start = time.time()
    guess = 0.5

    sc = SparkContext('local[*]', 'task')

    train_input = sc.textFile(folder_path + 'yelp_train.csv').filter(lambda string: string[:7] != "user_id").map(get_train_data).persist()
    uid_dict = train_input.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    bid_dict = train_input.map(lambda x: x[1][0]).distinct().zipWithIndex().collectAsMap()

    # print('train: num of distinct users and business', len(uid_dict), len(bid_dict))

    max_uid = len(uid_dict) - 1
    max_bid = len(bid_dict) - 1

    trainRDD = train_input.map(lambda x: (uid_dict[x[0]], (bid_dict[x[1][0]], x[1][1]))).persist()

    val_input = sc.textFile(test_file).filter(lambda string: string[:7] != "user_id").map(get_test_data).persist()

    new_users = val_input.map(lambda x: x[0]).subtract(train_input.map(lambda x: x[0])).distinct().collect()

    new_uid = dict(zip(new_users, range(len(uid_dict), len(uid_dict) + len(new_users))))
    uid_dict.update(new_uid)

    new_business = val_input.map(lambda x: x[1]).subtract(train_input.map(lambda x: x[1][0])).distinct().collect()
    new_bid = dict(zip(new_business, list(range(len(bid_dict), len(bid_dict) + len(new_business)))))
    bid_dict.update(new_bid)

    ''' convert uid, bid in testRDD to numeric value '''
    testRDD = val_input.map(lambda x: (uid_dict[x[0]], bid_dict[x[1]])).persist()

    ''' convert numeric indices to string name '''
    inverse_uid_dict = {v: k for k, v in uid_dict.items()}
    inverse_bid_dict = {v: k for k, v in bid_dict.items()}

    train_input.unpersist()

    popular_categories = ['Restaurants', 'Shopping', 'Food', 'Beauty & Spas', 'Home Services', 'Health & Medical', 'Local Services', 'Bars',
                          'Event Planning & Services', 'Automotive', 'Nightlife', 'Active Life', 'Fashion', 'Coffee & Tea', 'Hair Salons', 'Fast Food',
                          'American (Traditional)', 'Pizza', 'Home & Garden', 'Auto Repair', 'Hotels & Travel', 'Arts & Entertainment', 'Professional Services',
                          'Real Estate', 'Grocery', 'Financial Services', 'Pet Services', 'General Dentistry', 'Hotels', 'Apartments', 'Gyms', 'Furniture Stores',
                          'Plumbing', 'Building Supplies', 'Mobile Phones']

    # need to use get in case a business does not have categories keyword
    bus_info = sc.textFile(folder_path + "business.json").map(lambda x: json.loads(x))\
        .filter(lambda x: bid_dict.get(x["business_id"], -1) != -1)\
        .map(lambda x: (bid_dict.get(x["business_id"]), [bus_star(x)] + [bus_reviews(x)] + [x.get("latitude", 0)] + [x.get("longitude", 0)]
                        + [process_categories(x)] + process_attributes(x))).collectAsMap()

    # Select information about users in user.json which appears in train and test set,
    # It is possible that in train and test has users which do not present in user.json --> cold start --> treat separately
    user_info = sc.textFile(folder_path + "user.json").map(lambda x: json.loads(x))\
        .filter(lambda x: uid_dict.get(x["user_id"], -1) != -1)\
        .map(lambda x: (uid_dict.get(x["user_id"]), [user_star(x)] + [user_reviews(x)]
                        + [number_friends(x)] + [yelp_since(x)])).collectAsMap()

    train_data = trainRDD.map(lambda x: [user_info.get(x[0], -1), bus_info.get(x[1][0], -1), x[1][1]])\
        .filter(lambda x: x[0] != -1 and x[1] != -1).map(lambda x: x[0] + x[1] + [x[2]]).collect()

    # print('number of training samples existing in both user and business.json', len(train_data))
    # all training samples exist in both user and business.json

    train_data = np.array(train_data).astype(float)

    X_train = train_data[:, :-1]

    y_train = train_data[:, -1]

    test_data = testRDD.map(lambda x: [[x[0], x[1]], user_info.get(x[0], -1), bus_info.get(x[1], -1)]).persist()

    unk_testRDD = test_data.filter(lambda x: x[1] == -1 or x[2] == -1).map(lambda x: (x[0][0], x[0][1]))

    #print('cold start', len(unk_testRDD.collect()))

    #print(unk_testRDD.collect())

    if len(unk_testRDD.collect()) > 0:
        user_ave_rating = trainRDD.map(lambda x: (x[0], x[1][1]))\
            .aggregateByKey((0, 0), lambda x, y: (x[0]+y, x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))\
            .mapValues(lambda x: x[0]/x[1]).collectAsMap()

        business_ave_rating = trainRDD.map(lambda x: (x[1][0], x[1][1]))\
            .aggregateByKey((0, 0), lambda x, y: (x[0]+y, x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))\
            .mapValues(lambda x: x[0]/x[1]).collectAsMap()

        rating_dict = trainRDD.map(lambda x: (x[1][0], {x[0]: x[1][1]})).groupByKey().mapValues(tuple).collectAsMap()

        pearson_weights = get_similar_pairs(trainRDD, max_uid + 1)\
            .map(lambda x: (x, number_co_users(x[0], x[1], rating_dict))).filter(lambda x: len(x[1]) > co_rating_threshold)\
            .map(lambda x: item_based_prediction(x[0][0], x[0][1], x[1], rating_dict))\
            .filter(lambda x: x[1] > 0).sortBy(lambda x: x[1], ascending=False).collectAsMap()

        #print(pearson_weights)

        unk_result = unk_testRDD.leftOuterJoin(trainRDD.groupByKey())\
            .map(lambda x: make_prediction(x, pearson_weights, user_ave_rating, business_ave_rating, max_uid, max_bid, inverse_uid_dict, inverse_bid_dict))\
            .collect()

        #print('unk_result', unk_result.collect())

    else:

        unk_result = []

    # check this, need to see how to impute missing values
    test_data = test_data.filter(lambda x: x[1] != -1 and x[2] != -1).map(lambda x: x[0] + x[1] + x[2]).collect()

    test_data = np.array(test_data).astype(float)

    user_bus_arr = test_data[:, :2]

    X_test = test_data[:, 2:]

    train_data = xgb.DMatrix(X_train, y_train)
    test_data = xgb.DMatrix(X_test)

    params = {'learning_rate': 0.05, 'max_depth': 8, 'colsample_bytree': 0.5,
              'random_state': 2021, 'subsample': 1, 'min_child_weight': 4, 'booster': "gbtree", 'silent': 1, 'gamma': 0.1}

    # model = xgb.train(params=params, dtrain=train_data, num_boost_round=140)
    model = xgb.train(params=params, dtrain=train_data, num_boost_round=350)
    prediction = model.predict(test_data)
    size = len(prediction)
    prediction = prediction.reshape(size, 1)

    new_features = model.predict(train_data)

    for idx, score in enumerate(prediction):
        integer = int(score)
        temp = score - integer
        if 0 <= temp < 0.1:
            prediction[idx] = integer
        if temp > 0.9:
            prediction[idx] = integer + 1

    result = np.hstack((user_bus_arr, prediction))

    with open(output_file, 'w') as outfile:
        ''' Need to reformat it, no space, for submission '''
        outfile.write('user_id, business_id, prediction\n')

        for idx in range(0, size):
            outfile.write(inverse_uid_dict[result[idx][0]] + "," + inverse_bid_dict[result[idx][1]] + "," + str(result[idx][2]) + "\n")

        for pair in unk_result:
            outfile.write(pair[0][0] + "," + pair[0][1] + "," + str(pair[1]) + "\n")

    duration = time.time() - start
    print("Duration: " + str(duration))




