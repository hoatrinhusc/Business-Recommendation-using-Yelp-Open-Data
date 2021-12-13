# Business Recommendation using Yelp Open-Data
## Summary
In this project, I will use the raw data from [Yelp open dataset](https://www.yelp.com/dataset) to extract user and business information. 

Then I build a recommender to predict the rating score a user would give for a business. A business will be recommended to a user if the predicted rating score is high.
## Dataset
All files can be downloaded from [Yelp open dataset](https://www.yelp.com/dataset).

Among those files, yelp_academic_dataset_business.json and yelp_academic_dataset_user.json are used to mine user and business information.

yelp_academic_dataset_review.json is used to generate rating score for each user-business pai. I then select 60% of that data for training, 20% for validation and 20% for testing.
## Feature selection
I use XGBoost to predict the rating. The features are: average star of user and business, attributes of business,
number of user's friends, longtitude, latitude, review_count and total number of categories a restaurant has.
## Cold Start: New user-New business
For new business, item-based collaborative filtering is used to predict rating.

For new user or new user and new business, a default value 3.5 is used for rating score.
