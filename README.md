# test-snowflake

I created this codebase as an opportunity to learn and demo Snowflake.  To run this code, I used a [Snowflake Trial Account](https://signup.snowflake.com/) and also the [Snowflake Connector for Python](https://docs.snowflake.com/en/user-guide/python-connector.html).  The [main script](/main.py#L213) performs the following actions:

- [Creates](/main.py#L66) a test warehouse, database, and schema.
- [Inserts](/main.py#L48) this [Kaggle house prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) into a Snowflake [table](/main.py#L78).
- [Extracts](/main.py#L35) the data from Snowflake into a Pandas dataframe.
- [Prepares](/main.py#L116) the data for modeling, e.g. one-hot encoding, test/train split, etc.
- Uses the data to [train](/main.py#L138) an [SKLearn random forest regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
- Inserts the model's [predictions](/main.py#L167) and [importance factors](/main.py#L193) into Snowflake [tables](/main.py#L96).
