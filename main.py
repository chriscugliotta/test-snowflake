import os
import pandas
from pandas import DataFrame
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from snowflake.connector import connect, DictCursor
from typing import List
root = Path(__file__).parent



def get_connection():
    """Returns a Snowflake connection."""
    return connect(
        user='chriscugliottademo',
        password=os.environ['SNOWFLAKE_PASSWORD'],
        account='fz79423.us-east-2.aws',
        database='db1',
        schema='schema1',
        warehouse='house1',
    )


def execute(sql: str):
    """Executes a single (or list of) SQL statement(s)."""
    statements = [sql] if isinstance(sql, str) else sql
    with get_connection() as connection:
        for statement in statements:
            print(f'Executing SQL:  \n{statement.strip()}')
            connection.cursor().execute(statement)


def select(sql: str) -> DataFrame:
    """Executes a SQL select statement then returns the result as a Pandas dataframe."""
    with get_connection() as connection:
        print(f'Executing SQL:  \n{sql}')
        cursor = connection.cursor(DictCursor)
        cursor.execute(sql)
        rows = cursor.fetchall()
        df = DataFrame(rows)
        df = df.rename(columns={x: x.lower() for x in df})
        print(f'Selected {len(df):,} rows.')
        return df


def insert(table: str, csv_path: Path = None, df: DataFrame = None):
    """Inserts a local CSV (or dataframe) into a Snowflake table."""

    # Prepare local CSV.
    if df is not None:
        csv_path = root / f'{table}.csv'
        print(f'Writing {len(df):,} rows to {csv_path}.')
        df.to_csv(csv_path, index=False)

    # Stage, copy into, unstage.
    execute(sql=[
        f'put file://{csv_path} @~/staged overwrite = true',
        f'truncate table {table}',
        f'copy into {table} from @~/staged/{csv_path.name + ".gz"} file_format = (type = csv, skip_header = 1)',
        f'remove @~/staged',
    ])


def initialize_schema():
    """Initializes a test Snowflake schema."""
    execute(sql=[
        'create warehouse if not exists house1 with warehouse_size = xsmall',
        'create database if not exists db1',
        'create schema if not exists db1.schema1',
        get_table_ddl_houses(),
        get_table_ddl_predictions(),
        get_table_ddl_importances(),
    ])


def get_table_ddl_houses() -> str:
    return '''
create table if not exists houses (
    id integer,
    ms_sub_class varchar2(10),
    ms_zoning varchar2(10),
    lot_frontage varchar2(10),
    lot_area integer,
    street varchar2(10),
    alley varchar2(10),
    lot_shape varchar2(10),
    land_contour varchar2(10),
    utilities varchar2(10),
    sale_price integer
)
'''


def get_table_ddl_predictions() -> str:
    return '''
create table if not exists predictions (
    id integer,
    sale_price integer,
    predicted_price number(38, 2),
    dataset varchar2(10)
)
'''


def get_table_ddl_importances() -> str:
    return '''
create table if not exists importances (
    feature_name varchar2(100),
    importance number(38, 36)
)
'''


def get_model_input_data():
    """Extract, cleanse, and one-hot encode."""
    return (
        select(sql='select * from houses')
        .set_index('id')
        .assign(lot_frontage=lambda x: x['lot_frontage'].fillna(0).astype(float))
        .pipe(one_hot_encode)
    )


def one_hot_encode(df: DataFrame) -> DataFrame:
    """Converts string columns into one-hot-encoded booleans."""
    new = DataFrame()
    for column, data_type in df.dtypes.items():
        if data_type == 'object':
            one_hots = pandas.get_dummies(df[column], prefix=column)
            new = pandas.concat([new, one_hots], axis=1)
        else:
            new[column] = df[column]
    return new


def train_model(df: DataFrame) -> List[DataFrame]:
    """Trains a random forest regressor model."""

    # Get test/train split.
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Train model.
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict prices.
    p_test = model.predict(X_test)
    p_train = model.predict(X_train)

    # Calculate MAPE.
    mape_test = mean_absolute_error(y_test, p_test) / y_test.mean()
    mape_train = mean_absolute_error(y_train, p_train) / y_train.mean()
    print(f'mape_test = {mape_test}')
    print(f'mape_train = {mape_train}')

    # Return dataframes.
    return (
        get_df_predictions(y_test, y_train, p_test, p_train),
        get_df_importances(X_train, model),
    )


def get_df_predictions(y_test, y_train, p_test, p_train) -> DataFrame:
    """Returns a dataframe containing actual vs. predicted sale prices."""

    # Get predictions on test set.
    df_p_test = (
        DataFrame(p_test, columns=['predicted_price'], index=y_test.index)
        .join(y_test)
        .assign(dataset='test')
    )

    # Get predictions on train set.  (These predictions are cheating, but included as a curiosity.)
    df_p_train = (
        DataFrame(p_train, columns=['predicted_price'], index=y_train.index)
        .join(y_train)
        .assign(dataset='train')
    )

    # Combine them.
    return (
        pandas.concat([df_p_test, df_p_train])
        .reset_index()
        .sort_values(by=['id'])
        .loc[:, ['id', 'sale_price', 'predicted_price', 'dataset']]
    )


def get_df_importances(X_train: DataFrame, model: RandomForestRegressor) -> DataFrame:
    """Returns a dataframe containing each feature and its corresponding importance value."""
    return (
        DataFrame({
            'feature_name': X_train.columns.tolist(),
            'importance': model.feature_importances_
        })
        .sort_values(by='importance', ascending=False)
    )


def cleanup():
    (root / 'predictions.csv').unlink()
    (root / 'importances.csv').unlink()
    # execute(sql=[
    #     'drop database if exists db1',
    #     'drop warehouse if exists house1',
    # ])


if __name__ == '__main__':
    print('Begin.')
    initialize_schema()
    insert(table='houses', csv_path=root / 'houses.csv')
    df_inputs = get_model_input_data()
    df_predictions, df_importances = train_model(df_inputs)
    insert(table='predictions', df=df_predictions)
    insert(table='importances', df=df_importances)
    cleanup()
    print('Done.')
