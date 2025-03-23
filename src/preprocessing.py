import pandas as pd

def fill_missing_embarked(df):
    mode = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode, inplace=True)
    return df

def fill_missing_age(df):
    # Create a median lookup table based on Pclass and Sex
    age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median()

    # Fill missing Age using the median from the corresponding Pclass & Sex group
    def fill_age(row):
        if pd.isnull(row['Age']):
            return age_medians.loc[row['Pclass'], row['Sex']]
        else:
            return row['Age']

    df['Age'] = df.apply(fill_age, axis=1)
    return df

def simplify_cabin_column(df):
    # Extract first letter of cabin (the deck level)
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'Unknown')
    return df

def extract_title(df):
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)

    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'the Countess': 'Rare',
        'Capt': 'Rare', 'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare',
        'Rev': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
    })

    return df

def add_family_features(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

def add_age_band(df):
    df['AgeBand'] = pd.cut(
        df['Age'],
        bins=[0, 16, 25, 40, 60, 100],
        labels=[0, 1, 2, 3, 4],
        right=True,
        include_lowest=True
    )
    return df

def add_fare_band(df):
    df['FareBand'] = pd.qcut(
        df['Fare'],
        q=4,
        labels=[0, 1, 2, 3]
    )
    return df

def encode_categoricals(df):
    # Encode Sex manually
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # Encode Embarked
    df['Embarked'] = pd.factorize(df['Embarked'])[0]

    # Encode Title
    df['Title'] = pd.factorize(df['Title'])[0]

    # Encode Cabin (deck)
    df['Cabin'] = pd.factorize(df['Cabin'])[0]

    # AgeBand and FareBand already contain numeric labels
    df['AgeBand'] = df['AgeBand'].astype(int)
    df['FareBand'] = df['FareBand'].astype(int)

    return df

def drop_unused_columns(df, keep_continuous=True):
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch']
    if not keep_continuous:
        drop_cols += ['Age', 'Fare']
    return df.drop(columns=drop_cols)
