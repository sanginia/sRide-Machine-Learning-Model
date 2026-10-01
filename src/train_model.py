import pandas as pd
import numpy as np
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier

#Loading data
df = pd.read_csv("data/ride_requests.csv")

print("Loaded data:", df.shape)
print("\nTarget distribution:")
print(df["completed_label"].value_counts(normalize=True))

#Creating new features 

def safe_rate(numerator, denominator):
    
    # Avoid division by zero.
    # If denominator is 0, return NaN.
    # Later NaN values will be filled.

    return numerator / denominator.replace(0, np.nan)

def add_features(df):
    df = df.copy()

    #User A behavior features 
    df["userCompletionRate"] = safe_rate(df["userCompletedCount"], df["userRequestedCount"])
    df["userCancellationRate"] = safe_rate(df["userCancelledCount"], df["userRequestedCount"])
    df["userAcceptanceRate"] = safe_rate(df["userAcceptedCount"], df["userRequestedCount"])
    df["userDeclineRate"] = safe_rate(df["userDeclinedCount"], df["userRequestedCount"])

    # Matched User B behavior features 
    df["matchedUserCompletionRate"] = safe_rate(df["matchedUserCompletedCount"], df["matchedUserRequestedCount"])
    df["matchedUserCancellationRate"] = safe_rate(df["matchedUserCancelledCount"], df["matchedUserRequestedCount"])
    df["matchedUserAcceptanceRate"] = safe_rate(df["matchedUserAcceptedCount"], df["matchedUserRequestedCount"])
    df["matchedUserDeclineRate"] = safe_rate(df["matchedUserDeclinedCount"], df["matchedUserRequestedCount"])

    # Distance features 
    df["pickupDistanceKm"] = df["actualStartDistanceInMeter"] / 1000
    df["dropDistanceKm"] = df["actualEndDistanceInMeter"] / 1000
    df["airPickupDistanceKm"] = df["airStartDistanceInMeter"] / 1000
    df["airDropDistanceKm"] = df["airEndDistanceInMeter"] / 1000

    # Trip distance difference
    df["tripDistanceDifferenceKm"] = (df["userTripDistanceMeter"] - df["matchedUserTripDistanceMeter"]).abs() / 1000

    # Time feature
    df["rideTimeDifferenceAbs"] = df["rideTimeDifferenceInMinute"].abs()

    # Compatibility features 
    df["sameGender"] = (df["userGender"] == df["matchedUserGender"]).astype(int)
    df["sameCompanyName"] = (df["userCompany"] == df["matchedUserCompany"]).astype(int)
    df["bothSubscribed"] = ((df["userSubscriptionActive"] == 1) & (df["matchedUserSubscriptionActive"] == 1)).astype(int)

    # Review difference
    df["reviewPointDifference"] = (df["userReviewPoint"] - df["matchedUserReviewPoint"]).abs()

    # Difference in number of completed rides
    df["completedRideCountDifference"] = (df["userTripTotalCompletedRides"] - df["matchedUserTotalCompletedRides"]).abs()

    # Lowest gap in pickup and drop distance between two users 
    df["totalRouteInconvenienceKm"] = (df["actualStartDistanceInMeter"] + df["actualEndDistanceInMeter"]) / 1000

    # Pickup distance between two users relative to the overall trip distance 
    df["pickupDistanceToTripRatio"] = (df["actualStartDistanceInMeter"] / df["userTripDistanceMeter"].replace(0, np.nan))
    # 2km detour is bad for a 5km trip, but not so much for a 50km trip

    # Drop off distance between two users relative to the overall trip distance 
    df["dropDistanceToTripRatio"] = (df["actualEndDistanceInMeter"] / df["userTripDistanceMeter"].replace(0, np.nan))

    # Reliability features 
    df["completionRateDifference"] = (df["userCompletionRate"] - df["matchedUserCompletionRate"]).abs()
    df["cancellationRateDifference"] = (df["userCancellationRate"] - df["matchedUserCancellationRate"]).abs()

    # Combined reliability scores 
    df["pairCompletionStrength"] = (df["userCompletionRate"] * df["matchedUserCompletionRate"])
    df["pairCancellationRisk"] = (df["userCancellationRate"] + df["matchedUserCancellationRate"])

    # Average review strength of both users
    df["pairReviewStrength"] = (df["userReviewPoint"] + df["matchedUserReviewPoint"]) / 2

    # Candidate relative ranking features
    df["pickupDistanceRankForUser"] = df.groupby("user")["actualStartDistanceInMeter"].rank(method="average", ascending=True)  # Is this candidate one of the closest pickups for this user?

    df["dropDistanceRankForUser"] = df.groupby("user")["actualEndDistanceInMeter"].rank(method="average", ascending=True)

    df["timeDifferenceRankForUser"] = df.groupby("user")["rideTimeDifferenceAbs"].rank(method="average", ascending=True)       # Is this candidate one of the best time matches for this user?

    df["matchedRatingRankForUser"] = df.groupby("user")["matchedUserReviewPoint"].rank(method="average", ascending=False)       # Is this candidate one of the highest-rated options for this user?

    df["pickupDistancePercentileForUser"] = df.groupby("user")["actualStartDistanceInMeter"].rank(pct=True, ascending=True)

    df["timeDifferencePercentileForUser"] = df.groupby("user")["rideTimeDifferenceAbs"].rank(pct=True, ascending=True)

    df["matchedRatingPercentileForUser"] = df.groupby("user")["matchedUserReviewPoint"].rank(pct=True, ascending=False)

    return df

df = add_features(df)



target = "completed_label"

drop_cols = [
    "completed_label", # final prediction, so can't use it when training
    "objectId",        # causes model to memorize 
    "user",            # causes model to memorize 
    "matchedUser"      # causes model to memorize 
]

X = df.drop(columns=drop_cols)
y = df[target]

# Splitting training and testing data
train_end = int(len(df) * 0.8)    # 80% training, 20% testing 

X_train = X.iloc[:train_end]
X_test = X.iloc[train_end:]

y_train = y.iloc[:train_end]
y_test = y.iloc[train_end:]

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# Preprocessing 
# separates into numeric and categorical data fields 
# fills NaN values in numeric fields with median 
# fills NaN values in categorical fields with most frequently appearing value 
# categorical data fields get one  hot encoding (columns with strings become numerical. eg: geoCity = Bangalore becomes geoCity_Bangalore)
# last line tells it what to do for categorical/ numerical columns (eg: For categorical columns: use categorical_transformer, fill missing values with most frequent value, one-hot encode the text)
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

print("\nCategorical columns:", categorical_cols)
print("Number of numeric columns:", len(numeric_cols))

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols),("cat", categorical_transformer, categorical_cols)])


# Building model

negative_count = (y_train == 0).sum()   # counts rows where completed_label == 0
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count  # calculates how much extra weight to gve positive class

print("\nNegative count:", negative_count) 
print("Positive count:", positive_count)
print("Scale pos weight:", scale_pos_weight)

# collection of many small trees 
model = XGBClassifier(
    n_estimators=600,    # build 300 trees
    max_depth=4,         # controls complexity of each tree
    learning_rate=0.03,  # lower learning rate = model learns slowly and carefully 
    subsample=0.85,       # each tree trains on 85% of the rows to prevent overfitting. Each tree sees a different sample
    colsample_bytree=0.85, # each tree uses 85% of columns to prevent overfitting and the model relying heavily on one column
    min_child_weight=5,    # model becomes more careful when making branches in decision trees, prevents overreaction to small patterns 
    gamma=0.5,             # higher gamma allows splits that are useful, prevents model from making useless rules
    reg_lambda=2.0,        # prevents model from putting too much weight on one pattern, prevents obsession with one feature
    reg_alpha=0.2,         # helps ignore the influence of weak features that do not add much value by pushing their effect closer to zero
    eval_metric="logloss",  # how to measure prediction quality (logloss for probability predictions)
    scale_pos_weight=scale_pos_weight, # completed rides are rare, so model needs to care more about predicting completed rides correctly
    random_state=42, # makes results reproducible
    n_jobs=-1 # use all CPU power available, makes training faster
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)])  # combines preprocessing and model into one object

#Training model 
print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# Feature importance
xgb_model = pipeline.named_steps["model"]   # trained XGBoost classifier

importance = xgb_model.feature_importances_   # XGBoost calculates how useful each feature was during training

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()     # gets the names of the features after preprocessing

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)   # creates a sorted table in pandas with two columns: feature and importance

print("\nTop 20 most important features:")
print(feature_importance.head(20))


# Evaluate classification performance 
test_probs = pipeline.predict_proba(X_test)[:, 1]   # gives probability of completed_label = 1

roc_auc = roc_auc_score(y_test, test_probs) # if I randomly pick one completed ride and one non completed ride, how often does the model give the completed ride a higher score?
avg_precision = average_precision_score(y_test, test_probs) # When the model gives high scores, how often are those actually completed rides? (eg: Base rate: 0.12, Average Precision: 0.29, then model is better than random chance)

print("\nClassification metrics:")
print("ROC AUC:", roc_auc)
print("Average Precision:", avg_precision)


# Evaluate ranking performance

def ranking_metrics(test_meta, score_col, k=5): # test_meta = test data with user, matchedUser, completed_label, and scores
    precision_scores = []
    recall_scores = []
    hit_rates = []

    for user_id, group in test_meta.groupby("user"):
        top_k = group.sort_values(score_col, ascending=False).head(k) # sorts the user's matches from highest to lowest scores

        if len(top_k) == 0:
            continue            # if no matches, skip 

        total_completed_for_user = group["completed_label"].sum()  # how many completed rides user had over all matches
        hits_in_top_k = top_k["completed_label"].sum()         # how many completed rides occurred in top k 

        precision_scores.append(hits_in_top_k / len(top_k))     # fraction of completed rides (completed_label = 1) over recommendations 

        if total_completed_for_user > 0:
            recall_scores.append(hits_in_top_k / total_completed_for_user)  # Out of all the completed matches available for this user, how many did the model place in the top k?
            hit_rates.append(1 if hits_in_top_k > 0 else 0)         # hit rate = 1 if top k contains at least 1 ride, otherwise 0 

    return {
        f"precision@{k}": float(np.mean(precision_scores)),
        f"recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0,
        f"hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0              # returns average across all users 
    }

def ndcg_at_k(test_meta, score_col, k=5):  # Normalized Discounted Cumulative Gain - asks whether the model put the completed rides near the top of the recommendation list
                                           # better than precision as that only asks how many completed rides are in the top 5
    ndcg_scores = []

    for user_id, group in test_meta.groupby("user"):
        if len(group) < k:
            continue        # skips users who don't have enough candidates

        ranked = group.sort_values(score_col, ascending=False).head(k)     # sorts a user's candidate matches by the predicted completion probability 
        labels = ranked["completed_label"].values       # extracts actual completed-label outcomes for the matches 

        dcg = 0
        for i, label in enumerate(labels):
            dcg += label / np.log2(i + 2)          # A completed ride at rank 1 is worth more than a completed ride at rank 5
                                                   # formula gives higher weight to higher ranks 
                                                   # eg: Rank 1 gets divided by log2(2) = 1, Rank 2 gets divided by log2(3) = 1.58, Rank 3 gets divided by log2(4) = 2, Rank 4 gets divided by log2(5) = 2.32

        # ideal ranking - sorts by the actual answer, so completed rides come first. 
        ideal = group.sort_values("completed_label", ascending=False).head(k)
        ideal_labels = ideal["completed_label"].values

        idcg = 0
        for i, label in enumerate(ideal_labels):
            idcg += label / np.log2(i + 2)          # perfect possible ranking score  

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)          # divides model's score by the perfect ranking score
                                                    # NDCG is between 0 (terrible) and 1 (perfect)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0

test_meta = df.iloc[train_end:][
    [
        "user",
        "matchedUser",
        "completed_label",
        "actualStartDistanceInMeter"
    ]].copy()                         # creates a smaller table with only these fields, takes only the test portion (20%) of the data

test_meta["model_score"] = test_probs       # adds the predicted completion probability to table

# Baselines: nearest pickup, smallest time difference, highest matched rating, lowest matched cancellation
test_meta["closest_pickup_score"] = -test_meta["actualStartDistanceInMeter"]
test_meta["smallest_time_difference_score"] = -df.iloc[train_end:]["rideTimeDifferenceInMinute"].abs()
test_meta["highest_matched_rating_score"] = df.iloc[train_end:]["matchedUserReviewPoint"]
test_meta["lowest_matched_cancellation_score"] = -df.iloc[train_end:]["matchedUserCancellationRate"].fillna(0)

print("\nRanking metrics:")

for k in [1, 3, 5, 10]:
    print(f"\nModel top {k}:")
    print(ranking_metrics(test_meta, "model_score", k=k))    # sort each user's matches by predicted completion probability and calculate precision@k, recall@k, and hit_rate@k.

    print(f"Closest pickup baseline:")
    print(ranking_metrics(test_meta, "closest_pickup_score", k=k))      #sort each user's matches by by closest pickup distance and calculate the same metrics
                                                                        # This helps compare ML model vs closest pickup distance as a baseline
                                                                        # The model is only useful if its top k recommendations beat the closest pickup baseline
    print("Smallest time difference baseline:")
    print(ranking_metrics(test_meta, "smallest_time_difference_score", k=k))    # sort the user's matches by closest time match 

    print("Highest matched rating baseline:")
    print(ranking_metrics(test_meta, "highest_matched_rating_score", k=k))   # sort the user's matches by highest rated second user

    print("Lowest matched cancellation baseline:")
    print(ranking_metrics(test_meta, "lowest_matched_cancellation_score", k=k))    # sort the user's matches by lowest cancellation user

    print("\nNDCG metrics:")
    print(f"\nNDCG@{k}")
    print("Model:", ndcg_at_k(test_meta, "model_score", k=k))
    print("Closest pickup:", ndcg_at_k(test_meta, "closest_pickup_score", k=k))

results = []

base_completion_rate = y_test.mean()

for k in [1, 3, 5, 10]:
    model_metrics = ranking_metrics(test_meta, "model_score", k=k)
    closest_metrics = ranking_metrics(test_meta, "closest_pickup_score", k=k)

    results.append({
        "k": k,
        "model_precision": model_metrics[f"precision@{k}"],
        "closest_pickup_precision": closest_metrics[f"precision@{k}"],
        "model_hit_rate": model_metrics[f"hit_rate@{k}"],
        "closest_pickup_hit_rate": closest_metrics[f"hit_rate@{k}"],
        "lift_over_base_rate": model_metrics[f"precision@{k}"] / base_completion_rate,
        "lift_over_closest_pickup": (
            model_metrics[f"precision@{k}"] / closest_metrics[f"precision@{k}"]
            if closest_metrics[f"precision@{k}"] > 0 else None
        )
    })

results_df = pd.DataFrame(results)

print("\nFinal ranking comparison:")
print(results_df)

# Save model 
os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/ride_completion_model.pkl")

print("\nModel saved to models/ride_completion_model.pkl")

