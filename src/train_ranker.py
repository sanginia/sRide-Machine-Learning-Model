import pandas as pd
import numpy as np
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRanker

# Feature engineering

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

    df["candidateCountForUser"] = df.groupby("user")["matchedUser"].transform("count")   # how many possible matches a user has 

    df["pickupRankShareForUser"] = (df["pickupDistanceRankForUser"] / df["candidateCountForUser"])   # How good is this candidate’s pickup distance compared to the user’s other candidates? Value near 0 is closest, 1 is farthest

    df["timeRankShareForUser"] = (df["timeDifferenceRankForUser"] / df["candidateCountForUser"]) # ranks candidates by how close their ride time is to the particular user. Value near 0 is lowest time, 1 is greatest time

    df["ratingRankShareForUser"] = (df["matchedRatingRankForUser"] / df["candidateCountForUser"])  # ranks candidates by rating for a particular user. Value near 0 is highest rated options for this user, 1 is lowest rated options

    # Reliability-rank features
    # df["matchedCompletionRateRankForUser"] = df.groupby("user")["matchedUserCompletionRate"].rank(method="average", ascending=False)   # ranks each candidate by their completion rate compared to the other candidates for a particular user. higher completion rate = better rank

    # df["matchedCancellationRateRankForUser"] = df.groupby("user")["matchedUserCancellationRate"].rank(method="average", ascending=True)  # ranks candidates by cancellation rate. rank 1 = lowest cancellation rate

    # df["completionRateRankShareForUser"] = (df["matchedCompletionRateRankForUser"] / df["candidateCountForUser"])  # normalizes matched completion rate rank for user

    # df["cancellationRateRankShareForUser"] = (df["matchedCancellationRateRankForUser"] / df["candidateCountForUser"])   # normalizes matched cancellation rate rank for user
    
    # Interaction features
    # df["closePickupAndCloseTime"] = ((df["pickupRankShareForUser"] <= 0.33) & (df["timeRankShareForUser"] <= 0.33)).astype(int)   # checks whether a candidate is in the top third for both pickup distance and time match. If yes: closePickupAndCloseTime = 1, otherwise 0

    # df["closePickupAndHighRating"] = ((df["pickupRankShareForUser"] <= 0.33) & (df["ratingRankShareForUser"] <= 0.33)).astype(int)    # checks whether a candidate is in the top third for both pickup distance and rating. If yes: closePickupAndHighRating = 1, otherwise 0

    # df["goodTimeAndHighRating"] = ((df["timeRankShareForUser"] <= 0.33) & (df["ratingRankShareForUser"] <= 0.33)).astype(int)   # checks whether a candidate is in the top third for both time match and rating. If yes: goodTimeAndHighRating = 1, otherwise 0

    # df["strongCandidateOverall"] = ((df["pickupRankShareForUser"] <= 0.33) & (df["timeRankShareForUser"] <= 0.33) & (df["ratingRankShareForUser"] <= 0.33)).astype(int)   # checks whether a candidate is in the top third for pickup distance, time match, and rating. If yes: strongCandidateOverall = 1, otherwise 0
    
    return df

# Load data
df = pd.read_csv("data/ride_requests.csv") 
df = add_features(df)

# Split training and testing data
train_end = int(len(df) * 0.8) # 80% training, 20% testing 

train_df = df.iloc[:train_end].copy()
test_df = df.iloc[train_end:].copy()

# Filter A
# train_user_counts = train_df.groupby("user").size()
# valid_train_users = train_user_counts[train_user_counts >= 2].index
# train_df = train_df[train_df["user"].isin(valid_train_users)].copy()     # removes training users who only have 1 candidate. 
#                                                                         # because a ranker cannot learn ranking from a group with only one item as there is nothing to rank

# Filter B
# group_stats = train_df.groupby("user")["completed_label"].agg(["sum", "count"])

# valid_train_users = group_stats[
#     (group_stats["sum"] >= 1) &
#     (group_stats["sum"] < group_stats["count"])
# ].index

# train_df = train_df[train_df["user"].isin(valid_train_users)].copy()

# print("\nAfter filtering train groups:")
# print("Training rows:", len(train_df))
# print("Training users:", train_df["user"].nunique())

# sort so that all rows of candidate matches for a user are together, which is needed by XGBRanker 
train_df = train_df.sort_values("user").reset_index(drop=True)
test_df = test_df.sort_values("user").reset_index(drop=True)

target = "completed_label"

drop_cols = [           # remove these cols from model input 
    "completed_label",  # final prediction, so can't use it when training
    "objectId",        # causes model to memorize 
    "user",            # causes model to memorize 
    "matchedUser"      # causes model to memorize 
]                       

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]

# Tells XGBRanker how many candidate rows each user has
train_groups = train_df.groupby("user").size().to_list()
test_groups = test_df.groupby("user").size().to_list()


# Preprocessing
# separates into numeric and categorical data fields 
# fills NaN values in numeric fields with median 
# fills NaN values in categorical fields with most frequently appearing value 
# categorical data fields get one  hot encoding (columns with strings become numerical. eg: geoCity = Bangalore becomes geoCity_Bangalore)
# last line tells it what to do for categorical/ numerical columns (eg: For categorical columns: use categorical_transformer, fill missing values with most frequent value, one-hot encode the text)
categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols), ("cat", categorical_transformer, categorical_cols)])

X_train_processed = preprocessor.fit_transform(X_train)      # transform training data according to rules defined above 
X_test_processed = preprocessor.transform(X_test)            # transform test data 


# Train ranker

ranker = XGBRanker(
    objective="rank:pairwise", # Learn which candidate should be ranked above another candidate
    n_estimators=700,        # build 500 trees
    max_depth=3,            # controls complexity of each tree, each tree can ask up to 4 levels of questions
    learning_rate=0.03,     # lower learning rate = model learns slowly and carefully    
    subsample=0.9,         # each tree trains on 85% of the rows to prevent overfitting. Each tree sees a different sample
    colsample_bytree=0.9,  # each tree uses 85% of columns to prevent overfitting and the model relying heavily on one column
    min_child_weight=5,     # model becomes more careful when making branches in decision trees, prevents overreaction to small patterns 
    gamma=0.2,
    reg_lambda=2.0,
    reg_alpha=0.1,
    random_state=42,        # makes results reproducible
    n_jobs=-1               # use all CPU power available, makes training faster
)

print("Training XGBRanker...")
ranker.fit(
    X_train_processed,
    y_train,
    group=train_groups      # tells the model which rows belong to which user 
)
print("Training complete.")

# output top 10 features being used 

feature_names = preprocessor.get_feature_names_out()
booster = ranker.get_booster()

importance_dict = booster.get_score(
    importance_type="gain"
)

importance_rows = []

for index, feature_name in enumerate(feature_names):
    importance_rows.append({
        "Feature": feature_name,
        "Importance": importance_dict.get(
            f"f{index}",
            0.0
        )
    })

importance_df = pd.DataFrame(importance_rows)

total_importance = importance_df["Importance"].sum()

importance_df["ImportancePercent"] = (
    importance_df["Importance"]
    / total_importance
    * 100
    if total_importance > 0
    else 0.0
)

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

importance_df.index = importance_df.index + 1
importance_df.index.name = "Rank"

print("\nTop 10 Feature Importance by Gain:")
print(importance_df.head(10).to_string())


# Evaluation helpers

# def ranking_metrics(test_meta, score_col, k=5, min_candidates=None):  # test_meta = test data with user, matchedUser, completed_label, and scores
#     precision_scores = []
#     recall_scores = []
#     hit_rates = []

#     if min_candidates is None:
#         min_candidates = k     # test_meta = test data with user, matchedUser, completed_label, and scores

#     for user_id, group in test_meta.groupby("user"):
#         if len(group) < min_candidates:
#             continue                        # skip users without enough candidates

#         top_k = group.sort_values(score_col, ascending=False).head(k)     # sorts the user's matches from highest to lowest scores

#         total_completed_for_user = group["completed_label"].sum()       # how many completed rides user had over all matches
#         hits_in_top_k = top_k["completed_label"].sum()                  # how many completed rides occurred in top k 

#         precision_scores.append(hits_in_top_k / k)                      # fraction of completed rides (completed_label = 1) over recommendations 

#         if total_completed_for_user > 0:
#             recall_scores.append(hits_in_top_k / total_completed_for_user)      # Out of all the completed matches available for this user, how many did the model place in the top k?
#             hit_rates.append(1 if hits_in_top_k > 0 else 0)                      # hit rate = 1 if top k contains at least 1 ride, otherwise 0 

#     return {
#         f"precision@{k}": float(np.mean(precision_scores)) if precision_scores else 0,
#         f"recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0,
#         f"hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0,         # returns average across all users 
#         "users_evaluated": len(precision_scores)
#     }

def ranking_metrics(test_meta, score_col, k=5, min_candidates=None, group_col="user",):
    """
    Calculate ranking metrics using two clearly separated populations.

    Overall metrics:
        Include every group with enough candidates.
        Groups with no positive candidates contribute zero hits.

    Conditional metrics:
        Include only groups containing at least one positive candidate.
        These measure ranking performance when success is possible.
    """

    if min_candidates is None:
        min_candidates = k

    overall_precision_scores = []
    overall_hit_scores = []

    conditional_precision_scores = []
    conditional_recall_scores = []
    conditional_hit_scores = []

    overall_users = 0
    positive_users = 0

    total_hits = 0
    total_recommendations = 0

    for group_id, group in test_meta.groupby(group_col):
        if len(group) < min_candidates:
            continue

        overall_users += 1

        ranked = group.sort_values(score_col, ascending=False)
        top_k = ranked.head(k)

        positives_available = int(group["completed_label"].sum())
        hits_in_top_k = int(top_k["completed_label"].sum())

        # Use the number actually shown.
        # This equals k because groups shorter than k are skipped.
        recommendations_shown = len(top_k)

        group_precision = hits_in_top_k / recommendations_shown
        group_hit = int(hits_in_top_k > 0)

        # Overall population: includes zero-positive groups.
        overall_precision_scores.append(group_precision)
        overall_hit_scores.append(group_hit)

        total_hits += hits_in_top_k
        total_recommendations += recommendations_shown

        # Conditional population: only users where success was possible.
        if positives_available > 0:
            positive_users += 1

            conditional_precision_scores.append(group_precision)
            conditional_recall_scores.append(
                hits_in_top_k / positives_available
            )
            conditional_hit_scores.append(group_hit)

    return {
        # Macro precision: each user receives equal weight.
        f"overall_macro_precision@{k}": (
            float(np.mean(overall_precision_scores))
            if overall_precision_scores else 0.0
        ),

        # Micro precision: each shown recommendation receives equal weight.
        f"overall_micro_precision@{k}": (
            total_hits / total_recommendations
            if total_recommendations else 0.0
        ),

        # Request-level success across all eligible users.
        f"overall_hit_rate@{k}": (
            float(np.mean(overall_hit_scores))
            if overall_hit_scores else 0.0
        ),

        # Metrics only where at least one positive existed.
        f"conditional_precision@{k}": (
            float(np.mean(conditional_precision_scores))
            if conditional_precision_scores else 0.0
        ),
        f"conditional_recall@{k}": (
            float(np.mean(conditional_recall_scores))
            if conditional_recall_scores else 0.0
        ),
        f"conditional_hit_rate@{k}": (
            float(np.mean(conditional_hit_scores))
            if conditional_hit_scores else 0.0
        ),

        "overall_users_evaluated": overall_users,
        "positive_users_evaluated": positive_users,
        "zero_positive_users": overall_users - positive_users,
    }


# def ndcg_at_k(test_meta, score_col, k=5):           # Normalized Discounted Cumulative Gain - asks whether the model put the completed rides near the top of the recommendation list
#                                                     # better than precision as that only asks how many completed rides are in the top 5
#     ndcg_scores = []

#     for user_id, group in test_meta.groupby("user"):
#         if len(group) < k:
#             continue                # skips users who don't have enough candidates

#         ranked = group.sort_values(score_col, ascending=False).head(k)        # sorts a user's candidate matches by the predicted completion probability 
#         labels = ranked["completed_label"].values                           # extracts actual completed-label outcomes for the matches 

#         dcg = 0
#         for i, label in enumerate(labels):
#             dcg += label / np.log2(i + 2)        # A completed ride at rank 1 is worth more than a completed ride at rank 5
#                                                    # formula gives higher weight to higher ranks 
#                                                    # eg: Rank 1 gets divided by log2(2) = 1, Rank 2 gets divided by log2(3) = 1.58, Rank 3 gets divided by log2(4) = 2, Rank 4 gets divided by log2(5) = 2.32

#         # ideal ranking - sorts by the actual answer, so completed rides come first. 
#         ideal = group.sort_values("completed_label", ascending=False).head(k)
#         ideal_labels = ideal["completed_label"].values

#         idcg = 0
#         for i, label in enumerate(ideal_labels):
#             idcg += label / np.log2(i + 2)          # perfect possible ranking score  

#         if idcg > 0:
#             ndcg_scores.append(dcg / idcg)          # divides model's score by the perfect ranking score
#                                                     # NDCG is between 0 (terrible) and 1 (perfect)

#     return float(np.mean(ndcg_scores)) if ndcg_scores else 0

def ndcg_at_k(df, score_column, k, min_candidates=1):
    ndcg_scores = []

    for user, group in df.groupby("user"):
        if len(group) < min_candidates:
            continue

        ranked_group = group.sort_values(
            score_column,
            ascending=False
        )

        relevance = ranked_group["completed_label"].to_numpy()
        top_k_relevance = relevance[:k]

        dcg = 0.0
        for i, rel in enumerate(top_k_relevance):
            dcg += rel / np.log2(i + 2)

        ideal_relevance = np.sort(relevance)[::-1][:k]

        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


# def mean_reciprocal_rank(test_meta, score_col):
#     reciprocal_ranks = []

#     for user_id, group in test_meta.groupby("user"):
#         ranked = group.sort_values(score_col, ascending=False).reset_index(drop=True)

#         completed_positions = ranked.index[ranked["completed_label"] == 1].tolist()

#         if completed_positions:
#             first_position = completed_positions[0] + 1
#             reciprocal_ranks.append(1 / first_position)

#     return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0

def mean_reciprocal_rank(test_meta, score_col, positive_users_only=True, group_col="user"):
    reciprocal_ranks = []

    for group_id, group in test_meta.groupby(group_col):
        ranked = group.sort_values(
            score_col,
            ascending=False,
        ).reset_index(drop=True)

        positive_positions = ranked.index[
            ranked["completed_label"] == 1
        ].tolist()

        if positive_positions:
            first_position = positive_positions[0] + 1
            reciprocal_ranks.append(1 / first_position)

        elif not positive_users_only:
            # No positive candidate means the system did not produce a hit.
            reciprocal_ranks.append(0.0)

    return (
        float(np.mean(reciprocal_ranks))
        if reciprocal_ranks else 0.0
    )

# # asks that for users who had at least one completed candidate available, did the model find it?
# def ranking_metrics_positive_users_only(test_meta, score_col, k=5):
#     precision_scores = []
#     hit_rates = []

#     for user_id, group in test_meta.groupby("user"):
#         if len(group) < k:
#             continue

#         if group["completed_label"].sum() == 0:
#             continue

#         top_k = group.sort_values(score_col, ascending=False).head(k)
#         hits = top_k["completed_label"].sum()

#         precision_scores.append(hits / k)
#         hit_rates.append(1 if hits > 0 else 0)

#     return {
#         f"positive_user_precision@{k}": float(np.mean(precision_scores)) if precision_scores else 0,
#         f"positive_user_hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0,
#         "positive_users_evaluated": len(precision_scores)
#     }



# Evaluate

rank_scores = ranker.predict(X_test_processed)

test_meta = test_df[
    [
        "user",
        "matchedUser",
        "completed_label",
        "actualStartDistanceInMeter",
        "rideTimeDifferenceInMinute",
        "matchedUserReviewPoint"
    ]
].copy()                            # creates a smaller table with only these fields, takes only the test portion (20%) of the data

# Baselines: nearest pickup, smallest time difference, highest matched rating, lowest matched cancellation
test_meta["ranker_score"] = rank_scores
test_meta["closest_pickup_score"] = -test_meta["actualStartDistanceInMeter"]
test_meta["smallest_time_difference_score"] = -test_meta["rideTimeDifferenceInMinute"].abs()
test_meta["highest_matched_rating_score"] = test_meta["matchedUserReviewPoint"]


# candidate count analysis - tells you whether the model has successful matches available to rank.
# If many users have no completed candidates, then the model cannot do much for those users.
candidate_counts = test_meta.groupby("user").size()

print("\nCandidate count per user:")
print(candidate_counts.describe())

print("\nUsers with at least 1 candidate:", (candidate_counts >= 1).sum())
print("Users with at least 3 candidates:", (candidate_counts >= 3).sum())
print("Users with at least 5 candidates:", (candidate_counts >= 5).sum())
print("Users with at least 10 candidates:", (candidate_counts >= 10).sum())


positive_counts = test_meta.groupby("user")["completed_label"].sum()

print("\nCompleted candidates per user:")
print(positive_counts.describe())

print("\nShare of users with no completed candidates:")
print((positive_counts == 0).mean())

print("\nShare of users with exactly one completed candidate:")
print((positive_counts == 1).mean())

print("\nShare of users with more than one completed candidate:")
print((positive_counts > 1).mean())


print("\nRanking metrics on All Users:")

# for every k having different number of users 
for k in [1, 3, 5]:
    print(f"\nRanker top {k}:")
    print(ranking_metrics(test_meta, "ranker_score", k=k, min_candidates=k))

    print(f"Closest pickup top {k}:")
    print(ranking_metrics(test_meta, "closest_pickup_score", k=k, min_candidates=k))

    # NDCG function excludes users with no positive candidates, so this is conditional NDCG.
    print(f"Conditional NDCG@{k}")
    print("Ranker:", ndcg_at_k(test_meta, "ranker_score", k=k))
    
    print("Closest pickup:",ndcg_at_k(test_meta, "closest_pickup_score", k=k))

print("\nMean Reciprocal Rank on All Users:")

print("Ranker conditional MRR:", mean_reciprocal_rank(test_meta, "ranker_score", positive_users_only=True))
print("Ranker overall MRR:", mean_reciprocal_rank(test_meta, "ranker_score", positive_users_only=False))
print("Closest pickup conditional MRR:", mean_reciprocal_rank(test_meta, "closest_pickup_score", positive_users_only=True))
print("Closest pickup overall MRR:", mean_reciprocal_rank(test_meta, "closest_pickup_score", positive_users_only=False))


print("\nRanking metrics on a fixed user population:")

# Only evaluate users with at least 5 candidates.
# This keeps the same users for Top-1, Top-3, and Top-5.
COMMON_MIN = 5

fixed_user_ids = candidate_counts[candidate_counts >= COMMON_MIN].index
fixed_test_meta = test_meta[test_meta["user"].isin(fixed_user_ids)].copy()

for k in [1, 3, 5]:
    print(f"\nRanker top {k}:")
    print(ranking_metrics(fixed_test_meta, "ranker_score", k=k, min_candidates=COMMON_MIN))
    print(f"Closest pickup top {k}:")
    print(ranking_metrics(fixed_test_meta, "closest_pickup_score", k=k, min_candidates=COMMON_MIN))

    print(f"Conditional NDCG@{k}")
    print("Ranker:", ndcg_at_k(fixed_test_meta, "ranker_score", k=k, min_candidates=COMMON_MIN))
    print("Closest pickup:", ndcg_at_k(fixed_test_meta, "closest_pickup_score", k=k, min_candidates=COMMON_MIN))


print("\nMean Reciprocal Rank on Fixed Population:")

print("Ranker conditional MRR:", mean_reciprocal_rank(fixed_test_meta, "ranker_score", positive_users_only=True))
print("Ranker overall MRR:", mean_reciprocal_rank(fixed_test_meta, "ranker_score", positive_users_only=False))
print("Closest pickup conditional MRR:", mean_reciprocal_rank(fixed_test_meta, "closest_pickup_score", positive_users_only=True))
print("Closest pickup overall MRR:", mean_reciprocal_rank(fixed_test_meta, "closest_pickup_score", positive_users_only=False))

# for k in [1, 3, 5, 10]:
#     print(f"\nRanker top {k}:")
#     print(ranking_metrics(test_meta, "ranker_score", k=k, min_candidates=k))  # evaluate ranker model

#     print(f"Closest pickup top {k}:")
#     print(ranking_metrics(test_meta, "closest_pickup_score", k=k, min_candidates=k))    # evaluate closest pickup baseline.

#     print(f"NDCG@{k}")
#     print("Ranker:", ndcg_at_k(test_meta, "ranker_score", k=k))
#     print("Closest pickup:", ndcg_at_k(test_meta, "closest_pickup_score", k=k))     # NCDG - whether completed rides are being placed at the top


# print("\nMean Reciprocal Rank:")
# print("Ranker:", mean_reciprocal_rank(test_meta, "ranker_score"))
# print("Closest pickup:", mean_reciprocal_rank(test_meta, "closest_pickup_score"))

# print("Conditional MRR:", mean_reciprocal_rank(test_meta, "ranker_score", positive_users_only=True,),)

# print("Overall MRR:", mean_reciprocal_rank(test_meta, "ranker_score", positive_users_only=False,),)

# print("\nPositive-user-only ranking metrics:")

# for k in [1, 3, 5]:
#     print(f"\nRanker positive users top {k}:")
#     print(ranking_metrics_positive_users_only(test_meta, "ranker_score", k=k))

#     print(f"Closest pickup positive users top {k}:")
#     print(ranking_metrics_positive_users_only(test_meta, "closest_pickup_score", k=k))


# Save ranker and preprocessor

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "preprocessor": preprocessor,
        "ranker": ranker
    },
    "models/ride_completion_ranker.pkl"
)

print("\nRanker saved to models/ride_completion_ranker.pkl")

# # creates a table to compare results more clearly
# results = []
# for k in [1, 3, 5]:
#     ranker_metrics = ranking_metrics(test_meta, "ranker_score", k=k, min_candidates=k)
#     pickup_metrics = ranking_metrics(test_meta, "closest_pickup_score", k=k, min_candidates=k)

#     results.append({
#         "k": k,
#         "ranker_precision": ranker_metrics[f"precision@{k}"],
#         "pickup_precision": pickup_metrics[f"precision@{k}"],
#         "ranker_hit_rate": ranker_metrics[f"hit_rate@{k}"],
#         "pickup_hit_rate": pickup_metrics[f"hit_rate@{k}"],
#         "ranker_ndcg": ndcg_at_k(test_meta, "ranker_score", k=k),
#         "pickup_ndcg": ndcg_at_k(test_meta, "closest_pickup_score", k=k),
#         "users_evaluated": ranker_metrics["users_evaluated"]
#     })

# results_df = pd.DataFrame(results)
# print("\nFinal ranker comparison:")
# print(results_df)

# results_df.to_csv("models/ranker_results.csv", index=False)