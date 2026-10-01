import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "models/ride_completion_ranker.pkl"
DATA_PATH = "data/ride_requests.csv"


model = joblib.load(MODEL_PATH)
preprocessor = model["preprocessor"]
ranker = model["ranker"]
ride_requests = pd.read_csv(DATA_PATH)


def safe_rate(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def add_features(df):
    df = df.copy()

    df["userCompletionRate"] = safe_rate(df["userCompletedCount"], df["userRequestedCount"])
    df["userCancellationRate"] = safe_rate(df["userCancelledCount"], df["userRequestedCount"])
    df["userAcceptanceRate"] = safe_rate(df["userAcceptedCount"], df["userRequestedCount"])
    df["userDeclineRate"] = safe_rate(df["userDeclinedCount"], df["userRequestedCount"])

    df["matchedUserCompletionRate"] = safe_rate(df["matchedUserCompletedCount"], df["matchedUserRequestedCount"])
    df["matchedUserCancellationRate"] = safe_rate(df["matchedUserCancelledCount"], df["matchedUserRequestedCount"])
    df["matchedUserAcceptanceRate"] = safe_rate(df["matchedUserAcceptedCount"], df["matchedUserRequestedCount"])
    df["matchedUserDeclineRate"] = safe_rate(df["matchedUserDeclinedCount"], df["matchedUserRequestedCount"])

    df["pickupDistanceKm"] = df["actualStartDistanceInMeter"] / 1000
    df["dropDistanceKm"] = df["actualEndDistanceInMeter"] / 1000

    df["airPickupDistanceKm"] = df["airStartDistanceInMeter"] / 1000
    df["airDropDistanceKm"] = df["airEndDistanceInMeter"] / 1000

    df["tripDistanceDifferenceKm"] = (df["userTripDistanceMeter"] - df["matchedUserTripDistanceMeter"]).abs() / 1000

    df["rideTimeDifferenceAbs"] = df["rideTimeDifferenceInMinute"].abs()

    df["sameGender"] = (df["userGender"] == df["matchedUserGender"]).astype(int)

    df["sameCompanyName"] = (df["userCompany"] == df["matchedUserCompany"]).astype(int)

    df["bothSubscribed"] = ((df["userSubscriptionActive"] == 1) & (df["matchedUserSubscriptionActive"] == 1)).astype(int)

    df["reviewPointDifference"] = (df["userReviewPoint"] - df["matchedUserReviewPoint"]).abs()

    df["completedRideCountDifference"] = (df["userTripTotalCompletedRides"] - df["matchedUserTotalCompletedRides"]).abs()

    df["totalRouteInconvenienceKm"] = (df["actualStartDistanceInMeter"] + df["actualEndDistanceInMeter"]) / 1000

    df["pickupDistanceToTripRatio"] = (df["actualStartDistanceInMeter"] / df["userTripDistanceMeter"].replace(0, np.nan))
    df["dropDistanceToTripRatio"] = (df["actualEndDistanceInMeter"] / df["userTripDistanceMeter"].replace(0, np.nan))

    df["completionRateDifference"] = (df["userCompletionRate"] - df["matchedUserCompletionRate"]).abs()
    df["cancellationRateDifference"] = (df["userCancellationRate"] - df["matchedUserCancellationRate"]).abs()

    df["pairCompletionStrength"] = (df["userCompletionRate"] * df["matchedUserCompletionRate"])
    df["pairCancellationRisk"] = (df["userCancellationRate"] + df["matchedUserCancellationRate"])

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

    df["ratingRankShareForUser"] = (df["matchedRatingRankForUser"] / df["candidateCountForUser"])  # ranks candidates by rating for a particular user. Value near 0 is highest rated options for this user, 1 is lowest rated optionss

    # Reliability-rank features
    # df["matchedCompletionRateRankForUser"] = df.groupby("user")["matchedUserCompletionRate"].rank(method="average", ascending=False)   # ranks each candidate by their completion rate compared to the other candidates for a particular user. higher completion rate = better rank

    # df["matchedCancellationRateRankForUser"] = df.groupby("user")["matchedUserCancellationRate"].rank(method="average", ascending=True)  # ranks candidates by cancellation rate. rank 1 = lowest cancellation rate

    # df["completionRateRankShareForUser"] = (df["matchedCompletionRateRankForUser"] / df["candidateCountForUser"])  # normalizes matched completion rate rank for user

    # df["cancellationRateRankShareForUser"] = (df["matchedCancellationRateRankForUser"] / df["candidateCountForUser"])   # normalizes matched cancellation rate rank for user
    
    # # Interaction features
    # df["closePickupAndCloseTime"] = ((df["pickupRankShareForUser"] <= 0.33) & (df["timeRankShareForUser"] <= 0.33)).astype(int)   # checks whether a candidate is in the top third for both pickup distance and time match. If yes: closePickupAndCloseTime = 1, otherwise 0

    # df["closePickupAndHighRating"] = ((df["pickupRankShareForUser"] <= 0.33) & (df["ratingRankShareForUser"] <= 0.33)).astype(int)    # checks whether a candidate is in the top third for both pickup distance and rating. If yes: closePickupAndHighRating = 1, otherwise 0

    # df["goodTimeAndHighRating"] = ((df["timeRankShareForUser"] <= 0.33) & (df["ratingRankShareForUser"] <= 0.33)).astype(int)   # checks whether a candidate is in the top third for both time match and rating. If yes: goodTimeAndHighRating = 1, otherwise 0

    # df["strongCandidateOverall"] = ((df["pickupRankShareForUser"] <= 0.33) & (df["timeRankShareForUser"] <= 0.33) & (df["ratingRankShareForUser"] <= 0.33)).astype(int)   # checks whether a candidate is in the top third for pickup distance, time match, and rating. If yes: strongCandidateOverall = 1, otherwise 0

    return df

def recommend_for_user(user_id, top_n=10):   # user id for which we want recommendations and how many we want 
    candidates = ride_requests[ride_requests["user"] == user_id].copy()   # all matched users for this specific user id 

    if candidates.empty:
        print(f"No candidates found for user {user_id}")
        return pd.DataFrame() 

    candidates = add_features(candidates)   # applies feature engineering to the rows left 

    drop_cols = [
        "completed_label",
        "objectId",
        "user",
        "matchedUser"
    ]                     # columns we don't want to feed into the model 

    X_candidates = candidates.drop(
        columns=[col for col in drop_cols if col in candidates.columns])    # creates the input table for prediction without the dropped columns 

    X_candidates_processed = preprocessor.transform(X_candidates)
    candidates["ranking_score"] = ranker.predict(X_candidates_processed)   # return probability of each ride with every matched user completing 

    recommendations = candidates.sort_values(
        "ranking_score", ascending=False).head(top_n)   # sorts top n rows with highest probability first

    return recommendations[
        [
            "user",
            "matchedUser",
            "ranking_score",
            "actualStartDistanceInMeter",
            "actualEndDistanceInMeter",
            "rideTimeDifferenceInMinute",
            "geoCity",
            "transportMode",
            "hasSameCompany",
            "isFriend",
            "userReviewPoint",
            "matchedUserReviewPoint",
            "completed_label"
        ]]                            # only returns the useful columns instead of the whole dataframe


if __name__ == "__main__":    # only run the code below if recommend.py is being run directly
    user_id = "U001165"

    recommendations = recommend_for_user(user_id, top_n=10)

    print("\nTop recommendations:")
    print(recommendations)
