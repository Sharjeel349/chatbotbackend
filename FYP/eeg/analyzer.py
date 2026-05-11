def analyze_state(features):
    alpha = features["alpha"]
    beta = features["beta"]

    if beta > 0.7:
        return "STRESSED"
    elif alpha > 0.6:
        return "CALM"
    return "NEUTRAL"