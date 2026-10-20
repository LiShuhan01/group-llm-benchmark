def successful(predictions):
    return [x for x in predictions if x.get("status") == "success"]
