

def accuracy(predictions, true):
    right = 0

    for i in range(len(predictions)):
        if predictions[i] == true[i]:
            right += 1

    return right / len(predictions)