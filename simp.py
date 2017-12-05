from data import load_loan


def main():
    data, target, attributes = load_loan()

    t_pos = 0
    f_pos = 0

    t_neg = 0
    f_neg = 0

    for i in range(len(data)):
        dt = data[i]
        avg_payment_time = sum(dt[5:10])/5

        pred = 0
        if avg_payment_time >= 0.34:
            pred = 1
        elif avg_payment_time <= -0.33:
            pred = 0

        true = target[i]
        if true == 1:
            if pred == 1:
                t_pos += 1
            else:
                f_neg += 1
        else:
            if pred == 1:
                f_pos += 1
            else:
                t_neg += 1

    print("Accuracy = ", (t_pos + t_neg) / len(data))


if __name__ == "__main__":
    main()