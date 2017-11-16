from analysis import test_f_stars
from data import split

n_fs = 100
f_stars = [i/n_fs for i in range(n_fs)]


def test_model(data, target, attributes, create_model, *grid_search_params):
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    total = 1
    for p in grid_search_params:
        total *= len(p)

    ints = {}
    ints['total'] = total
    ints['run'] = 0
    ints['max_auc'] = 0
    loop_and_test_params(ints, training_val_data, training_val_target, attributes, create_model,
                         *grid_search_params)

def loop_and_test_params(ints, training_val_data, training_val_target, attributes, create_model, *grid_search_params):
    if isinstance(grid_search_params[-1], list):
        i = 0
        while not isinstance(grid_search_params[i],list):
            i += 1
        for param in grid_search_params[i]:
            loop_and_test_params(ints, training_val_data, training_val_target, attributes, create_model, *grid_search_params[0:i], param, *grid_search_params[i+1:])
    else:
        ints['run'] += 1
        print("Run {}/{}".format(ints['run'], ints['total']))

        try:
            auc = run_one(training_val_data, training_val_target, create_model, attributes, *grid_search_params)
        except TimeoutError as err:
            print(err)

        if auc > ints['max_auc']:
            ints['max_auc'] = auc
            print("Max AUC: {} ".format(auc))


def run_one(training_val_data, training_val_target, create_model, attributes, *grid_search_params):
    training_data, training_target, val_data, val_target = split(training_val_data, training_val_target)

    clf = create_model(*grid_search_params)
    clf.fit(training_data, training_target, attributes)

    val_preds = clf.predict_prob(val_data)

    auc = test_f_stars(val_preds, val_target, f_stars, status_delay=25)

    return auc
