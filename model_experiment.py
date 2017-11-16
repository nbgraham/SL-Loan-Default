from analysis import test_f_stars
from data import split

n_fs = 100
f_stars = [i/n_fs for i in range(n_fs)]


def test_model(data, target, attributes, create_model, *grid_search_params):
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    total = 1
    for p in grid_search_params:
        total *= len(p)

    experiment_data = {}
    experiment_data['total'] = total
    experiment_data['run'] = 0
    experiment_data['max_auc'] = 0
    experiment_data['auc_best_params'] = None
    experiment_data['max_acc'] = 0
    experiment_data['acc_best_params'] = None

    loop_and_test_params(experiment_data, training_val_data, training_val_target, attributes, create_model,
                         *grid_search_params)

    print("\n\n", experiment_data)


def loop_and_test_params(experiment_data, training_val_data, training_val_target, attributes, create_model, *grid_search_params):
    if isinstance(grid_search_params[-1], list):
        i = 0
        while not isinstance(grid_search_params[i],list):
            i += 1
        for param in grid_search_params[i]:
            loop_and_test_params(experiment_data, training_val_data, training_val_target, attributes, create_model, *grid_search_params[0:i], param, *grid_search_params[i+1:])
    else:
        experiment_data['run'] += 1
        print("--- Run {}/{}".format(experiment_data['run'], experiment_data['total']))

        auc, acc = run_one(training_val_data, training_val_target, create_model, attributes, *grid_search_params)
        print("AUC: {}".format(auc))

        if auc > experiment_data['max_auc']:
            experiment_data['max_auc'] = auc
            experiment_data['auc_best_params'] = grid_search_params
            print("New best AUC!")

        if auc > experiment_data['max_acc']:
            experiment_data['max_acc'] = auc
            experiment_data['acc_best_params'] = grid_search_params
            print("New best accuracy!")


def run_one(training_val_data, training_val_target, create_model, attributes, *grid_search_params):
    training_data, training_target, val_data, val_target = split(training_val_data, training_val_target)

    clf = create_model(*grid_search_params)
    clf.fit(training_data, training_target, attributes)

    val_preds = clf.predict_prob(val_data)

    auc, max_acc = test_f_stars(val_preds, val_target, f_stars, status_delay=25)

    return auc, max_acc
