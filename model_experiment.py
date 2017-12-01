import numpy as np

from analysis import test_f_stars, rel
from data import split

n_fs = 100
f_stars = [i/n_fs for i in range(n_fs)]


def test_model(data, target, attributes, create_model, *grid_search_params):
    _test_model(data, target, attributes, create_model, None, *grid_search_params)


def _test_model(data, target, attributes, create_model, save, *grid_search_params):
    training_val_data, training_val_target, test_data, test_target = split(data, target)
    total = 1
    shape = ()
    for p in grid_search_params:
        total *= len(p)
        shape += (len(p),)

    experiment_data = {}
    experiment_data['total'] = total
    experiment_data['run'] = 0
    experiment_data['max_auc'] = 0
    experiment_data['auc_best_params'] = None
    experiment_data['max_acc'] = 0
    experiment_data['acc_best_params'] = None
    experiment_data['auc_grid'] = np.empty(shape)
    experiment_data['acc_grid'] = np.empty(shape)
    experiment_data['rel_grid'] = np.empty(shape)

    loop_and_test_params(experiment_data, training_val_data, training_val_target, attributes, create_model, save, [0]*len(grid_search_params),
                         *grid_search_params)

    return experiment_data


def loop_and_test_params(experiment_data, training_val_data, training_val_target, attributes, create_model, save, indices, *grid_search_params):
    if isinstance(grid_search_params[-1], list):
        i = 0
        while not isinstance(grid_search_params[i],list):
            i += 1
        for param_i in range(len(grid_search_params[i])):
            param = grid_search_params[i][param_i]
            new_indices = indices[:]
            new_indices[i] = param_i
            loop_and_test_params(experiment_data, training_val_data, training_val_target, attributes, create_model, save, new_indices, *grid_search_params[0:i], param, *grid_search_params[i+1:])
    else:
        experiment_data['run'] += 1
        print("--- Run {}/{}".format(experiment_data['run'], experiment_data['total']))

        if experiment_data['run'] % 10 == 0 and save is not None:
            print("Saving current results")
            save(experiment_data['auc_grid'], experiment_data['rel_grid'])

        auc, acc, rel = run_one(training_val_data, training_val_target, create_model, attributes, *grid_search_params)

        experiment_data['auc_grid'][tuple(indices)] = auc
        experiment_data['acc_grid'][tuple(indices)] = acc
        experiment_data['rel_grid'][tuple(indices)] = rel
        print("AUC: {} REL: {} ACC:{}".format(auc, rel, acc))

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

    auc, max_acc,_,_ = test_f_stars(val_preds, val_target, f_stars, status_delay=25)

    r = rel(val_preds[:,1], val_target)
    return auc, max_acc, r
