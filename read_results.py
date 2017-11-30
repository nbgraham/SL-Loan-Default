import numpy as np
from matplotlib import pyplot as plt

def read(filename='random_forest_results.txt'):
    with open(filename) as f:
        read_data = f.read()
        lines = read_data.split('\n')

        aucs = []
        row = []

        depths = []
        n_ests = []
        new_est = True

        last_depth = -1
        run = 0
        for line in lines:
            if line[4:7] == "Run":
                run_num = line.split('/')[0].split(' ')[-1]
                run = int(run_num)
            elif line[:8] == "Creating":
                splits = line.split(";")
                depth = int(splits[0].split('=')[-1])
                scoring = splits[1].split(' ')[-1]
                n_est = int(splits[2].split('=')[-1])

                if new_est:
                    new_est = not n_est in n_ests
                    if new_est:
                        n_ests.append(n_est)
            elif line[:3] == "AUC":
                auc = float(line[4:])

                if last_depth == -1:
                    depths.append(depth)
                    row.append(auc)
                elif last_depth == depth:
                    row.append(auc)
                else:
                    depths.append(depth)
                    aucs.append(row)
                    row = [auc]
                last_depth = depth

        auc_grid = np.array(aucs)

        plt.imshow(auc_grid)
        plt.colorbar()
        plt.title("Gini AUC for RF")
        plt.xlabel("Number of trees")
        plt.xticks(range(len(n_ests)), n_ests)
        plt.ylabel("Max depth")
        plt.yticks(range(len(depths)), depths)
        plt.show()


if __name__ == "__main__":
    read()