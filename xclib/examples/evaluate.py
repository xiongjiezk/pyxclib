import os
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils


def compute_inv_propensity(train_file, A, B):
    """
        Compute Inverse propensity values
        Values for A/B:
            Wikpedia-500K: 0.5/0.4
            Amazon-670K, Amazon-3M: 0.6/2.6
            Others: 0.55/1.5
    """
    train_labels = data_utils.read_sparse_file(train_file)
    inv_propen = xc_metrics.compute_inv_propesity(train_labels, A, B)
    return inv_propen

def main_wo_psp(targets_file, predictions_file):
    true_labels = data_utils.read_sparse_file(targets_file)
    predicted_labels = data_utils.read_sparse_file(predictions_file)
    acc = xc_metrics.Metrics(true_labels=true_labels,
                             inv_psp=None)
    args = acc.eval(predicted_labels, 512)

    print(f"precision:")
    print([f"{i}, {x}" for i, x in enumerate(list(args[0]))])
    print("recall:")
    print([f"{i}, {x}" for i, x in enumerate(list(args[2]))])
    dir_path = os.path.dirname(os.path.realpath(predictions_file))
    with open(f"{dir_path}/metric.csv", "w") as f:
        f.write("idx,precision,ndcg,recall\n")
        for i in range(512):
            f.write(f"{i},{args[0][i]},{args[1][i]},{args[2][i]}\n")


def main(targets_file, train_file, predictions_file, A, B):
    """
        Args:
            targets_file: test labels
            train_file: train labels (to compute prop)
            prediction_file: predicted labels
            A: int: to compute propensity
            B: int: to compute propensity
    """
    true_labels = data_utils.read_sparse_file(targets_file)
    predicted_labels = data_utils.read_sparse_file(predictions_file)
    inv_psp = compute_inv_propensity(train_file, A, B)
    acc = xc_metrics.Metrics(true_labels=true_labels,
                             inv_psp=inv_psp)
    args = acc.eval(predicted_labels, 5)
    print(xc_metrics.format(*args))


# if __name__ == '__main__':
#     train_file = sys.argv[1]
#     targets_file = sys.argv[2]
#     predictions_file = sys.argv[3]
#     A = float(sys.argv[4])
#     B = float(sys.argv[5])
#     main(targets_file, train_file, predictions_file, A, B)

if __name__ == '__main__':
    targets_file = sys.argv[1]
    predictions_file = sys.argv[2]
    main_wo_psp(targets_file, predictions_file)