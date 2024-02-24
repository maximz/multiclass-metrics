import numpy as np
import pandas as pd
import pytest
import multiclass_metrics
import sklearn.metrics


def test_roc_auc_score_with_missing_labels_in_yscore():
    # confirm main functionality: handle missing classes automatically
    assert (
        multiclass_metrics.roc_auc_score(
            y_true=["Healthy", "Ebola", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.33, 0.33, 0.34],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        == 0.875
        # and confirm that adding missing classes as columns of 0s does not change ROC score
        == sklearn.metrics.roc_auc_score(
            y_true=["Healthy", "Ebola", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.0, 0.1, 0.8],
                    [0.33, 0.0, 0.33, 0.34],
                    [0.1, 0.0, 0.8, 0.1],
                    [0.05, 0.0, 0.05, 0.9],
                    [0.8, 0.0, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "Ebola", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        # and confirm that this is different from removing the entries for the missing classes
        != sklearn.metrics.roc_auc_score(
            y_true=["Healthy", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
    )

    # pass through if no missing classes
    assert (
        multiclass_metrics.roc_auc_score(
            y_true=["Healthy", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        == sklearn.metrics.roc_auc_score(
            y_true=["Healthy", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        == 1.0
    )

    # further confirmation that adding columns of 0 probabilities (for missing classes) does not change ROC AUC
    y_true = np.array([0, 0, 1, 2])
    y_scores = np.array(
        [[0.1, 0.9, 0.0], [0.3, 0.6, 0.1], [0.35, 0.6, 0.05], [0.8, 0.2, 0.0]]
    )
    y_scores2 = np.array(
        [
            [0.1, 0.9, 0.0, 0.0],
            [0.3, 0.6, 0.1, 0.0],
            [0.35, 0.6, 0.05, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
    )
    assert (
        sklearn.metrics.roc_auc_score(
            y_true, y_scores, multi_class="ovo", labels=[0, 1, 2], average="macro"
        )
        == sklearn.metrics.roc_auc_score(
            y_true, y_scores2, multi_class="ovo", labels=[0, 1, 2, 3], average="macro"
        )
        == 0.25
    )
    assert (
        sklearn.metrics.roc_auc_score(
            y_true, y_scores, multi_class="ovo", labels=[0, 1, 2], average="weighted"
        )
        == sklearn.metrics.roc_auc_score(
            y_true,
            y_scores2,
            multi_class="ovo",
            labels=[0, 1, 2, 3],
            average="weighted",
        )
        == 0.21875
    )


def test_roc_auc_with_missing_labels_must_be_multiclass_ovo_or_ovr_mode():
    y_true = ["Healthy", "Ebola", "HIV", "Healthy", "Covid"]
    labels = ["Covid", "HIV", "Healthy"]
    y_score = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.33, 0.33, 0.34],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
        ]
    )
    for multi_class in ["ovr", "ovo"]:
        # no error:
        multiclass_metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            labels=labels,
            multi_class=multi_class,
        )

    # error:
    with pytest.raises(ValueError, match="Only OvO and OvR multiclass are supported"):
        multiclass_metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            labels=labels,
            multi_class="something else",
        )


def test_roc_auc_score_doesnt_have_to_sum_to_one():
    assert multiclass_metrics.roc_auc_score(
        y_true=["Covid", "HIV", "Healthy"],
        y_score=np.array(
            [
                [0.1, 0.1, 0.8],
                [0.33, 0.33, 0.34],
                [0.1, 0.8, 0.1],
            ]
        ),
        labels=["Covid", "HIV", "Healthy"],
        multi_class="ovo",
        average="macro",
    ) == multiclass_metrics.roc_auc_score(
        y_true=["Covid", "HIV", "Healthy"],
        y_score=np.array(
            [
                [0.2, 0.1, 0.8],
                [0.66, 0.33, 0.34],
                [0.2, 0.8, 0.1],
            ]
        ),
        labels=["Covid", "HIV", "Healthy"],
        multi_class="ovo",
        average="macro",
    )


def test_roc_auc_score_with_missing_labels_in_ytest():
    # confirm main functionality: handle missing classes automatically
    # also tests that roc auc score does not have to sum to 1
    assert np.allclose(
        [
            multiclass_metrics.roc_auc_score(
                y_true=["Covid", "Covid", "Covid", "Healthy", "Healthy"],
                y_score=np.array(
                    [
                        [0.1, 0.1, 0.8],
                        [0.4, 0.5, 0.1],
                        [0.8, 0.1, 0.1],
                        [0.33, 0.33, 0.34],
                        [0.2, 0.2, 0.6],
                    ]
                ),
                labels=["Covid", "HIV", "Healthy"],
                multi_class="ovo",
                average="macro",
            ),
            # and confirm that removing missing classes does not change ROC score (though sklearn requires renorm to sum to 1)
            sklearn.metrics.roc_auc_score(
                y_true=["Covid", "Covid", "Covid", "Healthy", "Healthy"],
                y_score=np.array(
                    [
                        [0.2, 0.8],
                        [0.8, 0.2],
                        [0.8, 0.2],
                        [0.49, 0.51],
                        [0.25, 0.75],
                    ]
                )[:, 1],
                labels=["Covid", "Healthy"],
                multi_class="ovo",
                average="macro",
            ),
        ],
        2 / 3,
    )


def test_roc_auc_score_with_so_many_missing_labels_that_only_one_label_is_left():
    # 3 initial categorical labels
    # but all y_true belongs to a single label
    y_true = pd.Series(
        ["surviving_class", "surviving_class"],
        dtype=pd.CategoricalDtype(
            categories=["surviving_class", "other_class", "dropped_class"]
        ),
    )
    # classifier only had 2 of the 3 initial classes
    y_score = np.array([[0.6, 0.4], [0.5, 0.5]])
    labels = np.array(["surviving_class", "dropped_class"])

    y_score_modified, labels_modified = multiclass_metrics._inject_missing_labels(
        y_true=y_true,
        y_score=y_score,
        labels=labels,
    )
    assert np.array_equal(y_score_modified, np.array([[0.6], [0.5]]))
    assert np.array_equal(labels_modified, ["surviving_class"])

    for average in ["weighted", "macro"]:
        with pytest.raises(
            ValueError,
            match="Only one class present in y_true. Probability-based score is not defined in that case.",
        ):
            assert (
                multiclass_metrics.roc_auc_score(
                    y_true=y_true,
                    y_score=y_score,
                    labels=labels,
                    average=average,
                )
                == 0.5
            )


def test_auprc():
    ## Generate sample data: "muddied final column" from test_adjust_model_decision_thresholds:

    # first, clear diagonal
    # how many entries per class are clear diagonal
    n_diagonal_clear = 100
    labels = np.array([0, 1, 2, 3])
    # make labels categorical
    labels = np.array(["class" + str(i) for i in labels])

    clear_diagonal_probas = np.vstack(
        [
            np.tile([0.7, 0.1, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.7, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.7, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.1, 0.7], n_diagonal_clear).reshape(-1, 4),
        ]
    )
    clear_diagonal_trues = np.hstack(
        [np.tile([lbl], n_diagonal_clear) for lbl in labels]
    )

    # now, muddy up the final column:
    # all predictions consistent, but ground truth is a toss up
    n_muddy = 100
    muddy_final_row_probas = np.tile([0.1, 0.1, 0.1, 0.7], n_muddy * 4).reshape(-1, 4)
    muddy_final_row_trues = np.hstack([np.tile([lbl], n_muddy) for lbl in labels])

    y_score = np.vstack([clear_diagonal_probas, muddy_final_row_probas])
    y_true = np.hstack([clear_diagonal_trues, muddy_final_row_trues])

    # Sanity check accuracy score
    y_score_argmax = labels[y_score.argmax(axis=1)]
    assert sklearn.metrics.accuracy_score(y_true, y_score_argmax) == 0.625

    # make sure same result even if labels come in different order
    labels_reordered = np.array(["class3", "class0", "class1", "class2"])
    y_score_reordered = pd.DataFrame(y_score, columns=labels)[labels_reordered].values

    assert np.allclose(
        [
            multiclass_metrics.auprc(y_true, y_score),
            multiclass_metrics.auprc(y_true, y_score, labels=labels),
            multiclass_metrics.auprc(
                y_true, y_score_reordered, labels=labels_reordered
            ),
        ],
        0.72916,
    )

    assert not np.allclose(
        [
            multiclass_metrics.auprc(
                y_true, y_score_reordered
            ),  # no label order provided, so the assumed one will be wrong
            multiclass_metrics.auprc(
                y_true, y_score, labels=labels_reordered
            ),  # wrong label order provided
            multiclass_metrics.auprc(
                y_true, y_score_reordered, labels=labels
            ),  # wrong label order provided
        ],
        0.72916,
    )

    # make sure it supports binary too, whether y_score is provided as 2d or 1d array (sklearn style)
    chosen_labels = labels[:2]
    y_true_subselect = pd.Series(y_true)
    subselect_mask = y_true_subselect.isin(chosen_labels)
    y_true_subselect = y_true_subselect[subselect_mask].values
    y_score_subselect = pd.DataFrame(y_score, columns=labels)
    y_score_subselect = y_score_subselect[chosen_labels].loc[subselect_mask].values

    assert np.allclose(
        [
            multiclass_metrics.auprc(y_true_subselect, y_score_subselect),
            multiclass_metrics.auprc(
                y_true_subselect, y_score_subselect, labels=chosen_labels
            ),
            multiclass_metrics.auprc(y_true_subselect, y_score_subselect[:, 1]),
            multiclass_metrics.auprc(
                y_true_subselect, y_score_subselect[:, 1], labels=chosen_labels
            ),
            sklearn.metrics.average_precision_score(
                y_true_subselect, y_score_subselect[:, 1], pos_label=chosen_labels[1]
            ),
            sklearn.metrics.average_precision_score(
                y_true_subselect, y_score_subselect[:, 0], pos_label=chosen_labels[0]
            ),
        ],
        0.75,
    )
