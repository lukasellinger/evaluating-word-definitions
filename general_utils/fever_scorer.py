# Adapted from
# https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py
#
# Additional license and copyright information for this source code are available at:
# https://github.com/sheffieldnlp/fever-scorer/blob/master/LICENSE
"""Official scorer of the FEVER task."""


def check_predicted_evidence_format(instance):
    if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
        assert all(isinstance(prediction, list)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(len(prediction) == 2
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(isinstance(prediction[0], str)
                    for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"

        assert all(isinstance(prediction[1], int)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"


def is_correct_label(instance, use_gold_labels=False):
    label = instance["label"].upper()
    predicted_label = instance["predicted_label"].upper()

    # Check if the predicted label matches the actual label, or if using gold labels is enabled
    if label == predicted_label or use_gold_labels:
        return True

    # Special case: treat 'NOT_ENOUGH_INFO' as correct if predicted as 'NOT_SUPPORTED'
    if label == 'NOT_ENOUGH_INFO' and predicted_label == 'NOT_SUPPORTED':
        return True

    return False


def is_strictly_correct(instance, max_evidence=None, use_gold_labels=False):
    # Strict evidence matching is only for NEI class
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT_ENOUGH_INFO" and instance['predicted_evidence'] and is_correct_label(instance, use_gold_labels):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is None:
            max_evidence = len(instance["predicted_evidence"])

        for evidence_group in instance["evidence"]:
            # Filter out the annotation ids. We just want the evidence page and line number
            actual_sentences = [[e[2], e[3]] for e in evidence_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                return True

    # If the class is NEI or our topic modeling set it to not supported, we don't score the evidence retrieval component
    elif (instance["label"].upper() == "NOT_ENOUGH_INFO" or not instance['predicted_evidence']) and is_correct_label(instance, use_gold_labels):
        return True

    return False


def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT_ENOUGH_INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0


def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT_ENOUGH_INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


def fever_score(predictions, actual=None, max_evidence=5, use_gold_labels=False):
    correct = 0
    strict = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for idx, instance in enumerate(predictions):
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

        # If it's a blind test set, we need to copy in the values from the actual data
        if 'evidence' not in instance or 'label' not in instance:
            assert actual is not None, 'in blind evaluation mode, actual data must be provided'
            assert len(actual) == len(predictions), 'actual data and predicted data length must match'
            assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
            instance['evidence'] = actual[idx]['evidence']
            instance['label'] = actual[idx]['label']

        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        if is_correct_label(instance, use_gold_labels):
            correct += 1.0

            if is_strictly_correct(instance, max_evidence, use_gold_labels):
                strict += 1.0

        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    total = len(predictions)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    return strict_score, acc_score, pr, rec, f1


if __name__ == "__main__":
    instance1 = {"label": "REFUTES", "predicted_label": "REFUTES",
                 "predicted_evidence": [  # is not strictly correct - missing (page2,2)
                     ["page1", 1]  # page name, line number
                 ],
                 "evidence":
                     [
                         [
                             [None, None, "page1", 1],
                             # [(ignored) annotation job, (ignored) internal id, page name, line number]
                             [None, None, "page2", 2],
                         ]
                     ]
                 }

    instance2 = {"label": "REFUTES", "predicted_label": "REFUTES", "predicted_evidence": [
        ["page1", 1],
        ["page2", 2],
        ["page3", 3]
    ],
                 "evidence":
                     [
                         [
                             [None, None, "page1", 1],
                             [None, None, "page2", 2],
                         ]
                     ]
                 }

    predictions = [instance1, instance2]
    strict_score, label_accuracy, precision, recall, f1 = fever_score(predictions)

    print(strict_score)  # 0.5
    print(label_accuracy)  # 1.0
    print(precision)  # 0.833 (first example scores 1, second example scores 2/3)
    print(recall)  # 0.5 (first example scores 0, second example scores 1)
    print(f1)  # 0.625
