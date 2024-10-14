import sys

if len(sys.argv) != 3:
    print("Usage: python3 evaluate.py true_labels.csv predictions.csv")
    exit(1)

with open(sys.argv[1], 'r') as truth_file, open(sys.argv[2]) as predictions_file:
    truth = truth_file.read().split()
    predictions = predictions_file.read().split()

    assert len(truth) == len(predictions)

    correct_count = 0
    for i in range(len(truth)):
        if truth[i] == predictions[i]:
            correct_count += 1
    accuracy = correct_count / len(truth)

    print(f'Accuracy {accuracy:2.4f}')
