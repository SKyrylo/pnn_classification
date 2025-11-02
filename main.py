from classes.pnn import PNN
import pandas as pd
import numpy as np


def get_accuracy(preds: list[str], gt: list[str]):
    bools = [pred == ground for pred, ground in zip(preds, gt)]
    return sum(bools) / len(bools)


def main():
    print("===== PNN for classification of samples with N features =====")

    # Training data for testing purposes
    df = pd.DataFrame(
        data=[[1,2, 'A'], [2,1, 'A'], [10,20,'B'], [20,10,'B']],
        columns=['x', 'x2', 'target']
    )

    # ----- Initialize PNN model (trained when initialized) -----
    pnn = PNN(df)

    # ----- Validate model to see accuracy -----
    preds, gt = [], df.target.to_list()
    print("\n----- Validate model on training dataset -----")

    for i in range(len(df)):
        sample = np.array(df.iloc[i].drop('target').values)
        pred = pnn(sample)
        preds.append(pred)
        print(f"\t~ Sample - {sample}, Prediction - {pred}, True - {df.iloc[i].target}")
    print(f"Overall Accuracy: {get_accuracy(preds, gt)}")

    # ----- Test model on unseen data -----
    test_df = pd.DataFrame(
        data=[[3,4,'A'], [8,9,'A'], [16,11,'B'], [25,30,'B'], [0,0,'A']],
        columns=['x1','x2','target']
    )

    preds, gt = [], test_df.target.to_list()
    print("\n----- Test model on unseen data -----")

    for i in range(len(test_df)):
        sample = np.array(test_df.iloc[i].drop('target').values)
        pred = pnn(sample)
        preds.append(pred)
        print(f"\t~ Sample - {sample}, Prediction - {pred}, True - {test_df.iloc[i].target}")
    print(f"Overall Accuracy: {get_accuracy(preds, gt)}")


if __name__ == "__main__":
    main()
