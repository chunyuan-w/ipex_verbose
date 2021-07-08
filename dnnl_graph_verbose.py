import argparse
import os

import pandas as pd


header = ["dnnl_verbose", "action", "eng", "name", "impl", "prop", "format", "blank1", "blank2", "shape", "time"]

def preprocess(file):
    with open(file) as f:
        content = f.read().splitlines()
    reorders = []
    for i, line in enumerate(content):
        if line.startswith(("dnnl_graph_verbose", "dnnl_verbose")) and len(line.split(',')) == 11:
            reorder = line.replace("reorder", content[i-1]).split(",")
            assert len(reorder) == 11, "Please check the verbose format of:\nOP that leads to the reorder: %s\nThe reorder verbose: %s" % (content[i-1], line)
            reorders.append(reorder)
    df = pd.DataFrame(reorders, columns=header)
    return df


def main(file_name):
    df = preprocess(file_name)
    df["time"] = df["time"].astype(float)
    df_groupby_name = df.groupby("name").sum().sort_values(by="time", ascending=False)


    print(df_groupby_name)
    # pt = pd.pivot_table(df, index=["name"], values="time", aggfunc="sum")
    
    # print(pt)
    
    # pt["total_time"] = pt.groupby(level=["name"]).transform("sum").loc[:, "time"]
    # print(pt)
    # df.to_csv(file_name + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", default=None, type=str, required=True, help="path to the input onednn log file")
    args = parser.parse_args()

    df = main(args.file_name)
