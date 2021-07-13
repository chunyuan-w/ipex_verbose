import argparse
import os

import pandas as pd

dict_type = {
    "dnnl": {
        "name": "dnnl_verbose",
        "len": 11,
        "header": ["dnnl_verbose", "action", "eng", "name", "impl", "prop", "format", "blank1", "blank2", "shape", "time"]
    },
    "graph": {
        "name": "dnnl_graph_verbose",
        "len": 9,
        "header": ["dnnl_verbose", "name", "eng", "impl", "op", "data", "format", "backend", "time"]
    }
}


def preprocess(file, verbose_type, delimiter):
    with open(file) as f:
        content = f.read().splitlines()
    reorders = []
    if len(delimiter) == 0:
        for i, line in enumerate(content):
            if line.startswith(dict_type[verbose_type]["name"]) and len(line.split(',')) == dict_type[verbose_type]["len"]:
            # if line.startswith(("dnnl_graph_verbose", "dnnl_verbose")) and len(line.split(',')) == 11:
                reorder = line.split(",")
                # assert len(reorder) == 11, "Please check the verbose format of:\nOP that leads to the reorder: %s\nThe reorder verbose: %s" % (content[i-1], line)
                reorders.append(reorder)
    else:
        take = False
        for i, line in enumerate(content):
            if take:
                if line.startswith(dict_type[verbose_type]["name"]) and len(line.split(',')) == dict_type[verbose_type]["len"]:
                # if line.startswith(("dnnl_graph_verbose", "dnnl_verbose")) and len(line.split(',')) == 11:
                    reorder = line.split(",")
                    # assert len(reorder) == 11, "Please check the verbose format of:\nOP that leads to the reorder: %s\nThe reorder verbose: %s" % (content[i-1], line)
                    reorders.append(reorder)                    
            if line == delimiter:
                take = True


    df = pd.DataFrame(reorders, columns=dict_type[verbose_type]["header"])
    return df


def main(file_name, verbose_type, delimiter):
    df = preprocess(file_name, verbose_type, delimiter)
    df["time"] = df["time"].astype(float)
    # df_groupby_name = df.groupby("name").sum().sort_values(by="time", ascending=False)


    df_groupby_name = df.groupby('name')['time'].agg(['sum','count', 'mean'])

    df_groupby_name = df_groupby_name.rename(columns={'sum': 'sum (ms)', 'mean': 'mean (ms)'})
    print(df_groupby_name)
    # df_groupby_name.to_csv(file_name + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", default=None, type=str, required=True, help="path to the input onednn log file")
    parser.add_argument("-t", "--verbose_type", default="dnnl", type=str, choices=["dnnl", "graph"] ,required=True, help="dnnl or graph verbose")
    parser.add_argument("--delimiter", default="", type=str, required=False, help="trim lines at the beginning of verbose output")
    args = parser.parse_args()

    df = main(args.file_name, args.verbose_type, args.delimiter)


# python dnnl_graph_summary.py -f bs1_int8_verbose_0.log -t dnnl --delimiter "begin running..............."
