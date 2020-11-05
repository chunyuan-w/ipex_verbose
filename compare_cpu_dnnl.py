import argparse
import os

import pandas as pd


def preprocess(file, verbose_type):
    verbose_len = {"dnnl": 11, "mkldnn": 9}

    header_dnnl = ["dnnl_verbose", "action", "eng", "name", "impl", "prop", "format", "blank1", "blank2", "problem", "time"]
    header_mkldnn = ["mkldnn_verbose", "action", "name", "impl", "prop", "format", "aux", "problem", "time"]
    header = {"dnnl": header_dnnl, "mkldnn": header_mkldnn}

    with open(file) as f:
        content = f.read().splitlines()
    ops = []
    for i, line in enumerate(content):
        if line.startswith(verbose_type) and len(line.split(',')) == verbose_len[verbose_type]:
            ops.append(line.split(','))
    df = pd.DataFrame(ops, columns=header[verbose_type])
    return df

def sort_df(df, op_name):
    df = df[df["name"] == op_name]
    df["time"] = df["time"].astype(float)

    df.index = df.index.set_names(["index"])
    df = df.reset_index()

    grouped_df = df.groupby(["problem", "format", "impl"]) \
       .agg({'index':'size', 'time':'mean'}) \
       .rename(columns={'index':'count','time':'AVG'}) \
       .reset_index().sort_values("AVG", ascending=False)
    print(grouped_df)
    return grouped_df


def merge_two_file(df1, df2, df1_name, df2_name, save_file_name):
    df_merged = df1.merge(df2, on="problem", suffixes=["_%s" % df1_name, "_%s" % df2_name], how="outer")

    print(df_merged)
    if save_file_name:
        df_merged.to_csv(save_file_name)

if __name__ == "__main__":
    verbose_type_list = ["dnnl", "mkldnn"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu_file_name", default=None, type=str, required=True, help="path to the cpu onednn log file")
    parser.add_argument("-d", "--dpcpp_file_name", default=None, type=str, required=True, help="path to the dpcpp onednn log file")
    parser.add_argument("-o", "--op_name", default="convolution", type=str, help="op name")
    parser.add_argument("-s", "--save_file_name", default=None, type=str, help="path to the save the file")
    args = parser.parse_args()

    df_cpu = preprocess(args.cpu_file_name, "mkldnn")
    sorted_cpu = sort_df(df_cpu, args.op_name)

    df_dpcpp = preprocess(args.dpcpp_file_name, "dnnl")
    sorted_dpcpp = sort_df(df_dpcpp, args.op_name)

    merge_two_file(sorted_cpu, sorted_dpcpp, "cpu", "dpcpp", args.save_file_name)
