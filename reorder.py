import argparse
import os

import pandas as pd


header = ["dnnl_verbose", "action", "eng", "name", "impl", "prop", "format", "blank1", "blank2", "shape", "time"]


def group_and_sort(df, group_by, top_k, first_line, format_to_exclude, output_dir):
    if len(df) == 0:
        print("cannot find any OP that causes the reorder, you could try to add #define _DEBUG in DevOPs.cpp")
        return
    
    df["time"] = df["time"].astype(float)
    
    # remove format that starts with format_to_exclude
    if format_to_exclude:
        df = df[~df["format"].str.startswith(tuple(format_to_exclude))]
    
    print("*" * 50)
    if format_to_exclude:
        print("excluded format that starts with: %s" % (", ".join(format_to_exclude)))
    if top_k != -1:
        print("only show the top %d result in each group" % top_k)
    print("*" * 50)
    
    # op level time
    # df_groupby_name = df.groupby(group_by).sum().sort_values(by="time", ascending=False).reset_index()
    df_groupby_name = df.groupby(group_by).sum().sort_values(by="time", ascending=False)
    df_groupby_name = df_groupby_name.rename(columns={"time": "total_time"})
    if first_line != -1:
        print("*" * 5 + " Only print first %d lines in table " % first_line + "*" * 5)
        print(df_groupby_name.head(first_line))
    else:
        print(df_groupby_name)
    print()

    # op and shape level time
    pt = pd.pivot_table(df, index=["name", "shape"], values="time", aggfunc="sum")
    pt["total_time"] = pt.groupby(level=["name"]).transform("sum").loc[:, "time"]
    # TODO top_k == -1
    if top_k == -1:
        pt = pt.sort_values(["total_time", "time"], ascending=[False, False]).groupby("name").head(len(pt))
    else:
        pt = pt.sort_values(["total_time", "time"], ascending=[False, False]).groupby("name").head(top_k)
    if first_line != -1:
        print("*" * 5 + " Only print first %d lines in table " % first_line + "*" * 5)
        print(pt.head(first_line))
    else:
        print(pt)

    if output_dir:
        df_groupby_name.to_csv(os.path.join(output_dir, "op_level.csv"), index=False)
        pt.to_csv(os.path.join(output_dir, "op_shape_level.csv"))


def preprocess(file, op_to_include):
    with open(file) as f:
        content = f.read().splitlines()
    reorders = []
    for i, line in enumerate(content):
        if i == 0:
            continue
        if line.startswith("dnnl_verbose") and len(line.split(',')) == 11:
            if line.split(',')[3] == "reorder":
                if content[i-1].startswith(tuple(op_to_include)):
                    reorders.append(line.replace("reorder", content[i-1]).split(","))
    df = pd.DataFrame(reorders, columns=header)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", default=None, type=str, required=True, help="path to the input onednn log file")
    parser.add_argument("-o", "--output_dir", default=None, type=str, help="directory to save the output csv file")
    parser.add_argument("-g", "--group_by", nargs='+', choices=header, required=True, help="column names to groupby to calculate the total time")
    parser.add_argument("-p", "--op_to_include", nargs='+', choices=["AtenIpexCPUDefault", "AtenIpexCPUDev"], default="AtenIpexCPUDefault", required=True, help="OP to include")
    parser.add_argument("-e", "--exclude", nargs='+', help="format starts with the given strings will be excluded")
    parser.add_argument("-t", "--top_k", type=int, default=-1, help="only show the top k result within each group, if -1, show all the result")
    parser.add_argument("-l", "--first_line", type=int, default=-1, help="only print the first several lines in each table")
    args = parser.parse_args()

    df = preprocess(args.file_name, args.op_to_include)
    group_and_sort(df, args.group_by, args.top_k, args.first_line, args.exclude, args.output_dir)

# DevOPs.cpp:
# #define _DEBUG

# python reorder.py -f ../transformer/log/transFB.log -o ../transformer/log -g name -t 10 -e src_f32 -l 10 -p AtenIpexCPUDefault
