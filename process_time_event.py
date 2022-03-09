import pandas as pd

# file = "/home_local/chunyuan/code/TE-ipex/ipex/memory_time_nanosecond.csv"


raw_log = "/home_local/chunyuan/code/TE-ipex/intel_model_zoo/resnet50_latency_log_bf16_20220309143502_instance_0_cores_0-55.log"
with open(raw_log) as f:
    content = f.read().splitlines()

read_line = False
for i, line in enumerate(content):
    if line.startswith("----"):
        break


rows = []
index_to_start = i + 1
for i, line in enumerate(content):
    if i >= index_to_start:
        if len(line.split('|')) == 2:
            row = line.split("|")
            row[1] = row[1].strip()
            rows.append(row)

print(rows)
df = pd.DataFrame(rows, columns=["event", "TimeStamp(nanoseconds)"])
print(df.head())



# df_summary = pd.DataFrame(columns=["start", "end", "diff"])

# df = pd.read_csv(file)

df["TimeStamp(nanoseconds)"] = df["TimeStamp(nanoseconds)"].astype(int)

df['dTimeStamp'] = df['TimeStamp(nanoseconds)'] - df['TimeStamp(nanoseconds)'].shift(1)

new_df = df.iloc[1::2, :]

# df_summary["start"] = df[::2, :]

# print(df.head())
print(new_df)
print("sum (in microseconds): ", new_df["dTimeStamp"].sum() / 1000)