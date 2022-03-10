import argparse

num_inst = 14

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OneDNN Verbose Toolkit')
    parser.add_argument('--file', '-f', nargs='+', action='append', help='[label:]filepath')
    args = parser.parse_args()

    te = float(args.file[0][0])
    bo = float(args.file[1][0]) # bottleneck

    assert te < bo, "first throughput need to be smaller"

    single_instance_perf = te / num_inst
    seconds = 1 / single_instance_perf

    percentage = bo / te - 1

    diff_seconds = seconds * percentage

    microseconds = diff_seconds * 1000000

    print("diff in microseconds: ",microseconds)
