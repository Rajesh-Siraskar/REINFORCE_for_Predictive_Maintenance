import pandas as pd
import glob
import sys

if (len(sys.argv)<=1):
    print('Error! Please pass a path\n')
else:
    PATH = sys.argv[1]
    CONCAT_REPORT = f'{PATH}/Concat_Experiments.csv'
    print(f'PATH: {PATH}')
    EXPT_REPORTS = f'{PATH}/Experiment_Results_*.csv'

    report_files = glob.glob(EXPT_REPORTS)
    print(f'Number of expt. files: {len(report_files)}\n')
    print(report_files)

    dfes = [pd.read_csv(f, header = 0) for f in report_files]
    df_report = pd.concat(dfes)
    df_report.to_csv(CONCAT_REPORT)

    print(f'Concataned expt. file: {CONCAT_REPORT}\n')
