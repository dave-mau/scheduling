import subprocess
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import shutil
import math
import argparse
import mysql.connector

class MySQLServer:
    def __init__(self, user="root", password=""):
        self.user = user
        self.password = password
        self.process = None

    def get_endpoint(self):
        return f"mysql+mysqlconnector://{self.user}@localhost"

    def get_db_endpoint(self, db_name):
        return f"{self.get_endpoint()}/{db_name}"

    # Check if MySQL server is running
    def check_server_running(self):
        conn = mysql.connector.connect(host="localhost", user=self.user, password=self.password)
        conn.close()

    def database_exists(self, db_name):
        conn = mysql.connector.connect(host="localhost", user=self.user, password=self.password)
        with conn.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            for db in cursor.fetchall():
                if db[0] == db_name:
                    return True
        return False

    def create_database(self, db_name):
        conn = mysql.connector.connect(host="localhost", user=self.user, password=self.password)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        conn.close()

    def drop_database(self, db_name):
        conn = mysql.connector.connect(host="localhost", user=self.user, password=self.password)
        with conn.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        conn.close()

def run_training(system_config_path: Path, opt_log_path: Path, log_path: Path, db_path: str, study_name: str, num_procs: int):
    # Construct command
    cmd = [
        "python3",
        "train.py",
        "--algo",
        "ppo",
        "--env",
        "ParsedHierarchicalSystem-v0",
        "--env-kwargs",
        f'system_config_file:"{str(system_config_path)}"',
        "max_time_steps:6000", # 6_000 steps (10ms per step) = 1 minute
        "render_mode:None",
        "--eval-env-kwargs",
        f'system_config_file:"{str(system_config_path)}"',
        "max_time_steps:6000", # 6_000 steps (10ms per step) = 1 minute
        "render_mode:None",
        "--optimization-log-path",
        f"{str(opt_log_path)}",
        "--log-folder",
        f"{str(log_path)}",
        "--device",
        "cpu",
        "--optimize-hyperparameters",
        "--n-jobs",
        "1",
        "--sampler",
        "tpe",
        "--pruner",
        "median",
        "--n-evaluations",
        "10",
        "--eval-episodes",
        "5", # 5 episodes, 1 minute each = 5 minutes
        "--n-eval-envs",
        "1",
        "--vec-env",
        "dummy",
        "--storage",
        str(db_path),
        "--study-name",
        study_name,
        "--n-trials",
        str(math.ceil(600 / num_procs)),
        "--verbose", "1",
        "--n-startup-trials", "0"
    ]

    # Run the command blocking
    subprocess.run(cmd, check=True, cwd=Path(__file__).absolute().parents[2] / "rl-baselines3-zoo")

def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--study-name', type=str, default='test_optimization',
                        help='Name of the optimization study')
    parser.add_argument('--n-processes', type=int, default=1,
                        help='Number of parallel processes to use')
    parser.add_argument('--reset-log-directories', action='store_true', default=False,
                        help='Delete all previous log directories')
    parser.add_argument('--drop-database', action='store_true', default=False, help='Drop the optuna database.')
    return parser.parse_args()


def main():
    args = parse_args()

    trial_name = args.study_name
    n_processes = args.n_processes # keep one core free for the MySQL server
    print(f"Running optimization of name {trial_name} with {n_processes} processes")
    system_config_file: Path = Path(__file__).absolute().parent / "system_config.json"
    data_dir: Path = Path(__file__).absolute().parent / "logs" / "train"
    opt_data_dir: Path = Path(__file__).absolute().parent / "logs" / "hparam"

    # Wipe data directories
    if args.reset_log_directories:
        print("Deleting previous log directories")
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if opt_data_dir.exists():
            shutil.rmtree(opt_data_dir)
    data_dir.mkdir(parents=True)
    opt_data_dir.mkdir(parents=True)

    # Initialize MySQL database
    sql_server = MySQLServer(user="root")
    sql_server.check_server_running()
    if sql_server.database_exists(trial_name) and args.drop_database:
        print("Dropping previous database")
        sql_server.drop_database(trial_name)
    sql_server.create_database(trial_name)

    # Run optimization with process pool
    with Pool(processes=n_processes) as pool:
        results = []
        for i in range(n_processes):
            print("Starting process", i)
            result = pool.apply_async(
                run_training,
                (
                    system_config_file,
                    opt_data_dir / f"proc_{i}",
                    data_dir / f"proc_{i}",
                    sql_server.get_db_endpoint(trial_name),
                    trial_name,
                    n_processes
                ))
            if i == 0:
                print("Waiting for MySQL server to start")
                time.sleep(5.0)
            results.append(result)

        for result in results:
            result.get()


if __name__ == "__main__":
    main()
