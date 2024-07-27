from collections import namedtuple

import numpy as np
from agents.metrics import MovingAverage, MovingTotal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# instead of dict use namedtuple
SystemMetrics = namedtuple("Metrics", ["lost_messages", "output_age_min", "output_age_max", "output_age_avg"])
LearningMetrics = namedtuple("Metrics", ["loss", "epsilon", "reward"])


class DataLogger(object):
    def __init__(self, window_size: int = 100, tensorboard_writer: SummaryWriter = None, tensorboard_period: int = 10):
        self.window_size = window_size
        self.tensorboard_writer = tensorboard_writer
        self.tensorboard_period = tensorboard_period
        self.reset()

    def reset(self):
        # Filters for metrics
        self.mt_lost_messages = MovingTotal(self.window_size)
        self.ma_output_age_min = MovingAverage(self.window_size)
        self.ma_output_age_max = MovingAverage(self.window_size)
        self.ma_output_age_avg = MovingAverage(self.window_size)
        self.ma_reward = MovingAverage(self.window_size)

        # Logging for filtered metrics
        self.sys_time = list()
        self.lost_messages = list()
        self.output_age_min = list()
        self.output_age_max = list()
        self.output_age_avg = list()

        self.learn_time = list()
        self.avg_reward = list()
        self.loss = list()
        self.epsilon = list()

        self.iteration = 0

    def check_write_tensorboard(self) -> bool:
        return (
            (self.tensorboard_writer is not None)
            and (len(self.sys_time) % self.tensorboard_period == 0)
            and (len(self.sys_time) > 0)
        )

    def log_system_metrics(self, time: int, metrics: SystemMetrics):
        # Update the filters
        self.mt_lost_messages.push(metrics.lost_messages)
        self.ma_output_age_min.push(metrics.output_age_min)
        self.ma_output_age_max.push(metrics.output_age_max)
        self.ma_output_age_avg.push(metrics.output_age_avg)

        # Store the filtered metrics
        self.sys_time.append(time)
        self.lost_messages.append(self.mt_lost_messages.value)
        self.output_age_min.append(self.ma_output_age_min.value)
        self.output_age_max.append(self.ma_output_age_max.value)
        self.output_age_avg.append(self.ma_output_age_avg.value)

    def log_train_metrics(self, time: int, metrics: LearningMetrics):
        # Update the filters
        self.ma_reward.push(metrics.reward)

        # Store the filtered metrics
        self.learn_time.append(time)
        self.avg_reward.append(self.ma_reward.value)
        self.loss.append(metrics.loss)
        self.epsilon.append(metrics.epsilon)

    def write_to_tensorboard(self):
        self.iteration += 1
        if self.tensorboard_writer is None:
            return
        if len(self.sys_time) == 0:
            return
        if len(self.learn_time) == 0:
            return
        if (self.iteration % self.tensorboard_period) != 0:
            return
        self.tensorboard_writer.add_scalar("system/lost_messages", self.mt_lost_messages.value, self.sys_time[-1])
        self.tensorboard_writer.add_scalar("system/output_age_min", self.ma_output_age_min.value, self.sys_time[-1])
        self.tensorboard_writer.add_scalar("system/output_age_max", self.ma_output_age_max.value, self.sys_time[-1])
        self.tensorboard_writer.add_scalar("system/output_age_avg", self.ma_output_age_avg.value, self.sys_time[-1])
        self.tensorboard_writer.add_scalar("train/loss", self.loss[-1], self.learn_time[-1])
        self.tensorboard_writer.add_scalar("train/epsilon", self.epsilon[-1], self.learn_time[-1])
        self.tensorboard_writer.add_scalar("train/reward", self.avg_reward[-1], self.learn_time[-1])


def plot_system_metrics(logger: DataLogger):
    # Convert time from ms to minutes
    sys_time_minutes = np.array(logger.sys_time) / (1000 * 60)

    # Plot two axes in once figure, one for the output* times, and the second for the lost messages
    fig, axs = plt.subplots(2, figsize=(10, 10))
    fig.suptitle("System Metrics")
    axs[0].plot(sys_time_minutes, logger.output_age_min, label="Output Age Min")
    axs[0].plot(sys_time_minutes, logger.output_age_avg, label="Output Age Avg")
    axs[0].plot(sys_time_minutes, logger.output_age_max, label="Output Age Max")
    axs[0].set_ylabel("Average Output Age over Last Minute (ms)")
    axs[0].legend()
    axs[0].grid(True)
    # Calculate the lower and upper bounds for y-axis scaling
    upper_bound = np.percentile(logger.output_age_max, 99.9)
    # Set the y-axis limits
    axs[0].set_ylim(0.0, upper_bound)
    axs[1].plot(sys_time_minutes, logger.lost_messages, label="Lost Messages")
    axs[1].set_xlabel("Time (minutes)")
    axs[1].set_ylabel("Number of Lost Messages during Last Minute (counts)")
    axs[1].set_ylim(0.0, 10.0)
    axs[1].grid(True)
    axs[1].legend()
    return fig, axs


def plot_train_metrics(logger: DataLogger):
    fig, axs = plt.subplots(3, figsize=(10, 10))
    train_time_minutes = np.array(logger.learn_time) / (1000 * 60)

    fig.suptitle("Train Metrics")
    axs[0].plot(train_time_minutes, logger.loss, label="Loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_time_minutes, logger.epsilon, label="Epsilon")
    axs[1].set_ylabel("Epsilon")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(train_time_minutes, logger.avg_reward, label="Reward")
    axs[2].set_xlabel("Time (minutes)")
    axs[2].set_ylabel("Reward")
    axs[2].legend()
    axs[2].grid(True)
    return fig, axs
