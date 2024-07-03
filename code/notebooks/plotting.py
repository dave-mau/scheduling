import graphviz


def plot_sampler(ax, sampler, N=10_000, num_bins=20):
    samples = [sampler.sample() for _ in range(N)]
    hist = ax.hist(samples, density=True, alpha=0.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [-]")
    return hist


def plot_samplers(ax, sampler_dict, **kwargs):
    legends = []
    for name, val in sampler_dict.items():
        hist = plot_sampler(ax, val, **kwargs)
        legends.append(name)
    ax.legend(legends)
