# kleine hulpfuncties zodat alle staafdiagrammen er hetzelfde uitzien
import matplotlib.pyplot as plt


def annotate_bars(ax, as_percentage=False, fmt="{:.0f}"):
    # zet de waarde boven elke staaf (met % als as_percentage)
    for container in ax.containers:
        if as_percentage:
            labels = [f"{bar.get_height():.0f}%" for bar in container]
        else:
            labels = [fmt.format(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)


def finalize_bar_axis(ax, ymax=None):
    # y-as altijd op 0 beginnen (anders lijken verschillen groter).
    # zonder ymax: bovenkant + 12% lucht voor de labels.
    _, upper = ax.get_ylim()
    ax.set_ylim(0, ymax if ymax is not None else upper * 1.12)


def new_axes(figsize=(8, 5)):
    # lege figuur + assen aanmaken
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
