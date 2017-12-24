def plot_on(axes, x, y, title, x_label, y_label, **kwargs):
    axes.plot(x, y, **kwargs)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)

    return axes
