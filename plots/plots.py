import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_train_test_loss(model):
    mpl.rcParams['figure.figsize'] = (12, 8)
    mpl.rcParams['axes.grid'] = False
    # plot history
    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def fun_plot_train_test_loss(model, history):
    mpl.rcParams['figure.figsize'] = (12, 8)
    mpl.rcParams['axes.grid'] = False
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()