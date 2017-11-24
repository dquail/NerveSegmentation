from pylab import *


def make_plot(yValues, mainTitle, subtitle):
    print("Making plot")
    fig = plt.figure(1)
    
    fig.suptitle(mainTitle, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    titleLabel = subtitle
    ax.set_title(titleLabel)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice co-efficient')

    ax.plot(yValues)
    
    #ax.plot(optimalActionsNonStationary)
    plt.show()
    
def simpleUNET():
    a = [0.0281, 0.2015, 0.2652, 0.2767, 0.2648, 0.2887, 0.2640, 0.2971, 0.2925, 0.3065, 0.3470, 0.3541, 0.3608, 0.3804, 0.4383, 0.4508, 0.4286, 0.4736, 0.4982, 0.5049]
    make_plot(a, "UNET CNN", "Performance over epoch")
    
def augmentedUNET():
    a = [0.034, 0.38, 0.389, 0.41, 0.413, 0.45, 0.481, 0.49, 0.501, 0.502, 0.51, 0.514, 0.55, 0.57, 0.589, 0.60, 0.642, 0.654, 0.670, 0.6710]
    make_plot(a, "UNET CNN with Augmentation", "Performance over epoch")
    
def segNET():
    a = [0.030, 0.36, 0.378, 0.413, 0.414, 0.419, 0.423, 0.46, 0.498, 0.50, 0.503, 0.506, 0.515, 0.519, 0.55, 0.589, 0.599, 0.620, 0.639, 0.642]
    make_plot(a, "UNET CNN with Augmentation", "Performance over epoch")
    
def make_plot_comparison():
    print("Making plot")
    fig = plt.figure(1)
    x1Values = [1,4,5,8,9]
    y1Values = [1,2,3,4,5]
    y2Values = [5,4,3,2,1]
    fig.suptitle("mainTitle", fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    titleLabel = "subtitle"
    ax.set_title(titleLabel)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice co-efficient')

    ax.plot(x1Values, y1Values, label = "100 examples")
    ax.plot(x1Values, y2Values, label = "1000 examples")
    ax.legend()
    #ax.plot(optimalActionsNonStationary)
    plt.show()    