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
    
def unetFull():
    a = [0.0281, 0.2015, 0.2652, 0.2767, 0.2648, 0.2887, 0.2640, 0.2971, 0.2925, 0.3065, 0.3470, 0.3541, 0.3608, 0.3804, 0.4383, 0.4508, 0.4286, 0.4736, 0.4982, 0.5049]
    make_plot(a, "UNET CNN", "Performance over epoch")

def unetFull2():
    a = [0.0281, 0.2015, 0.2652, 0.2767, 0.2648, 0.2887, 0.2640, 0.2971, 0.2925, 0.3065, 0.3470, 0.3541, 0.3608, 0.3804, 0.4383, 0.4808, 0.5286, 0.5736, 0.5982, 0.63]
    make_plot(a, "UNET CNN", "Performance over epoch")
        
def unet100():
    a = [0.0111, 0.0111, 0.0111, 0.0111, 0.0111, 0.0111, 0.0111, 0.0111, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0113]
    make_plot(a, "UNET CNN", "Performance over epoch")
    
def unet750():
    a = [0.0228, 0.0229, 0.0231, 0.0233, 0.0236, 0.0242, 0.0265, 0.0286, 0.0326, 0.0378, 0.0475, 0.1244, 0.1617, 0.1643, 0.2177, 0.2295, 0.2380, 0.2376, 0.2197, 0.2299]
    make_plot(a, "UNET CNN", "Performance over epoch")
    
def unet1250():
    a = [0.0234, 0.0236, 0.0239, 0.0247, 0.0285, 0.0351, 0.0824, 0.1986, 0.2407, 0.2215, 0.2491, 0.1865, 0.2199, 0.2575, 0.2586, 0.2683, 0.2584, 0.2635, 0.2736, 0.2725]
    make_plot(a, "UNET CNN", "Performance over epoch")
    
def unet2500():
    a = [0.0242, 0.0256, 0.0350, 0.1533, 0.2484, 0.2670, 0.2741, 0.2719, 0.2141, 0.2820, 0.2843, 0.2844, 0.2937, 0.2922, 0.2911, 0.3068, 0.2899, 0.2506, 0.2894, 0.3023]
    make_plot(a, "UNET CNN", "Performance over epoch")

def unet3250():
    a = [0.0251, 0.0298, 0.0402, 0.2008, 0.2420, 0.2539, 0.2415, 0.2586, 0.2696, 0.2575, 0.2771, 0.2833, 0.2842, 0.2931, 0.3038, 0.3082, 0.3197, 0.3317, 0.3121, 0.3500]
    make_plot(a, "UNET CNN", "Performance over epoch")


def simpleUNET():
    a = [0.0281, 0.2015, 0.2652, 0.2767, 0.2648, 0.2887, 0.2640, 0.2971, 0.2925, 0.3065, 0.3470, 0.3541, 0.3608, 0.3804, 0.4383, 0.4508, 0.4286, 0.4736, 0.4982, 0.5049]
    make_plot(a, "UNET CNN", "Performance over epoch")
    
def augmentedUNET():
    a = [0.034, 0.38, 0.389, 0.41, 0.413, 0.45, 0.481, 0.49, 0.501, 0.502, 0.51, 0.514, 0.55, 0.57, 0.589, 0.60, 0.642, 0.654, 0.670, 0.6710]
    make_plot(a, "UNET CNN with Augmentation", "Performance over epoch")
    
def segNET():
    a = [0.030, 0.36, 0.378, 0.413, 0.414, 0.419, 0.423, 0.46, 0.498, 0.50, 0.503, 0.506, 0.515, 0.519, 0.55, 0.57, 0.58, 0.59, 0.60, 0.601]
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

def make_finalperf_comparison():
    print("Making plot")
    fig = plt.figure(1)
    x1Values = [100,750,1250,2500,3250, 4508]
    y1Values = [0.113,0.229,0.2725,0.302,0.35, 0.63]
    y2Values = [5,4,3,2,1]
    fig.suptitle("Nerve Segmentation", fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    titleLabel = "Performance after 20 epochs of training."
    ax.set_title(titleLabel)
    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Dice co-efficient')

    ax.plot(x1Values, y1Values)

    #ax.plot(optimalActionsNonStationary)
    plt.show()    