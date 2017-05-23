import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid", palette = "muted")

def generategraph(x, accuracy, lost):
    
    fig = plt.figure(figsize = (15, 5))
    
    plt.subplot(1, 2, 1)
    
    plt.plot(x, lost)
    plt.xlabel('Epoch')
    plt.ylabel('lost')
    plt.title('LOST')
    
    plt.subplot(1, 2, 2)
    
    plt.plot(x, accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('ACCURACY')
    
    fig.tight_layout()
    plt.savefig('graph.png')
    plt.savefig('graph.pdf')
    plt.cla()