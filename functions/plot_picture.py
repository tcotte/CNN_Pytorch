import matplotlib.pyplot as plt


# helper function to show an image
def matplotlib_imshow(img, label):
    img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.title(label)
    plt.imshow(npimg, cmap="Greys")
