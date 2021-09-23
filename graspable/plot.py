import matplotlib.pyplot as plt


def save_images_cascade(path, images, labels, size=(15, 15)):
    fig, axs = plt.subplots(len(images[0]), len(labels))
    fig.set_figheight(size[0])
    fig.set_figwidth(size[1])

    for i, label in enumerate(labels):
        for j, image in enumerate(images[i]):
            axs[j, i].imshow(image, vmin=0, vmax=1)
            axs[j, i].set_axis_off()
            if j == 0:
                axs[j, i].set_title(label)

    plt.savefig(path)
    plt.close(fig)
