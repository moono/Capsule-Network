import numpy as np
import matplotlib.pyplot as plt


# save losses
def save_loss(loss, label, fn):
    fig, ax = plt.subplots()
    plt.plot(loss, alpha=0.5)
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5))
    plt.title(label)
    plt.legend()
    plt.savefig(fn)
    plt.close(fig)
    return


def form_image(multiple_images, val_block_size):
    def preprocess(img):
        img = (img * 255.0).astype(np.uint8)
        return img

    preprocesed = preprocess(multiple_images)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(multiple_images.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)

    return final_image
