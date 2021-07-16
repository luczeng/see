from matplotlib import pyplot as plt
import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
import numpy as np
import cv2
from skimage import color
from tqdm.auto import tqdm
import logging
from pathlib import Path
import argparse


def load_video(input_path: str, grayscale=True, time_compression=1.0):
    """
    Loads video from disk

    :param input_path: full path of the video
    :param verbose: Outputs deptails of the process (default=False)
    :param grayscale: Load the video as grayscale (one channel) or RGB (default=True)
    :param time_compression: values between zero and one to compress time.
    Use 1.0 to load the original video (default=1.0)
    :returns: numpy array containing the video (First dimension is time)
    """

    if not Path(input_path).is_file():
        raise ValueError(f"File {input_path} does NOT exist!")

    # Log info
    logging.info(f"Orginal path: {input_path}")

    cap = cv2.VideoCapture(str(input_path))

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Dimensions: {frameCount,frameHeight,frameWidth}")

    # Check for strange frameCount
    if frameCount > 1000000 or frameCount == 0:
        logging.debug(
            f"Got from cv2 a total of {frameCount} frames, thats way too much!"
        )
        logging.debug("Double checking...")
        nframes_check = 0

        while True:
            # Read frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            nframes_check += 1
        logging.info(f"Frames checked, total of {nframes_check} found")
        frameCount = nframes_check  # Update the previous number
        # Restart capture
        cap.release()
        cap = cv2.VideoCapture(str(input_path))

    # Create buffer array
    if grayscale:
        logging.debug("Video will be converted to grayscale")
        buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype("uint8"))
    else:
        logging.debug("Video will be loaded as RGB")
        buf = np.empty(
            (frameCount, frameHeight, frameWidth, 3), np.dtype("uint8")
        )

    # Read video
    fc = 0
    ret = True
    pbar_load_video = tqdm(total=frameCount, unit_scale=True, leave=False)
    pbar_load_video.set_description("Loading video")

    while fc < frameCount and ret:
        pbar_load_video.update(1)
        ret, img = cap.read()

        if grayscale:
            gray_img = color.rgb2gray(img)
            buf[fc] = np.uint8(gray_img * 255)
        else:
            buf[fc] = img
        fc += 1

    cap.release()

    logging.info(f"File loaded: {np.shape(buf)} {buf.dtype}")
    pbar_load_video.close()

    return buf


class PageSlider(matplotlib.widgets.Slider):

    def __init__(self, ax, label, numpages = 10, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True,
                 dragging=True, **kwargs):

        self.facecolor=kwargs.get('facecolor',"w")
        self.activecolor = kwargs.pop('activecolor',"b")
        self.fontsize = kwargs.pop('fontsize', 7)
        self.numpages = numpages

        super(PageSlider, self).__init__(ax, label, 0, numpages,
                            valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle((float(i)/numpages, 0), 1./numpages, 1,
                                transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax,label='$\u25C0$',color=self.facecolor,hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax,label='$\u25B6$', color=self.facecolor,hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i+1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i-1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)


def run_videre(input_video):

    data = load_video(input_video)
    num_pages = data.shape[0]
    print(data.shape)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)

    im = ax.imshow(data[0, :, :], cmap='viridis', interpolation='nearest')

    ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])
    slider = PageSlider(ax_slider, 'Page', num_pages, activecolor="orange")

    def update(val):
        i = int(slider.val)
        im.set_data(data[i,:,:])

    slider.on_changed(update)

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs the motion blur estimation network on a blurred image with a randomly generated kernel"
    )
    parser.add_argument(
        "-i",
        "--input_video",
        type=str
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    run_videre(args.input_video)
