import matplotlib.pyplot as plt
import numpy as np
import sys


class CountingWindow:
    """opens a matplotlib window, displays an image, and provides cell counting related functionality"""

    def __init__(self, image):
        self.image = np.array(image)
        self.count = 0
        self.locations = []
        self.save = False
        self.fig, ax = plt.subplots()

    def open_window(self):
        plt.imshow(self.image)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.show()

    def get_results(self):
        if self.save:
            return self.count, self.locations
        else:
            return None, None

    def _on_click(self, event):
        if event.inaxes is None:
            return

        self.count += 1
        self.locations.append([event.xdata, event.ydata])
        self._place_cell_markers()

    def _on_key(self, event):
        if event.key is '=' or event.key is '+':
            self.count += 1
        elif event.key is '-':
            self.count -= 1
        elif event.key is 'backspace' and len(self.locations) >= 1:
            self._remove_cell_marker()
        elif event.key is 'enter':
            self.save = True
            plt.close()
        elif event.key is 'escape':
            self.save = False
            plt.close()
        elif event.key is 'q':
            self.save = False
            plt.close()
            sys.exit()

    def _place_cell_markers(self):
        if len(self.locations) >= 1:
            x = [*zip(*self.locations)][0]
            y = [*zip(*self.locations)][1]
            plt.scatter(x, y, color='red', marker='x')
        plt.title('Current count is {}'.format(self.count))
        self.fig.canvas.draw_idle()

    def _remove_cell_marker(self):
        self.count -= 1
        del self.locations[-1]
        self.fig.clf()
        plt.imshow(self.image)
        self._place_cell_markers()

