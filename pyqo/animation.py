"""
This module implements animations with matplotlib.
"""

import numpy
import tempfile
import os
import sys
import shutil

try:
    import pylab
except:
    print("matplotlib not found - plotting not possible.")

try:
    import PyQt4.Qt as Qt
except:
    print("PyQt4 not found - animations not possible")
    class Qt(object):
        def __getattr__(self, name):
            return object
    Qt = Qt()

try:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT,\
        FigureCanvasQTAgg
    from matplotlib.backends.qt4_compat import _getSaveFileName
except:
    print("Can't import matplotlib gtk backends - animations not possible.")
    NavigationToolbar2QT = object
    FigureCanvasQTAgg = object

try:
    from mpl_toolkits.mplot3d import axes3d, art3d
except:
    print("Can't import matplotlib 3D toolkit - 3D animations not possible.")

class PlayButton(Qt.QPushButton):
    def __init__(self):
        Qt.QPushButton.__init__(self, "Play")
        self._plays = False
        self.connect(self, Qt.SIGNAL("clicked()"), self.handle_clicked)

    def handle_clicked(self, *args):
        if self.plays():
            self.pause()
        else:
            self.play()

    def plays(self):
        return self._plays

    def play(self):
        self._plays = True
        self.emit(Qt.SIGNAL("clicked-play"))

    def pause(self):
        self._plays = False
        self.emit(Qt.SIGNAL("clicked-pause"))

class PositionScale(Qt.QSlider):
    def __init__(self, steps):
        Qt.QSlider.__init__(self, 1)
        self.setMinimum(0)
        self.setMaximum(steps-1)
        self.setTickPosition(0)

class AnimationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, window, steps):
        NavigationToolbar2QT.__init__(self, canvas, window)
        self.win = window
        self.playbutton = PlayButton()
        self.positionscale = PositionScale(steps)
        self.addWidget(self.playbutton)
        self.addWidget(self.positionscale)
        self.addAction("save movie", self.handle_save_movie)

    def handle_save_movie(self, *args):
        fname = _getSaveFileName(self, "Choose a filename to save t",
                                 "movie.avi", "avi (*.avi)", "avi (*.avi)")
        if fname:
            self.win.save_movie(fname)

class Canvas(FigureCanvasQTAgg):
    def plot(self, step):
        raise NotImplementedError()

    def fast_plot(self, step):
        return self.plot(step)


class Animation(Qt.QMainWindow):
    def __init__(self, canvas, steps):
        Qt.QMainWindow.__init__(self)

        self.step = 0
        self.steps = steps
        self._play = False
        self.canvas = canvas
        self.create_widgets()

    def create_widgets(self):
        self.toolbar = AnimationToolbar(self.canvas, self, self.steps)
        self.timer = Qt.QTimer()

        self.connect(self.timer, Qt.SIGNAL("timeout()"),
                     self.do_step)
        self.connect(self.toolbar.playbutton, Qt.SIGNAL("clicked-play"),
                     self.handle_play)
        self.connect(self.toolbar.playbutton, Qt.SIGNAL("clicked-pause"),
                     self.handle_pause)
        self.connect(self.toolbar.positionscale, Qt.SIGNAL("valueChanged(int)"),
                     self.scroll)
        vbox = Qt.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.toolbar)

        central_widget = Qt.QWidget()
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)
        self.canvas.plot(0)

    def do_step(self, *args):
        if self.step < self.steps-1:
            self.step += 1
            self.canvas.fast_plot(self.step)
            self.toolbar.positionscale.setValue(self.step)
            return True
        self.toolbar.playbutton.pause()
        return False

    def handle_play(self, *args):
        if self.step >= self.steps-1:
            self.step = 0
        self.timer.start(0.20)

    def handle_pause(self, *args):
        self.timer.stop()

    def scroll(self, value):
        if value == self.step:
            return
        elif value < 0:
            self.step = 0
        elif value < self.steps:
            self.step = int(value)
        else:
            self.step = self.steps-1
        self.canvas.fast_plot(self.step)

    def handle_resize(self, *args):
        self.canvas.plot(self.step)

    def save_movie(self, filename, format="avi"):
        import subprocess
        self.step = 0
        self.canvas.plot(self.step)
        tempdirpath = tempfile.mkdtemp(prefix="pycppqed_movie_")
        w, h = self.canvas.get_width_height()
        while self.step < self.steps:
            self.canvas.fast_plot(self.step)
            path = os.path.join(tempdirpath, "%04d.png" % self.step)
            self.canvas.print_figure(str(path))
            self.step += 1
        COMMAND = ('mencoder',
                'mf://' + os.path.join(tempdirpath, "*.png"),
                '-mf',
                'type=png:w=%s:h=%s:fps=25' % (w, h),
                '-ovc',
                'lavc',
                '-lavcopts',
                #'vcodec=ffvhuff',
                #'vcodec=ffv1',
                'vcodec=mpeg4',
                #'vcodec=snow:vstrict=-2',
                '-oac',
                'copy',
                '-o',
                filename)
        subprocess.call(COMMAND)
        shutil.rmtree(tempdirpath)

def animate(length, plot, fast_plot=None):
    class SpecializedCanvas(Canvas):
        def plot(self, step):
            plot(self.figure, step)
            self.draw()
        def fast_plot(self, step):
            if fast_plot is None:
                plot(self.figure, step)
            else:
                fast_plot(step)
            self.draw()
    fig = pylab.figure()
    app = Qt.QApplication(sys.argv)
    canvas = SpecializedCanvas(fig)
    animation = Animation(canvas, length)
    animation.show()
    app.exec_()

