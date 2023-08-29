import numpy as np
import os
import tensorflow as tf
import pyqtgraph as pg
from PySide6 import QtGui, QtCore, QtWidgets
from nn import MLP, one_hot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def get_batch(x, y, batch_size):
    indices = np.random.randint(0, x.shape[1], batch_size)
    x_ = x[:, indices]
    y_ = y[indices, :]
    return x_, y_

def to_float(value, offset=-1):
    return offset + value/255*2.

def to_int(value, base=127, scaling=2):
    value = int(round(base + (value*255/scaling)))
    value = 0 if value < 0 else value
    value = 255 if value > 255 else value
    return value

class GroupBox(QtWidgets.QGroupBox):
    def __init__(self, title='', parent=None):
        super().__init__(title, parent=parent)
        self.par = parent
        self.setStyleSheet("QGroupBox {font: bold 14px;}")
        # center text
        self.setAlignment(QtCore.Qt.AlignCenter)



class ParameterSpinBox(QtWidgets.QWidget):
    def __init__(self, parent, init_val, name):
        super(ParameterSpinBox, self).__init__(parent=parent)

        self.par = parent
        self.spin_box = QtWidgets.QSpinBox()
        self.spin_box.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spin_box.setValue(init_val)
        self.slider.valueChanged[int].connect(self.value_changed)
        self.spin_box.valueChanged[int].connect(self.par.change_image)
        self.spin_box.value
        
        self.label = QtWidgets.QLabel(name)
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.spin_box)
        self.main_layout.addWidget(self.slider)
        self.setLayout(self.main_layout)

    def set_ranges(self, min, max):
        self.spin_box.setRange(min, max)
        self.slider.setRange(min, max)

    def value_changed(self, value):
        self.spin_box.setValue(value)


class ParameterCheckBox(QtWidgets.QWidget):
    def __init__(self, name, parent, fn):
        super(ParameterCheckBox, self).__init__(parent=parent)
        self.fn = fn
        self.name = name
        self.label = QtWidgets.QLabel(name)
        self.check = QtWidgets.QCheckBox()
        self.check.clicked[bool].connect(self.fn)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.check)
        # self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0 ,0)
        # equal spacing
        self.main_layout.addStretch(1)
        # self.label.setFixedWidth(100)

class NeuronCheckBox(QtWidgets.QCheckBox):
    def __init__(self, parent):
        self.par = parent
        super(NeuronCheckBox, self).__init__(parent=parent)
        self.clicked[bool].connect(self.dropout)
        self.setFixedSize(10, 10)
        self.setCheckable(True)
        self.setChecked(True)
        ## No margins around the checkbox (tight fit)
        self.setStyleSheet("QCheckBox::indicator { width: 10px; height: 10px; }")
    
    def dropout(self, status):
        # self.par.par.layer.mask[self.par.index] = 1 if status else 0
        self.par.par.dropout(self.par.index, status)


class LabeledComboBox(QtWidgets.QWidget):
    def __init__(self, parent, label, items, selection='Default', use_label=True):
        super(LabeledComboBox, self).__init__(parent=parent)
        if use_label:
            self.label = QtWidgets.QLabel(label, alignment=QtGui.Qt.AlignRight)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(items)
        self.combo.setCurrentIndex(0)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.label) if use_label else None
        layout.addWidget(self.combo)
        layout.setContentsMargins(0, 0, 0 ,0)
        # self.combo.setFixedWidth(145)


class ToggleButton(QtWidgets.QPushButton):
    """
    Button with extended toggling functionality
    """
    def __init__(self, parent, names, trigger, status=None, text=False, tip=None):
        super(ToggleButton, self).__init__(parent=parent)
        self.par = parent
        self.setCheckable(True)
        self.names = names
        self.status = status
        self.use_text = text
        self.status_change(False)
        self.clicked[bool].connect(getattr(self.par, trigger))
        self.clicked[bool].connect(self.status_change)
        if tip:
            self.setToolTip(tip)

        icon = QtGui.QIcon
        icon = icon(parent=self)
        self.setIcon(icon)
        if self.use_text:
            self.setMinimumHeight(25)
        else:
            self.setFixedSize(30, 30)
    
    def status_change(self, toggled):
        tip = self.names[1] if toggled else self.names[0]
        status = self.status[1] if toggled else self.status[0]
        self.setStatusTip(status)
        if self.use_text:
            self.setText('  '+tip)


class App(QtWidgets.QMainWindow):
    def __init__(self, dataset, arch, parent=None):
        super(App, self).__init__(parent)
        self.ui = MLPUI(dataset=dataset, arch=arch)
        self.setCentralWidget(self.ui)
        self.status_bar = self.statusBar()


class Indicator(QtWidgets.QPushButton):
    def __init__(self, parent=None, index=''):
        super().__init__(str(index), parent=parent)
        self.par = parent
        self.current_color = []
        self.index = index
        self.setFixedSize(45, 45)
        self.setEnabled(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum);
        self.set_color(0, 0, 0, str(index))
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.reset)
        self.lit = False
        self.timer.setSingleShot(True)

    def set_color(self, r, g, b, text=None):
        stylesheet = """ 
            QPushButton{
                border-style: outset;
                border-width: 2px;
                border-radius: 22px;
                border-color: gray;
                font: 15px;
                color: rgb(255,255,255);
                padding: 2px;"""
        if r > 150:
            stylesheet = """ 
            QPushButton{
                border-style: outset;
                border-width: 2px;
                border-radius: 22px;
                border-color: gray;
                font: 15px;
                color: rgb(0,0,0);
                padding: 2px;"""
        style = stylesheet + "\n    background-color: rgb({}, {}, {})".format(r, g, b) + "\n}"
        self.current_color = (r, g, b)
        self.setStyleSheet(style)
        if not text:
            text = str(self.index)
        self.setText(text)

    def reset(self):
        self.set_color(0, 0, 0)

    def highlight(self, correct=False):
        color = 'rgb(100, 200, 100)' if correct else 'rgb(200, 50, 50)'
        r, g, b = self.current_color
        stylesheet = """ 
        QPushButton{
            border-style: solid;
            border-width: 10px;
            border-radius: 22px;
            border-color: %s;
            font: bold 16px;
            color: rgb(0,0,0);
            padding: 2px;""" % color
        style = stylesheet + "\n    background-color: rgb({}, {}, {})".format(r, g, b) + "\n}"
        self.current_color = (r, g, b)
        self.setStyleSheet(style)
        self.setText(str(self.index))
    
    def light(self, color, lt=False):
        if color == 'r':
            self.set_color(255, 100, 100)
        elif color == 'g':
            self.set_color(100, 255, 100)
        t = 1000 if lt else self.par.par.delay
        self.timer.start(t)

    def set_activation(self, val):
        # self.setText(str(self.index) + '\n' + val)
        pass

class PassIndicator(QtWidgets.QToolButton):
    def __init__(self, parent=None, up=True, index=''):
        super().__init__(parent=parent)
        type = QtCore.Qt.UpArrow if not up else QtCore.Qt.DownArrow
        self.setArrowType(type)
        self.par = parent
        self.current_color = []
        self.index = index
        self.setFixedSize(45, 45)
        self.setEnabled(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum);
        self.set_color(0, 0, 0, str(index))
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.reset)
        self.lit = False
        self.timer.setSingleShot(True)

    def set_color(self, r, g, b, text=None):
        stylesheet = """ 
            QPushButton{
                border-style: outset;
                border-width: 2px;
                border-radius: 22px;
                border-color: gray;
                font: 15px;
                color: rgb(255,255,255);
                padding: 2px;"""
        if r > 150:
            stylesheet = """ 
            QPushButton{
                border-style: outset;
                border-width: 2px;
                border-radius: 22px;
                border-color: gray;
                font: 15px;
                color: rgb(0,0,0);
                padding: 2px;"""
        style = stylesheet + "\n    background-color: rgb({}, {}, {})".format(r, g, b) + "\n}"
        self.current_color = (r, g, b)
        self.setStyleSheet(style)
        if not text:
            text = str(self.index)
        self.setText(text)

    def reset(self):
        self.set_color(0, 0, 0)

    def highlight(self, correct=False):
        color = 'rgb(100, 200, 100)' if correct else 'rgb(200, 50, 50)'
        r, g, b = self.current_color
        stylesheet = """ 
        QPushButton{
            border-style: solid;
            border-width: 10px;
            border-radius: 22px;
            border-color: %s;
            font: bold 16px;
            color: rgb(0,0,0);
            padding: 2px;""" % color
        style = stylesheet + "\n    background-color: rgb({}, {}, {})".format(r, g, b) + "\n}"
        self.current_color = (r, g, b)
        self.setStyleSheet(style)
        self.setText(str(self.index))
    
    def light(self, color, lt=False):
        if color == 'r':
            self.set_color(255, 100, 100)
        elif color == 'g':
            self.set_color(100, 255, 100)
        t = 5000 if lt else self.par.par.delay
        self.timer.start(t)

    def set_activation(self, val):
        # self.setText(str(self.index) + '\n' + val)
        pass

class ImageDisplay(pg.ImageView):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.display = pg.ImageItem(None, border="w")
        self.addItem(self.display)
        for element in ['menuBtn', 'roiBtn', 'histogram']:
            getattr(self.ui, element).hide()

    def update_image(self, image):
        img = np.flip(np.rot90(image, k=3), axis=1)
        self.display.setImage(image)


class WeightSlider(QtWidgets.QSlider):
    def __init__(self, init_val, neuron_index, weight_index, parent=None, is_bias=False):
        super().__init__(QtGui.Qt.Vertical, parent=parent)

        self.setStyleSheet(self.get_style('black'))
        self.is_bias = is_bias
        self.setMaximumWidth(15)
        self.setRange(0, 255)
        # self.setSingleStep(127)
        self.setTickInterval(1)
        # self.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        # self.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.n_index = neuron_index
        self.w_index = weight_index
        self.setValue(to_int(init_val))
        self.sliderMoved.connect(self.update_weight)
        self.par = parent

    
    def get_style(self, color):
        # pass

        style = """
            QSlider::groove:vertical {
                background: rgb(50,100,100);
                position: absolute;
                left: 4px; right: 4px;
                width: 2px;}
            QSlider::handle:vertical {
                background: %s;
                height: 8px;
                width: 8px;
                margin: 0 -4px;}
            """ % color
        return style

    def update_weight(self):
        if not self.is_bias:
            self.par.par.layer.w[self.w_index, self.n_index] = to_float(self.value())
        else:
            self.par.par.layer.b[self.w_index] = to_float(self.value())    
        self.par.par.par.forward_pass()

    def set_weight(self, value):
        prev_val = self.value()
        new = to_int(value)
        self.setValue(new)
        if new >= prev_val+1:
            self.setStyleSheet(self.get_style('rgb(255, 255, 255)'))
        elif new <= prev_val-1:
            self.setStyleSheet(self.get_style('rgb(0, 0, 0)'))
        else:
            self.setStyleSheet(self.get_style('rgb(127, 127, 127)'))

    # ## Add horizontal tick lines at the top, middle and the bottom of the slider
    # def paintEvent(self, event):
    #     super().paintEvent(event)
    #     painter = QtGui.QPainter(self)
    #     painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1, QtCore.Qt.SolidLine))
    #     painter.drawLine(self.width()/4, 0, self.width()/4*3, 0)
    #     painter.drawLine(self.width()/4, self.height()/2, self.width()/4*3, self.height()/2)
    #     painter.drawLine(self.width()/4, self.height()-1, self.width()/4*3, self.height()-1)
    # def paintEvent(self, event)
    
    #     paintEvent(event)

    #     // Add your custom tick locations
    #     QPainter painter(this);
    #     painter.setRenderHint(QPainter::Antialiasing);
    #     painter.setPen(Qt::darkGreen);
    #     painter.drawRect(1, 2, 6, 4);    

    #     painter.setPen(Qt::darkGray);
    #     painter.drawLine(2, 8, 6, 2);
    # }


class WeightContainer(QtWidgets.QWidget):
    def __init__(self, weights, parent=None, is_bias=False):
        super().__init__(parent=parent)
        self.par = parent

        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(2)
        self.weights = []
        l1 = QtWidgets.QLabel('+1')
        l2 = QtWidgets.QLabel('-1')
        font = l1.font()
        font.setPointSize(7)
        font.italic()
        l1.setFont(font)
        l2.setFont(font)
        self.is_bias = is_bias

        lay.addWidget(l1, 1, 0, alignment=QtCore.Qt.AlignTop)
        lay.addWidget(l2, 2, 0, alignment=QtCore.Qt.AlignBottom)
        for i, w in enumerate(weights):
            weight = WeightSlider(w, parent.index, i, parent=parent, is_bias=is_bias)
            self.weights.append(weight)
            label = QtWidgets.QLabel(str(f'w<sub>{i}</sub>'))
            label.setFixedHeight(15)
            font = label.font()
            font.setPointSize(9)
            font.italic()
            label.setFont(font)
            lay.addWidget(label, 0, i+1)
            lay.addWidget(weight, 1, i+1, 2, 1)

class Neuron(QtWidgets.QGroupBox):
    def __init__(self, weights, index, parent=None, title='', is_bias=False):
        super().__init__(parent=parent, title=title)

        self.par = parent
        self.index = index
        self.indicator = Indicator(self, index)
        self.weights = WeightContainer(weights, parent=self, is_bias=is_bias)
        # self.activation_edit = QtWidgets.QLineEdit()
        # font = QtGui.QFont('Ubuntu', 6)
        # self.activation_edit.setFont(font)
        # self.activation_edit.setMaximumSize(30, 12)
        # self.activation_edit.setText('0.0')
        self.setFixedWidth(80)
        self.check = NeuronCheckBox(self)
        widgets = [self.weights, self.indicator, self.check] #, self.activation_edit
        ## Small checkbox at the bottom left corner of the neuron group box

        lay = QtWidgets.QVBoxLayout(self)
        lay.setAlignment(QtGui.Qt.AlignHCenter)
        lay.setSpacing(0)
        for widget in widgets:
            lay.addWidget(widget, alignment=QtGui.Qt.AlignCenter)
        lay.setContentsMargins(0, 0, 0, 0)
    
    def set_weights(self, weights):
        for i, w in enumerate(weights):
            self.weights.weights[i].set_weight(w)


class InputNeuron(QtWidgets.QGroupBox):
    def __init__(self, weights, index, parent=None, title=''):
        super().__init__(parent=parent, title=title)
        self.par = parent

        self.index = index
        self.indicator = Indicator(self, index)

        self.weights = ImageDisplay(self)
        self.weights.setFixedSize(125, 125)
        # self.setFixedWidth(80)
        # self.activation_edit = QtWidgets.QLineEdit()
        # font = QtGui.QFont('Ubuntu', 6)
        # self.activation_edit.setFont(font)
        # self.activation_edit.setMaximumSize(30, 12)
        # self.activation_edit.setText('0.0')
        
        lay = QtWidgets.QGridLayout(self)
        lay.setSpacing(0)
        for i, widget in enumerate([self.weights, self.indicator]): #, self.activation_edit
            lay.addWidget(widget, i, 1, 1, 1,  alignment=QtGui.Qt.AlignCenter)

        w = self.par.layer.w[:, index].reshape(*self.par.par.data_container.data_shape)
        self.weights.update_image(w)
        lay.setContentsMargins(0, 0, 0, 0)
            
    def set_weights(self, data):
        data = np.flip(np.rot90(data, k=3), axis=1)
        self.weights.update_image(data)


class LayerContainer(QtWidgets.QGroupBox):
    def __init__(self, layer, is_input=False, is_output=False, parent=None, label=''):
        super(LayerContainer, self).__init__(label, parent=parent)
        self.par = parent
        self.lay = QtWidgets.QHBoxLayout(self)
        self.lay.setSpacing(0)
        self.layer = layer
        self.neurons = []
        self.fwd_indicator = Indicator(self)
        self.fwd_indicator.setFixedSize(30, 30)
        self.bck_indicator = Indicator(self)
        self.bck_indicator.setFixedSize(30, 30)
        self.lay.addWidget(self.fwd_indicator)
        self.lay.setContentsMargins(2, 2, 2, 2)
        self.lay.setSpacing(10)
        self.lay.addStretch()

        for i in range(self.layer.nodes):
            neuron_type = InputNeuron if is_input else Neuron
            neuron = neuron_type(weights=self.layer.w[:, i], index=i, parent=self)
            self.neurons.append(neuron)
            self.lay.addWidget(neuron)
        self.lay.addStretch()
        self.bias = Neuron(weights=self.layer.b, index=-1, parent=self, is_bias=True)
        self.bias.indicator.set_color(255, 255, 255, 'bias')
        self.lay.addWidget(self.bias)
        self.lay.addWidget(self.bck_indicator)

    def layer_changed(self):
        for i, neuron in enumerate(self.neurons):
            a = self.layer.a[i]
            if a.ndim > 0:
                a = np.mean(a)
            val = to_int(a*1.5, base=0, scaling=1)
            neuron.indicator.set_color(val, val, val)
            neuron.indicator.set_activation(str(round(a, 2)))
            # neuron.activation_edit.setText(str(round(a, 2)))

    def activate(self, forward=True, keep_lit=False):
        if forward:
            self.fwd_indicator.light('g', keep_lit)
        else:
            self.bck_indicator.light('r', keep_lit)
        
    def dropout(self, index, status):
        self.layer.mask[index] = 1 if status else 0
        self.layer.forward_pass(self.layer.prev_layer.a)
        self.par.forward_pass()


class DataContainer(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super(DataContainer, self).__init__('Input data', parent=parent)
        self.data_shape = ()
        self.par = parent
        self.data = []
        self.labels = []
        self.t_data = []
        self.t_labels = []

        self.display = ImageDisplay(self)
        self.slider = ParameterSpinBox(self, 0, 'Sample / Batch #')
        self.slider.set_ranges(0, 9999)
        self.slider.slider.sliderMoved[int].connect(self.change_image)

        lay = self.lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.display)
        lay.addWidget(self.slider)
        self.setMinimumSize(400, 300)

    def change_image(self, value):
        if not self.par.training:
            index = value
            data = self.x_[:, index]
            label = self.y_[index, :]
            d = np.flip(np.rot90(data.reshape(*self.data_shape), k=3), axis=1)
            self.display.update_image(d)
            self.par.forward_pass(data, label)

    def set_image(self, image):
        self.display.update_image(image)

    def load_dataset(self, dataset):
        if dataset == "MNIST":
            self.data_shape = (28, 28)
            (data, labels), (t_data, t_labels) = tf.keras.datasets.mnist.load_data()
            n_samples, h, w = data.shape
            n_samples_t, _, _ = t_data.shape
            data = np.reshape(data, [n_samples, h * w]).T/255.
            t_data = np.reshape(t_data, [n_samples_t, h * w]).T/255.

        elif dataset == "CIFAR10":
            (data, labels), (t_data, t_labels) = tf.keras.datasets.cifar10.load_data()
            self.data_shape = (32, 32, 3)
            n_samples, h, w, c = data.shape
            n_samples_t, _, _, _ = t_data.shape
            data = np.reshape(data, [n_samples, h * w * c]).T/255.
            t_data = np.reshape(t_data, [n_samples_t, h * w * c]).T/255.
            
        labels = one_hot(labels)
        t_labels = one_hot(t_labels)
        self.data = data
        self.labels = labels
        self.x_ = t_data
        self.y_ = t_labels
        self.display.update_image(self.data[:, 0].reshape(*self.data_shape))
    

class Control(QtWidgets.QGroupBox):
    def __init__(self,  parent=None):
        super(Control, self).__init__('Control', parent=parent)
        self.setMaximumWidth(300)
        self.par = parent

        # self.dataset_selection = LabeledComboBox(self, 'Dataset: ', ['MNIST', 'CIFAR10'])
        # self.dataset_selection.combo.currentTextChanged[str].connect(self.par.data_container.load_dataset)

        self.train_button = ToggleButton(self.par, ['Train', 'Stop'], 'start_training', ['Train on batches of samples.', 'Stop training.'], text=True)

        self.run_button = ToggleButton(self.par, ['Loop', 'Stop'], 'continuous_cascade', ['Continuously performs a forward and backward pass on a single example.', 'Stop'], text=True)

        self.cascade_button = QtWidgets.QPushButton('Cascade')
        self.cascade_button.clicked.connect(self.par.start_cascade)
        self.cascade_button.setStatusTip('Perform a forward and backward pass on a single example.')

        self.single_step_button = QtWidgets.QPushButton('Single step')
        self.single_step_button.clicked.connect(self.par.cascade)
        self.single_step_button.setStatusTip('Advance one layer in the forward/backward pass.')

        self.reset_button = QtWidgets.QPushButton('Reset weights')
        self.reset_button.clicked.connect(self.par.reset)
        self.reset_button.setStatusTip('Reset the weights to random values.')

        self.accuracy_button = QtWidgets.QPushButton('Test accuracy')
        self.accuracy_button.clicked.connect(self.par.accuracy)
        self.acc_box = QtWidgets.QLineEdit()
        self.accuracy_button.setStatusTip('Evaluate the accuracy on the test set (10000 samples).')

        self.lr_input = QtWidgets.QLineEdit()
        self.lr_input.setText(str(parent.model.optimizer.lr))
        self.lr_input.textChanged.connect(self.update_lr)

        self.animation_level_selection = LabeledComboBox(self, 'Animations: ', ['0', '1', '2', '3'])
        self.animation_level_selection.combo.currentTextChanged.connect(self.set_animations)

        self.batch_combo = LabeledComboBox(self, 'Batch Size: ', ['4', '16', '64', '256', '1024'])
        self.batch_combo.combo.currentTextChanged.connect(self.set_batch)

        # self.backprop_checkbox = ParameterCheckBox('Backpropagate', self, self.enable_backprop)
        # self.backprop_checkbox.check.setChecked(True)

        self.cascade_delay_input = QtWidgets.QLineEdit()
        self.cascade_delay_input.setText(str(parent.delay))
        self.cascade_delay_input.textChanged.connect(self.update_timer)

        lay = QtWidgets.QGridLayout(self)
        lay.addWidget(self.train_button, 1, 1, 1, 1)
        lay.addWidget(self.reset_button, 1, 2, 1, 1)
        lay.addWidget(QtWidgets.QLabel('Learning rate: '), 2, 1, 1, 1, alignment=QtCore.Qt.AlignRight)
        lay.addWidget(self.lr_input, 2, 2, 1, 1)
        lay.addWidget(self.batch_combo, 3, 1, 1, 2)
        lay.addWidget(self.accuracy_button, 4, 1, 1, 1)
        lay.addWidget(self.acc_box, 4, 2, 1, 1)
        # lay.addWidget(self.backprop_checkbox, 8, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel(' '), 5, 1, 1, 2)
        lay.addWidget(self.single_step_button, 6, 1, 1, 1)
        lay.addWidget(self.cascade_button, 6, 2, 1, 1)
        lay.addWidget(self.run_button, 7, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel('Cascade delay'), 8, 1, 1, 1, alignment=QtCore.Qt.AlignRight)
        lay.addWidget(self.cascade_delay_input, 8, 2, 1, 1)
        lay.addWidget(self.animation_level_selection, 9, 1, 1, 2)

        # lay.addWidget(self.dataset_selection, 10, 1, 1, 2)
        lay.setSpacing(5)
        lay.setContentsMargins(5, 5, 5, 5)

    def update_lr(self):
        lr = float(self.lr_input.text())
        self.par.model.optimizer.lr = lr

    def enable_backprop(self, status):
        self.par.backprop = status
    
    def set_animations(self):
        self.par.animations = int(self.animation_level_selection.combo.currentText())

    def set_batch(self):
        self.par.batch_size = int(self.batch_combo.combo.currentText())

    def update_timer(self):
        self.par.delay = int(self.cascade_delay_input.text())


class MLPUI(QtWidgets.QWidget):
    def __init__(self, parent=None, dataset='MNIST', arch=[8, 8]):
        super(MLPUI, self).__init__(parent=parent)
        print('Loading MNIST data...')
        self.backprop = False
        self.index = 0
        self.forward = True
        self.animations = 1
        self.training = False
        self.auto_cascade = False
        self.delay = 25

        self.batch_size = 16
        self.current_data = []
        self.current_labels = []
        self.continuous = False

        self.data_container = DataContainer(parent=self)
        self.data_container.load_dataset(dataset)
        self.model = MLP()
        self.errors = [0.5]
        self.t_errors = [0.5]
        self.acc = [0.]
        self.control_panel = Control(self)

        self.graph_container = QtWidgets.QGroupBox('Loss, Accuracy')
        self.graph_container.setMinimumHeight(200)
        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        self.graph = pg.PlotWidget()
        self.graph.setYRange(-.1, 1.1)
        # self.graph.setXRange(-.1, 1.1)
        # add grid
        self.graph.showGrid(x=True, y=True)
        graph_layout.addWidget(self.graph)
        self.graph_container.setMaximumWidth(300)

        self.timer = QtCore.QTimer(self, interval=1)
        self.timer.timeout.connect(self.train)
        self.cascade_timer = QtCore.QTimer(self, interval=1)
        self.cascade_timer.timeout.connect(self.cascade)

        self.model.build(
            input_vector=self.data_container.data[:, 0],
            nodes_hidden=arch,
            n_outputs=10)

        self.layers = [
            LayerContainer(self.model.layers[1], is_input=True, parent=self, label='Hidden layer #1'),
            LayerContainer(self.model.layers[2], parent=self, label='Hidden layer #2'),
            LayerContainer(self.model.layers[3], is_output=True, parent=self, label='Output layer'),
        ]

        self.lay = QtWidgets.QGridLayout(self)

        layer_cont = QtWidgets.QGroupBox(parent=self)
        lay_ = QtWidgets.QVBoxLayout(layer_cont)
        for layer in self.layers:
            lay_.addWidget(layer)
        
        layer_cont.setMinimumHeight(450)

        self.lay.addWidget(self.data_container, 1, 2, 2, 1)
        self.lay.addWidget(self.graph_container, 2, 1, 1, 1)
        self.lay.addWidget(self.control_panel, 1, 1, 1, 1)
        self.lay.addWidget(layer_cont, 1, 3, 2, 1)
        self.lay.setColumnMinimumWidth(1, 260)
        self.lay.setColumnMinimumWidth(2, 450)
        self.data_container.change_image(0)
    
    def load_dataset(self):
        self.data_container.load_dataset(self.control.dataset_selenction.text())
        # self. 
    #     dataset = self.control.dataset_selection.text()
    #     if dataset == "MNIST":
    #         (data, labels), (t_data, t_labels) = tf.keras.datasets.mnist.load_data()


    def update_layers(self):
        for layer in self.layers:
            layer.layer_changed()

    def forward_pass(self, data=None, labels=None, index=None, backprop=False):
        if data is None:
            data = self.current_data
            labels = self.current_labels

        self.model(data)
        self.current_data = data
        self.current_labels = labels
        self.update_layers()
        self.highlight_class()
        if backprop:
            data = np.expand_dims(data, axis=1)
            labels = np.expand_dims(labels, axis=0)
            error = self.model.optimizer.backprop(data, labels)
            self.update_weights()

    def highlight_class(self):
        index = np.argmax(self.model.layers[-1].a)
        correct = True if index == np.argmax(self.current_labels) else False
        self.layers[-1].neurons[index].indicator.highlight(correct=correct)

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                weights = layer.layer.w[:, j]
                if i == 0: 
                    weights = weights.reshape(*self.data_container.data_shape)
                neuron.set_weights(weights)
            layer.bias.set_weights(layer.layer.b)
            print(layer.layer.b)

    def start_training(self, status):
        if status:
            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.indicator.reset()
            n_batches = self.data_container.data.shape[1] // self.batch_size
            self.data_container.slider.slider.setRange(0, n_batches)
            self.batch = 0
            self.training = True
            self.timer.start(1)
        else:
            self.timer.stop()
            self.training = False
            self.data_container.slider.slider.setRange(0, self.data_container.x_.shape[1])

    def train(self):
        self.batch += 1

        if self.batch > self.data_container.data.shape[1]//self.batch_size:
            self.batch = 0

        self.data_container.slider.slider.setValue(self.batch)
        data, labels = get_batch(self.data_container.data, self.data_container.labels, self.batch_size)
        image = data.reshape(*self.data_container.data_shape, self.batch_size)
        n = int(np.sqrt(self.batch_size))

        cols = []
        data_dim = len(list(self.data_container.data_shape))
        for i in range(n):
            row = []
            for j in range(n):
                if data_dim > 2:
                    row.append(image[:, :, :, (i*n)+j])
                else:
                    row.append(image[:, :, (i*n)+j])
            cols.append(np.hstack(row))
        image = np.vstack(cols)
        image = np.flip(np.rot90(image, k=3), axis=1)

        if self.animations > 0:
            self.data_container.set_image(image)
        error = self.model.optimizer.backprop(data, labels)
        self.errors.append(error)
        
        if self.animations >= 2:
            self.update_weights()
        
        if self.animations >= 3:
            for layer in self.layers:
                self.show_error(layer)
        
        if self.batch % 10 == 0:
            loss, acc = self.accuracy()
            self.acc.append(acc/100.)
            self.t_errors.append(loss)
            print('Accuracy: {:.2f}%'.format(acc))
        else:
            self.t_errors.append(self.t_errors[-1])
            self.acc.append(self.acc[-1])
        pen = pg.mkPen('b', width=2)
        linepen = pg.mkPen('r', width=2, style=QtGui.Qt.DashLine)
        seekpen = pg.mkPen('g', width=1)
        if self.animations == 0:
            if self.batch % 5 == 0:
                self.graph.getPlotItem().plot(self.errors, clear=True, pen=pen)
                self.graph.getPlotItem().plot(self.t_errors, pen=seekpen)
                self.graph.getPlotItem().plot(self.acc, pen=linepen)
        else:
            self.graph.getPlotItem().plot(self.errors, clear=True, pen=pen)
            self.graph.getPlotItem().plot(self.t_errors, pen=seekpen)
            self.graph.getPlotItem().plot(self.acc, pen=linepen)


    def start_cascade(self):

        self.auto_cascade = True
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.indicator.reset()
        self.forward = True
        self.index = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.indicator.reset()

        self.cascade_timer.start(self.delay)
        self.model(self.current_data)
        
    def cascade(self):
        
        if self.forward and self.index == 0:
            self.update_weights()
            self.update_weights()
            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.indicator.reset()
            # self.model(self.current_data)
            self.forward_pass()
            # for layer in self.layers:
            #     layer.activate()

        layer = self.layers[self.index]

        if self.forward:
            layer.activate(forward=True, keep_lit=not self.auto_cascade)
            layer.layer_changed()
            if self.index == len(self.layers)-1:
                self.forward = False
                data = np.expand_dims(self.current_data, axis=1)
                labels = np.expand_dims(self.current_labels, axis=0)
                error = self.model.optimizer.backprop(data, labels)
                self.highlight_class()
            else:
                self.index += 1
            
        else:
            layer.activate(forward=False, keep_lit=not self.auto_cascade)
            for j, neuron in enumerate(layer.neurons):
                weights = layer.layer.w[:, j]
                if self.index == 0:
                    weights = weights.reshape(*self.data_container.data_shape)
                    # for layer in self.layers:
                    #     for neuron in layer.neurons:
                    #         neuron.indicator.reset()
                        
                neuron.set_weights(weights)
            self.show_error(layer)

            if self.index == 0:
                if not self.continuous:
                    self.cascade_timer.stop()
                    self.auto_cascade = False
                self.forward = True
                self.forward_pass()
                # for layer in self.layers:
                #     layer.activate()

            else:
                self.index -= 1

    def reset(self):
        for layer in self.model.layers[1:]:
            layer.init_weights()

        self.forward_pass()
        self.update_weights()
        self.update_weights()
        self.errors = [0.5]
        self.t_errors = [0.5]
        self.acc = [0.]
        self.graph.getPlotItem().plot(self.errors, clear=True)
        self.graph.getPlotItem().plot(self.t_errors, pen=pg.mkPen('g'))

    def accuracy(self):
        loss, acc = self.model.optimizer.eval(self.data_container.x_, self.data_container.y_)
        self.control_panel.acc_box.setText(str(round(acc,2))+'%')
        return loss, acc

    def show_error(self, layer):
        for j, neuron in enumerate(layer.neurons):
            if np.mean(layer.layer.error[j]) >= 0:
                r, g, b = 50, 50, 50
                text = u"\u25BC"
            else:
                r, g, b = 200, 200, 200 
                text = u"\u25B2"
            neuron.indicator.set_color(r, g, b, text=text)

    def continuous_cascade(self, status):
        self.continuous = status
        if status:
            self.start_cascade()