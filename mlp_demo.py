
from PySide6 import QtGui, QtWidgets
from interface import App
import argparse
from qt_material import apply_stylesheet

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MNIST')
    parser.add_argument('-a', '--arch', type=int, nargs='*', default=[8, 8])

    args = parser.parse_args()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Breeze')
    thisapp = App(dataset=args.dataset, arch=args.arch)
    thisapp.setWindowTitle("Introduction to Deep Learning 2023: Multilayer Perceptron")
    thisapp.resize(800, 800)
    font = QtGui.QFont('Monospace', 11)
    app.setFont(font)
    thisapp.show()
    sys.exit(app.exec())