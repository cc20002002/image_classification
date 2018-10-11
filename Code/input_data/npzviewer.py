#! /usr/bin/python
from __future__ import print_function
import sys,os
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

class MyWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="NPZ Viewer")
        self.set_border_width(3)

        self.notebook = Gtk.Notebook()
        self.add(self.notebook)

        ''' The code is supposed to be executed as "python npzviewer.py <npz file>""
        The npzviewer.desktop file lists the executable as "exec=<path to npzviewer.py> %f",
        indicating that the filename is the first argument.
        '''
        if len(sys.argv)>1:
            try:
                with np.load(sys.argv[1]) as f:
                    self.arrays = dict(f.items())
                    self.set_title(os.path.basename(sys.argv[1]))
            except IOError:
                print("Could not read npz file")
                exit()
        else:
            print("No file specified")
            exit()

        self.pages = []
        for k in self.arrays:
            page = Gtk.Box()
            page.set_border_width(10)
            arr = np.atleast_2d(self.arrays[k])
            self.liststore = Gtk.ListStore(*[str]*arr.shape[1])

            for i in arr:
                self.liststore.append(["{:.1f}".format(j) for j in i.tolist()])

            self.tree = Gtk.TreeView(self.liststore)
            renderer = Gtk.CellRendererText()
            column = Gtk.TreeViewColumn("{} by {} array".format(*arr.shape), renderer, text=0)
            column.set_spacing(5)
            for colno in xrange(1,arr.shape[1]):
                renderer = Gtk.CellRendererText()
                column.pack_start(renderer,True)
                column.add_attribute(renderer,"text",colno)

            self.tree.append_column(column)

            self.scrollTree = Gtk.ScrolledWindow()
            self.scrollTree.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            self.scrollTree.set_min_content_height(100)
            self.scrollTree.set_min_content_width(300)
            self.scrollTree.add(self.tree)
            page.pack_start(self.scrollTree,True,True,True)

            self.show_all()
            self.notebook.append_page(page, Gtk.Label(k))
            self.pages.append(page)

win = MyWindow()
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()
