import tkinter as tk
from tkinter import ttk

class Dialog(tk.Toplevel):

    def __init__(self, parent, title = None, literals=[]):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        self.configure(background=parent.controller._get_bg_color())

        if title:
            self.title(title)
        if literals:
            self.literals = {}
            for littuple in literals:
                self.literals[littuple[0]] = littuple[1]

        self.parent = parent

        self.result = None
        
        self.test = "A"
        self.okbtn = None
        self.cnbtn = None
        self.packbtns = True

        # body = tk.Frame(self)
        body = ttk.Frame(self)

        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        self.okbtn.pack_forget()
        self.cnbtn.pack_forget()
        if self.packbtns == True:
            self.okbtn.pack(side=tk.LEFT, padx=5, pady=5)
            self.cnbtn.pack(side=tk.LEFT, padx=5, pady=5)
        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()

        self.wait_window(self)

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = ttk.Frame(self)

        self.okbtn = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        self.okbtn.pack(side=tk.LEFT, padx=5, pady=5)
        self.cnbtn = ttk.Button(box, text="Cancel", width=10, command=self.cancel)
        self.cnbtn.pack(side=tk.LEFT, padx=5, pady=5)

        # self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):
        print("ok")
        if not self.validate():
            print("invalid")
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()

    def cancel(self, event=None):

        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override

class DialogBlockNonGrab(tk.Toplevel):

    def __init__(self, parent, title = None, literals=[]):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)
        if literals:
            self.literals = {}
            for littuple in literals:
                self.literals[littuple[0]] = littuple[1]
        self.parent = parent
        self.result = None
        self.test = "A"
        self.okbtn = None
        self.cnbtn = None
        self.packbtns = True
        body = ttk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

    # construction hooks
    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden
        pass

    # standard button semantics

    def ok(self, event=None):
        print("ok")
        if not self.validate():
            print("invalid")
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()


    def cancel(self, event=None):
        self.validate()
        self.withdraw()
        self.update_idletasks()
        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override

class DialogNonBlock(tk.Toplevel):

    def __init__(self, parent, title = None, literals=[]):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)
        if literals:
            self.literals = {}
            for littuple in literals:
                self.literals[littuple[0]] = littuple[1]

        self.parent = parent

        self.result = None
        
        self.test = "A"
        self.okbtn = None
        self.cnbtn = None
        self.packbtns = True

        body = ttk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        self.okbtn.pack_forget()
        self.cnbtn.pack_forget()
        if self.packbtns == True:
            self.okbtn.pack(side=tk.LEFT, padx=5, pady=5)
            self.cnbtn.pack(side=tk.LEFT, padx=5, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.ok)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        # self.initial_focus.focus_set()

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = ttk.Frame(self)

        self.okbtn = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        self.okbtn.pack(side=tk.LEFT, padx=5, pady=5)
        self.cnbtn = ttk.Button(box, text="Cancel", width=10, command=self.cancel)
        self.cnbtn.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):
        print("ok")
        if not self.validate():
            print("invalid")
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()

    def cancel(self, event=None):

        # put focus back to the parent window
        self.parent.focus_set()
        self.literals["updatable_frame"].delete_settings()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override

class DialogNonBlockMax(tk.Toplevel):

    def __init__(self, parent, title = None, literals=[]):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        if title:
            self.title(title)
        if literals:
            self.literals = {}
            for littuple in literals:
                self.literals[littuple[0]] = littuple[1]
        self.parent = parent
        self.result = None
        self.test = "A"
        self.okbtn = None
        self.cnbtn = None
        self.packbtns = True
        self.geometryx = parent.winfo_rootx()+50
        self.geometryy = parent.winfo_rooty()+50
        body = ttk.Frame(self)
        self.initial_focus = self.body(body)
        # body.pack(padx=5, pady=5)
        body.grid(row=0, column=0, sticky=tk.NSEW)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.resizable(False, False)
        
        self.protocol("WM_DELETE_WINDOW", self.ok)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    #
    # standard button semantics

    def ok(self, event=None):
        print("ok")
        if not self.validate():
            print("invalid")
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()

    def cancel(self, event=None):

        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override