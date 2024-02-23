"""
＋つけたい機能
・反転
・ディレクトリ選択ボタン
・進めるボタン
・初期位置
・色表示選択
"""
from log import Log

import os
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk


class KifuViewer(tk.Frame):
    """ ガイスター棋譜閲覧アプリ

    """
    BAN_X = 6   
    BAN_Y = 6
    MASU_SIZE = 80 
    BAN_SIZE_X = BAN_X*MASU_SIZE
    BAN_SIZE_Y = BAN_Y*MASU_SIZE

    def __init__(self, master):
        super().__init__(master)
        self.log = None
        self.dir_path = './'
        self.master.geometry('1000x660')

        self.master.resizable(width=False, height=False)
        self.master.title('Geister Kifu Viewer')

        self.init_widgets()
        self.draw_file_name()
        self.draw_ban()

    def init_widgets(self):
        self.label = tk.Label(self.master, text="File Name : ")     # 開いているファイル名を表示するlabel
        self.viewer = tk.Canvas(self.master, width=self.BAN_SIZE_X, height=self.BAN_SIZE_Y, bg='black') # 盤面表示用キャンバス


        # フォルダー選択
        self.button = tk.Button(self.master, text='Open Folder', width=24, height = 2, activebackground='#aaaaaa')
        self.button.bind('<ButtonPress>', self.folder_dialog)
        self.button.place(x=650, y=5)


        # ファイル選択
        self.files_frame = tk.Frame(self.master)    # 開いているフォルダー配下のファイルを選択するFrame
        self.files_frame.place(x=650, y=40)
        self.files_listbox = tk.Listbox(self.files_frame, width=40, height=15)
        # files_file = [ f for f in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, f)) ]
        # print(files_file)
        # for name in files_file:
        #     if(name.startswith('log') and name.endswith('.txt')):
        #         self.files_listbox.insert(tk.END, name)
        self.files_scroll_y = tk.Scrollbar(self.files_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        self.files_listbox['yscrollcommand'] = self.files_scroll_y.set
        self.files_scroll_x = tk.Scrollbar(self.files_frame, orient=tk.HORIZONTAL, command=self.files_listbox.xview)
        self.files_listbox['xscrollcommand'] = self.files_scroll_x.set
        self.files_listbox.grid(row=0, column=0)
        self.files_scroll_y.grid(row=0, column=1, sticky='nsw')
        self.files_scroll_x.grid(row=1, column=0, sticky='new')
        self.files_listbox.bind('<<ListboxSelect>>', lambda e: self.select_file())

        # 指し手選択
        self.moves_frame = tk.Frame(self.master)
        self.moves_frame.place(x=650, y=320)
        self.moves_listbox = tk.Listbox(self.moves_frame, width=40, height=15)
        self.moves_scroll_y = tk.Scrollbar(self.moves_frame, orient=tk.VERTICAL, command=self.moves_listbox.yview)
        self.moves_listbox['yscrollcommand'] = self.moves_scroll_y.set
        self.moves_listbox.grid(row=0, column=0)
        self.moves_scroll_y.grid(row=0, column=1, sticky='nsw')
        # if self.dir_path != './':
        self.moves_listbox.bind('<<ListboxSelect>>', lambda e: self.select_move())

    def draw_file_name(self):
        self.label.place(x=20, y=20)

    def draw_ban(self):
        # for i in range(7):
        #     self.viewer.create_line(i*100, 0, i*100, self.BAN_SIZE_Y)
        #     self.viewer.create_line(0, i*100, self.BAN_SIZE_X, i*100)
        # self.viewer.create_rectangle(0, 0, self.MASU_SIZE, self.MASU_SIZE, fill='green')
        for y in range(6):
            for x in range(6):
                sx, sy = x*self.MASU_SIZE, y*self.MASU_SIZE
                ex, ey = (x+1)*self.MASU_SIZE, (y+1)*self.MASU_SIZE
                # self.viewer.create_rectangle(x*self.MASU_SIZE, y*self.MASU_SIZE, (x+1)*self.MASU_SIZE, (y+1)*self.MASU_SIZE, fill='white')
                self.viewer.create_rectangle(sx, sy, ex, ey, fill='white')
                if self.log != None:
                    name = self.log.get_board_index(self.sel_move[0])[y][x]
                    if name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                        if name in self.log.red_pos0:
                            self.viewer.create_polygon((sx+ex)/2, sy, sx, ey, ex, ey, fill = 'red')
                        else:
                            self.viewer.create_polygon((sx+ex)/2, sy, sx, ey, ex, ey, fill = 'blue')
                    if name in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                        if name in self.log.red_pos1:
                            self.viewer.create_polygon(sx, sy, ex, sy, (sx+ex)/2, ey, fill = 'red')
                        else:
                            self.viewer.create_polygon(sx, sy, ex, sy, (sx+ex)/2, ey, fill = 'blue')
                    self.viewer.create_text((sx+ex)/2, (sy+ey)/2, text=name)
        self.viewer.place(x=20,y=40)    # 位置調整
        self.master.bind('<space>', self.move)

    def folder_dialog(self, event):
        viewer_path = os.path.abspath(os.path.dirname(__file__))
        folder_name = tkinter.filedialog.askdirectory(initialdir=viewer_path)
        if len(folder_name) == 0:
            self.dir_path = './'
        else:
            self.dir_path = folder_name + '/'
        self.open_folder()

    def open_folder(self):
        self.files_listbox.delete(0, tk.END)
        files_file = [ f for f in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, f)) ]
        # print(files_file)
        for name in files_file:
            if(name.startswith('log') and name.endswith('.txt')):
                self.files_listbox.insert(tk.END, name)

    def select_file(self):
        file_name = self.files_listbox.get(self.files_listbox.curselection())
        self.label['text'] = 'File Name : ' + file_name
        self.open_kifu(self.dir_path+file_name)

    def select_move(self):
        self.sel_move = self.moves_listbox.curselection()
        self.draw_ban()
        # self.label['text'] = 'File Name : ' + sel_move[0]
        # print('bb')

    def open_kifu(self, path):
        self.log = Log(path)
        self.log.init_pieces_list()
        self.log.to_pieces_list()
        # with open(path, 'r', encoding='utf-8') as f:
        #     lines = f.read().split('\n')
        lines = self.log.get_moves()
        
        self.moves_listbox.delete(0, tk.END)
        for line in lines:
            self.moves_listbox.insert(tk.END, line)

    def move(self, event):
        self.viewer.move("id1",5,5)


def main():
    root=tk.Tk()
    app=KifuViewer(master=root)
    app.mainloop()


if __name__ == '__main__':
    main()
