""" ガイスターサーバーのログ操作

* 記載事項
* 記載事項2

Todo:
    * 使うために必要なことを記載 今だけ作成予定を記載
    * logから盤面復元
    * logから受信文字列復元
    * 受信文字列の操作は別クラスで

"""

import pprint # 配列表示に利用
import copy
import itertools

import piece


class Log:
    """ ガイスターサーバーのログ操作

    miyo/geister_serverが生成したガイスター対戦ログに対する操作を行う．
    
    Attributes:
        path (str): ログパス...ログファイルのパス
        red_pos0 (str): 先手赤駒列挙文字列
        red_pos1 (str): 後手赤駒列挙文字列
        moves (list[str]): move配列...読み込んだ棋譜を配列で管理（例 self.moves[0] == "0AN" ... player0の駒AがNorthに移動）
        winner (int): 読み込んだ棋譜の勝敗（0:player0勝 1:player1勝 2:引き分け -1:勝敗なし）
        moves_num (int): 読み込んだ棋譜の手数

    """

    def __init__(self, path: str):
        """ コンストラクタ

        読み込むログファイルのパスから初期化処理をする．

        Args:
            path (str): 読み込むログファイルのパス

        """
        self.init_all(path) # 初期化処理

    def init_all(self, path: str):
        """ 初期化処理
        
        各クラス変数を初期化，ログファイルのパスを保存し，ログファイルを読み込む．

        Args:
            path (str): 読み込むログファイルのパス

        """
        self.init_log()  # 各クラス変数を初期化
        self.set_path(path) # ログファイルのパスを保存
        # ログファイルを読み込む
        if(self.read_log()):
            return True
        else:
            return False

    def init_log(self):
        self.path = ''
        self.red_pos0 = ''
        self.red_pos1 = ''
        self.moves = []
        self.winner = -1
        self.moves_num = -1

    def init_pieces_list(self):
        self.pieces_list = [
                {
                'A': piece.Piece(1, 4, 'b'),
                'B': piece.Piece(2, 4, 'b'),
                'C': piece.Piece(3, 4, 'b'),
                'D': piece.Piece(4, 4, 'b'),
                'E': piece.Piece(1, 5, 'b'),
                'F': piece.Piece(2, 5, 'b'),
                'G': piece.Piece(3, 5, 'b'),
                'H': piece.Piece(4, 5, 'b'),
                'a': piece.Piece(4, 1, 'b'),
                'b': piece.Piece(3, 1, 'b'),
                'c': piece.Piece(2, 1, 'b'),
                'd': piece.Piece(1, 1, 'b'),
                'e': piece.Piece(4, 0, 'b'),
                'f': piece.Piece(3, 0, 'b'),
                'g': piece.Piece(2, 0, 'b'),
                'h': piece.Piece(1, 0, 'b'),
                },
                ]
        for r in self.red_pos0:
            self.pieces_list[0][r].set_color('r')
        for r in self.red_pos1:
            self.pieces_list[0][r].set_color('r')
        # for v in self.pieces_list.values():
        #     v.print_all()

    def init_boards(self):
        self.boards = [
                [
                ['x', 'h', 'g', 'f', 'e', 'x'],
                ['x', 'd', 'c', 'b', 'a', 'x'],
                ['x', 'x', 'x', 'x', 'x', 'x'],
                ['x', 'x', 'x', 'x', 'x', 'x'],
                ['x', 'x', 'x', 'x', 'x', 'x'],
                ['x', 'A', 'B', 'C', 'D', 'x'],
                ['x', 'E', 'F', 'G', 'H', 'x'],
                ]
                ]

    def set_path(self, path: str):
        """ pathを設定する

        Args:
            path (str): 読み込むログファイルのパス

        """
        self.path = path

    def read_log(self) -> bool:
        """ ログファイルを読み込む

        ログパスからログファイルを読みこみ，各クラス変数（red_pos0, red_pos1, moves, winner）に保存する．
        
        Returns:
            bool: エラーなく終局していたらTrue

        """
        moves = []
        winner = -1
        with open(self.path, 'r', encoding='utf-8') as f:    # logパスからlogファイルを読みこむ
            for l in f.read().split('\n'):    # logファイルを改行ごとに区切って処理
                if l != '': # 空行がたまにあるので消す
                    o = l.split(',')
                    if o[0].startswith('player='):
                        if o[1].startswith('SET:'):
                            if o[0] == 'player=0':
                                self.red_pos0 = o[1][4:]
                            elif o[0] == 'player=1':
                                self.red_pos1 = o[1][4:].lower()
                        elif o[1].startswith('MOV:'):
                            moves.append(f'{o[0][-1]}{o[1][-1]}{o[2][0]}')
                    elif o[0].startswith('winner='):
                        winner = int(o[0][-1])
                    else:
                        return False
        if winner == -1:
            return False 
        self.moves = moves
        self.winner = winner
        self.moves_num = len(moves)
        return True

    def get_red_pos0(self) -> str:
        """ 先手赤駒を取得

        先手赤駒を文字列の列挙で表す（例：BCDE ... B,C,D,Eが赤駒）

        Returns:
            str: 先手赤駒列挙文字列

        """
        return self.red_pos0

    def get_red_pos1(self) -> str:
        """ 後手赤駒を取得

        後手赤駒を文字列の列挙で表す（例：BCDE ... B,C,D,Eが赤駒）

        Returns:
            str: 後手赤駒列挙文字列

        """
        return self.red_pos1

    def get_moves(self) -> list[str]:
        """ moves配列を取得

        moves配列を返す

        Returns:
            list[str]: moves配列

        """
        return self.moves

    def get_move(self, index: int) -> str:
        """ movesを取得

        index手目のmoveを返す

        Args:
            index (int): index手目

        Returns:
            str: move

        """
        if index-1 >=0: # index-1が0より大きいならindex-1手目のmovesを返す
            return self.moves[index-1]
        return self.moves[0] # index-1が0より小さいときは0手目のmovesを返す

    def get_moves_index(self, index: int) -> list[str]:
        """ index手目までのmoves配列を取得

        index手目までのmoves配列を返す

        Args:
            index (int): index手目

        Returns:
            list[str]: moves配列

        """
        return self.moves[:index]


    def get_winner(self) -> int:
        """ 勝敗を取得

        勝敗をint型で取得
        
        Returns:
            int: 勝敗（0: 先手勝, 1: 後手勝, 2: 引分, -1: 勝敗不明）
        """
        return self.winner

    def is_end(self):
        if self.winner == -1:
            return False
        return True

    def get_moves_num(self) -> int:
        """ 総手数を取得

        総手数をint型で取得

        Returns:
            int: 総手数（初期化時は-1）
        """
        return self.moves_num

    def print_log(self):
        """ ログ情報を表示

        pprintでログ情報を整形して表示

        """
        print(f'先手赤駒位置：{self.red_pos0}')
        print(f'後手赤駒位置：{self.red_pos1}')
        print(f'---log---')
        pprint.pprint(self.moves)
        print(f'勝者：{self.winner}')
        print(f'総手数：{self.moves_num}')

    def to_boards(self) -> bool:
        """ ログから盤面を復元

        Returns:
            bool: 復元できたらTrue

        """
        return True

    def to_pieces_list(self) -> bool:
        """ ログから盤面を復元

        Returns:
            bool: 復元できたらTrue

        """
        # pieces = copy.deepcopy(self.pieces_list[-1])
        for move in self.moves:
            pieces = copy.deepcopy(self.pieces_list[-1])
            if move[0] == '0':
                pieces[move[1]].move(int(move[0]), move[2])
                pos = pieces[move[1]].get_pos()
                for k, v in pieces.items():
                    if v.get_pos() == pos:
                        pieces[k].set_pos(9, 9)
                if pos[0] == -1 or pos[0] == 6:
                    pieces[move[1]].set_pos(8, 8)
                else:
                    pieces[move[1]].set_pos(pos[0], pos[1])

            elif move[0] == '1':
                pieces[move[1].lower()].move(int(move[0]), move[2])
                pos = pieces[move[1].lower()].get_pos()
                for k, v in pieces.items():
                    if v.get_pos() == pos:
                        pieces[k].set_pos(9, 9)
                if pos[0] == -1 or pos[0] == 6:
                    pieces[move[1].lower()].set_pos(8, 8)
                else:
                    pieces[move[1].lower()].set_pos(pos[0], pos[1])
            self.pieces_list.append(pieces)

        return True

    def print_pieces_list(self):
        for pieces in self.pieces_list:
            print('---pieces---')
            for k, v in pieces.items():
                print(f'{k}: {v.get_pos()}, {v.get_color()}')
            print('------------')
            print()

    def print_board(self):
        for pieces in self.pieces_list:
            board = [['x' for _ in range(6)] for _ in range(6)]
            for k, v in pieces.items():
                idx = v.get_pos()
                if idx[0] == 9 or idx[0] == 8:
                    pass
                else:
                    board[idx[1]][idx[0]] = k
            pprint.pprint(board)
                
    def get_board_index(self, idx):
        board = [['x' for _ in range(6)] for _ in range(6)]
        for k, v in self.pieces_list[idx].items():
            idx = v.get_pos()
            if idx[0] == 9 or idx[0] == 8:
                pass
            else:
                board[idx[1]][idx[0]] = k
        return board

    def get_board_last(self):
        board = [['x' for _ in range(6)] for _ in range(6)]

        for k, v in self.pieces_list[-1].items():
            idx = v.get_pos()
            if idx[0] == 9 or idx[0] == 8:
                pass
            else:
                print(f'({idx[1]}, {idx[0]} -> {k})')
                board[idx[1]][idx[0]] = k
        return board

    def get_board(self, index: int) -> list[list[str]]:
        """ index手目の盤面を取得

        Args:
            index (int): index手目

        Returns:
            list[list[str]]: board

        """
        return self.boards[index]

    def get_boards(self) -> list[list[list[str]]]:
        """ 盤面配列を取得

        Returns:
            list[list[list[str]]]: boards配列

        """
        return self.boards

    def get_boards_index(self, index: int) -> list[list[list[str]]]:
        """ index手目までの盤面配列を取得

        Args:
            index (int): index手目

        Returns:
            list[list[list[str]]]: boards配列

        """
        return self.boards[:index]

    def get_board_index_1line(self, index: int):
        return list(itertools.chain.from_iterable(self.get_board_index(index)))

    def get_board_last_1line(self):
        return list(itertools.chain.from_iterable(self.get_board_last()))

    def get_colors_last(self) -> dict[str, str]:
        pieces = self.pieces_list[-1]
        enemy_colors = {}
        for n in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            enemy_piece = pieces[n]
            enemy_pos = enemy_piece.get_pos()
            if enemy_pos[0] == 8 or enemy_pos[0] == 9:
                enemy_colors[n] = enemy_piece.get_color()
            else:
                enemy_colors[n] = 'u'
        return enemy_colors


    # def to_sendstr(self, move_num):
    #     """ 任意の手数の受信文字列を復元する"""
    #     print('move_num番目の受信文字列')

    # def to_sendstr_list(self):
    #     """ 受信文字列を配列ですべて復元する"""

    def make_data(self):
        my_pos = self.get_red_pos0()
        data = list(my_pos) + self.moves
        return [data, list(self.red_pos1)]
       

def main():
    """ メイン関数 """
    log = Log('../log/Naotti_hayazashi/log/log-2023-10-31-01-38-22-946.txt')  # ログファイル読み込み
    # log.print_log() # ログ配列確認
    # print(log.make_data())
    # print(log.get_moves_index(0))
    log.init_pieces_list()
    log.to_pieces_list()
    log.print_pieces_list()
    # log.print_board()
    pprint.pprint(log.get_board_last())
    print(log.get_board_last_1line())
    pprint.pprint(log.get_colors_last())


if __name__ == '__main__':
    main()  # メイン関数を呼ぶ
